"""Experimental cache policy, L2 prefetch and register limits for fused DP4A.

Derived from the selected norm/projection kernel; floating operation order and
quantized weights are retained. Each setting requires independent validation.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


SELECTED={'qkv':{'rows':8,'integer_groups':1,'integer_rows':1},
          'up':{'rows':32,'integer_groups':2,'integer_rows':4}}


@gluon.jit
def _round(x):
    return gl.inline_asm_elementwise('cvt.rni.s32.f32 $0,$1;',constraints='=r,f',args=[x],dtype=gl.int32,is_pure=True,pack=1)


@gluon.jit
def _dot(w,a,b):
    return gl.inline_asm_elementwise("""{
        .reg .b32 lo,hi;
        shl.b32 lo,$1,4;
        and.b32 lo,lo,0xf0f0f0f0;
        and.b32 hi,$1,0xf0f0f0f0;
        dp4a.s32.s32 $0,lo,$2,0;
        dp4a.s32.s32 $0,hi,$3,$0;
    }""",constraints='=r,r,r,r',args=[w,a,b],dtype=gl.int32,is_pure=True,pack=1)


@gluon.jit
def _projection(a,b,xscale,W,S,base,N:gl.constexpr,R:gl.constexpr,I:gl.constexpr,F:gl.constexpr,WC:gl.constexpr,SC:gl.constexpr,EP:gl.constexpr):
    ri=base+gl.arange(0,R,layout=gl.SliceLayout(1,gl.SliceLayout(2,I)))
    gi=gl.arange(0,128,layout=gl.SliceLayout(0,gl.SliceLayout(2,I)))
    c=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(0,I)))
    pos=gi[None,:,None]*4+gl.expand_dims(gl.expand_dims(c,0),0)
    w=gl.load(gl.cast(W,gl.pointer_type(gl.uint32))+ri[:,None,None]*512+pos,ri[:,None,None]<N,0,cache_modifier=WC,eviction_policy=EP)
    dot=_dot(w,a[None,:,:],b[None,:,:])
    sums=gl.convert_layout((gl.sum(dot,2)>>4).to(gl.float32),F)
    rf=base+gl.arange(0,R,layout=gl.SliceLayout(1,F))
    gf=gl.arange(0,128,layout=gl.SliceLayout(0,F))
    scale=gl.load(S+rf[:,None]*128+gf[None,:],rf[:,None]<N,0,cache_modifier=SC).to(gl.float32)
    return gl.sum(sums*(scale*xscale[None,:]),1)


@gluon.jit
def _prefetch(pointer,valid):
    gl.inline_asm_elementwise("""{
        .reg .pred p;
        setp.ne.s32 p,$2,0;
        @p prefetch.global.L2 [$1];
        mov.u32 $0,0;
    }""",constraints='=r,l,r',args=[pointer,valid.to(gl.int32)],dtype=gl.int32,is_pure=False,pack=1)


@gluon.jit
def _kernel(X,RES,NW,W,S,SUM,Y,OQ,OS,NY,NQ,NS,N:gl.constexpr,R:gl.constexpr,
            EPS:gl.constexpr,ADD:gl.constexpr,FUSED:gl.constexpr,IG:gl.constexpr,IR:gl.constexpr,
            DEBUG:gl.constexpr,WC:gl.constexpr,SC:gl.constexpr,EP:gl.constexpr,PF:gl.constexpr,SW:gl.constexpr):
    # Four warps are fixed because the normalization's FP32 order is part
    # of the tested arithmetic contract.
    V:gl.constexpr=gl.BlockedLayout([8],[32],[4],[0])
    block=gl.program_id(0)
    if SW>1:block=(block%SW)*(gl.num_programs(0)//SW)+block//SW
    if PF:
        P:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
        offset=gl.arange(0,R*2048//PF,layout=P)*PF
        address=gl.cast(W,gl.pointer_type(gl.uint8))+block*R*2048+offset
        valid=block*R+offset//2048<N
        _prefetch(address,valid)
        if FUSED:_prefetch(address+N*2048,valid)
    i=gl.arange(0,4096,layout=V)
    x=gl.load(X+i).to(gl.float32)
    if ADD:
        x=(x+gl.load(RES+i).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
        gl.store(SUM+i,x,gl.program_id(0)==0)
    inv=gl.rsqrt(gl.sum(x*x,0)/4096+EPS)
    normalized=((x*inv).to(gl.bfloat16).to(gl.float32)*gl.load(NW+i).to(gl.float32)).to(gl.bfloat16)
    grouped=gl.reshape(normalized.to(gl.float32),(128,32))
    scale=gl.maximum(gl.div_rn(gl.max(gl.abs(grouped),1),127.0),1e-8)
    q=_round(grouped*gl.div_rn(1.0,scale)[:,None])
    if DEBUG:
        gl.store(NY+i,normalized,gl.program_id(0)==0)
        gl.store(NQ+i,gl.reshape(q,(4096,)).to(gl.int8),gl.program_id(0)==0)
        gi=gl.arange(0,128,layout=scale.type.layout)
        gl.store(NS+gi,scale,gl.program_id(0)==0)
    q4=gl.reshape(q,(1024,4))
    c=gl.arange(0,4,layout=gl.SliceLayout(0,q4.type.layout))
    words=gl.sum((q4&255)<<(c[None,:]*8),1)
    a,b=gl.split(gl.reshape(words,(512,2)))
    I:gl.constexpr=gl.BlockedLayout([1,IG,4],[1,32,1],[IR,4//IR,1],[2,1,0])
    F:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    a=gl.convert_layout(gl.reshape(a,(128,4)),gl.SliceLayout(0,I))
    b=gl.convert_layout(gl.reshape(b,(128,4)),gl.SliceLayout(0,I))
    scale=gl.convert_layout(scale,gl.SliceLayout(0,F))
    base=block*R
    value=_projection(a,b,scale,W,S,base,N*2 if FUSED else N,R,I,F,WC,SC,EP).to(gl.bfloat16)
    if FUSED:
        up=_projection(a,b,scale,W,S,base+N,N*2,R,I,F,WC,SC,EP).to(gl.bfloat16).to(gl.float32)
        gate=value.to(gl.float32)
        silu=(gate/(1+gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value=(silu*up).to(gl.bfloat16)
    O:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
    value=gl.convert_layout(value,O)
    rows=base+gl.arange(0,R,layout=O)
    gl.store(Y+rows,value,rows<N)
    if FUSED:
        output=gl.reshape(value.to(gl.float32),(R//32,32))
        output_scale=gl.maximum(gl.div_rn(gl.max(gl.abs(output),1),127.0),1e-8)
        quant=_round(output*gl.div_rn(1.0,output_scale)[:,None])
        gl.store(OQ+rows,gl.reshape(quant,(R,)).to(gl.int8),rows<N)
        groups=block*(R//32)+gl.arange(0,R//32,layout=output_scale.type.layout)
        gl.store(OS+groups,output_scale,groups<N//32)


def linear(x,residual,norm_weight,eps,w,s,*,fused=False,rows=4,integer_groups=1,integer_rows=1,debug=False,weight_cache="",scale_cache="",eviction="",prefetch=0,swizzle=1,max_registers=None,return_kernel=False):
    n2,k2=w.shape;n=n2//2 if fused else n2
    if x.numel()!=4096 or x.dtype!=torch.bfloat16 or k2!=2048 or n<=0 or norm_weight.numel()!=4096:
        raise ValueError('One BF16 K4096 activation row required')
    if integer_groups not in (1,2,4) or integer_rows not in (1,2,4) or rows not in ((32,64) if fused else (4,8,16)):
        raise ValueError('Unsupported fusion tile')
    if fused and n2%64:raise ValueError('Complete gate/up output quantization groups required')
    tensors=(x,norm_weight,w,s)+(() if residual is None else (residual,))
    if not x.is_cuda or any(t.device!=x.device or not t.is_contiguous() for t in tensors):raise ValueError('Contiguous CUDA inputs required')
    if norm_weight.dtype!=torch.bfloat16 or residual is not None and (residual.dtype!=x.dtype or residual.shape!=x.shape) or w.dtype!=torch.uint8 or s.shape!=(n2,128) or s.dtype not in (torch.bfloat16,torch.float32):
        raise ValueError('Invalid norm, residual or grouped weight buffers')
    if weight_cache not in ('','.ca','.cg') or scale_cache not in ('','.ca','.cg') or eviction not in ('','evict_first','evict_last') or prefetch not in (0,128,256,512) or swizzle not in (1,2,4,8,16,32) or triton.cdiv(n,rows)%swizzle:
        raise ValueError('Invalid cache, prefetch or grid layout')
    if max_registers is not None and (max_registers<24 or max_registers>256):raise ValueError('Invalid register limit')
    if weight_cache and eviction:raise ValueError('This CUDA toolchain rejects explicit cache operators combined with eviction priority')
    summed=torch.empty_like(x) if residual is not None else x
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    oq=torch.empty(n if fused else 0,device=x.device,dtype=torch.int8);os=torch.empty(n//32 if fused else 0,device=x.device)
    ny=torch.empty_like(x) if debug else torch.empty(0,device=x.device,dtype=x.dtype)
    nq=torch.empty(4096 if debug else 0,device=x.device,dtype=torch.int8);ns=torch.empty(128 if debug else 0,device=x.device)
    compiled=_kernel[(triton.cdiv(n,rows),)](x,residual,norm_weight,w,s,summed,y,oq,os,ny,nq,ns,n,rows,eps,residual is not None,fused,integer_groups,integer_rows,debug,weight_cache,scale_cache,eviction,prefetch,swizzle,num_warps=4,maxnreg=max_registers)
    result=(summed,y,(oq,os)) if fused else (summed,y)
    value=(result,(ny,nq,ns)) if debug else result
    return (value,compiled) if return_kernel else value


def enable_qkv_load_policy(llm):
    """Select the validated G32 QKV load policy before graph capture."""
    if llm.graph is not None or llm.prefill_graphs:
        raise RuntimeError('Select QKV load policy before graph capture')
    if 'qkv' not in getattr(llm,'norm_projection_fused',{}):
        raise ValueError('The selected G32 QKV normalization fusion is required')
    if getattr(llm,'group128_norm_fused',None):
        raise ValueError('This policy is qualified for the G32 QKV consumer')
    llm.qkv_load_policy=True
