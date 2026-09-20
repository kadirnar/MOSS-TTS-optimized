"""PDL with independent output/down projection loads before the grid wait."""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from .pdl_control import wait, trigger


@gluon.jit
def _dot(w,p,valid):
    return gl.inline_asm_elementwise("""{
        .reg .b32 a,b,lo,hi;
        .reg .pred valid;
        mov.b32 a,0; mov.b32 b,0;
        setp.ne.s32 valid,$3,0;
        @valid ld.global.v2.u32 {a,b},[$2];
        shl.b32 lo,$1,4;
        and.b32 lo,lo,0xf0f0f0f0;
        and.b32 hi,$1,0xf0f0f0f0;
        dp4a.s32.s32 $0,lo,a,0;
        dp4a.s32.s32 $0,hi,b,$0;
    }""",constraints='=r,r,l,r',args=[w,p,valid.to(gl.int32)],dtype=gl.int32,is_pure=True,pack=1)


@gluon.jit
def _projection(Q,XS,W,S,base,N:gl.constexpr,K:gl.constexpr,BG:gl.constexpr,
                R:gl.constexpr,I:gl.constexpr,F:gl.constexpr,PW,PS,PRE:gl.constexpr):
    ri=base+gl.arange(0,R,layout=gl.SliceLayout(1,gl.SliceLayout(2,I)))
    gi=gl.arange(0,BG,layout=gl.SliceLayout(0,gl.SliceLayout(2,I)))
    c=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(0,I)))
    pos=gi[None,:,None]*4+gl.expand_dims(gl.expand_dims(c,0),0)
    if PRE&2:w=PW
    else:w=gl.load(gl.cast(W,gl.pointer_type(gl.uint32))+ri[:,None,None]*(K//8)+pos,
              (ri[:,None,None]<N)&(pos<K//8),0)
    dot=_dot(w,gl.cast(Q,gl.pointer_type(gl.uint64))+pos,pos<K//8)
    isum=gl.sum(dot,2)>>4
    sums=gl.convert_layout(isum.to(gl.float32),F)
    rf=base+gl.arange(0,R,layout=gl.SliceLayout(1,F))
    gf=gl.arange(0,BG,layout=gl.SliceLayout(0,F))
    if PRE&1:scale=PS
    else:scale=gl.load(S+rf[:,None]*(K//32)+gf[None,:],(rf[:,None]<N)&(gf[None,:]<K//32),0).to(gl.float32)
    xs=gl.load(XS+gf,gf<K//32,0)
    return gl.sum(sums*(scale*xs[None,:]),1)


@gluon.jit
def _kernel(Q,XS,W,S,Y,OQ,OS,N:gl.constexpr,K:gl.constexpr,BG:gl.constexpr,R:gl.constexpr,
            FUSED:gl.constexpr,WARPS:gl.constexpr,IG:gl.constexpr,IROWS:gl.constexpr,PDL:gl.constexpr,TRIGGER:gl.constexpr,PRE:gl.constexpr):
    I:gl.constexpr=gl.BlockedLayout([1,IG,4],[1,32,1],[IROWS,WARPS//IROWS,1],[2,1,0])
    F:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[WARPS if K==4096 else WARPS//2,1 if K==4096 else 2],[1,0])
    base=gl.program_id(0)*R
    gl.static_assert(not FUSED)
    pw=0;ps=0
    if PRE&2:
        ri=base+gl.arange(0,R,layout=gl.SliceLayout(1,gl.SliceLayout(2,I)))
        gi=gl.arange(0,BG,layout=gl.SliceLayout(0,gl.SliceLayout(2,I)))
        c=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(0,I)))
        pos=gi[None,:,None]*4+gl.expand_dims(gl.expand_dims(c,0),0)
        pw=gl.load(gl.cast(W,gl.pointer_type(gl.uint32))+ri[:,None,None]*(K//8)+pos,(ri[:,None,None]<N)&(pos<K//8),0)
    if PRE&1:
        rf=base+gl.arange(0,R,layout=gl.SliceLayout(1,F))
        gf=gl.arange(0,BG,layout=gl.SliceLayout(0,F))
        ps=gl.load(S+rf[:,None]*(K//32)+gf[None,:],(rf[:,None]<N)&(gf[None,:]<K//32),0).to(gl.float32)
    if PDL:wait()
    if PDL and TRIGGER==1:trigger()
    value=_projection(Q,XS,W,S,base,N,K,BG,R,I,F,pw,ps,PRE).to(gl.bfloat16)
    if FUSED:
        up=_projection(Q,XS,W,S,base+N,N*2,K,BG,R,I,F).to(gl.bfloat16).to(gl.float32)
        gate=value.to(gl.float32)
        silu=(gate/(1+gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value=(silu*up).to(gl.bfloat16)
    if PDL and TRIGGER==3:trigger()
    V:gl.constexpr=gl.BlockedLayout([1],[32],[WARPS],[0])
    value=gl.convert_layout(value,V)
    rows=base+gl.arange(0,R,layout=V)
    gl.store(Y+rows,value,rows<N)
    if FUSED:
        grouped=gl.reshape(value.to(gl.float32),(R//32,32))
        scale=gl.maximum(gl.div_rn(gl.max(gl.abs(grouped),1),127.0),1e-8)
        inv=gl.div_rn(1.0,scale)
        product=grouped*inv[:,None]
        quant=gl.inline_asm_elementwise('cvt.rni.s32.f32 $0,$1;',constraints='=r,f',args=[product],dtype=gl.int32,is_pure=True,pack=1)
        gl.store(OQ+rows,gl.reshape(quant,(R,)).to(gl.int8),rows<N)
        groups=gl.program_id(0)*(R//32)+gl.arange(0,R//32,layout=scale.type.layout)
        gl.store(OS+groups,scale,groups<N//32)


def linear(x,w,s,*,fused=False,rows=4,warps=4,integer_groups=1,integer_rows=1,prequantized,pdl=True,trigger_mode=1,return_kernel=False,prefetch=3):
    if fused or prefetch not in (0,1,2,3):raise ValueError("Unfused projection and valid preload mode required")
    if trigger_mode not in (0,1,3):raise ValueError("Invalid PDL trigger mode")
    n2,k2=w.shape;k=k2*2;n=n2//2 if fused else n2
    if k not in (4096,12288) or rows not in (2,4,8,16,32,64) or warps not in (2,4,8) or integer_groups not in (1,2,4) or integer_rows not in (1,2,4,8) or warps%integer_rows:
        raise ValueError('Invalid explicit projection layout')
    if x.dtype!=torch.bfloat16 or x.numel()!=k or fused and (k!=4096 or rows%32 or n%32):
        raise ValueError('Specialized one-row projection required')
    q,sx=prequantized
    if not x.is_cuda or any(t.device!=x.device or not t.is_contiguous() for t in (x,w,s,q,sx)):
        raise ValueError('Contiguous CUDA tensors on the same device required')
    if w.dtype!=torch.uint8 or s.shape!=(n2,k//32) or s.dtype not in (torch.bfloat16,torch.float32) or q.dtype!=torch.int8 or q.numel()!=k or sx.dtype!=torch.float32 or sx.numel()!=k//32:
        raise ValueError('Invalid packed weight/activation buffers')
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    oq=torch.empty(n if fused else 0,device=x.device,dtype=torch.int8);os=torch.empty(n//32 if fused else 0,device=x.device)
    compiled=_kernel[(triton.cdiv(n,rows),)](q,sx,w,s,y,oq,os,n,k,triton.next_power_of_2(k//32),rows,fused,warps,integer_groups,integer_rows,pdl,trigger_mode,prefetch,num_warps=warps,launch_pdl=pdl)
    value=(y,(oq,os)) if fused else y
    return (value,compiled) if return_kernel else value
