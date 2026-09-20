"""Experimental G64 weights with unchanged G32 activations and SM90 PDL.

FACTOR=0 uses the nominal selected expression with each weight scale repeated
across two activation groups; compiler lowering is not guaranteed bit-identical. FACTOR=1 factors that common scale after pairing
activation-weighted integer sums; its changed FP32 order is explicitly tested.
Neither mode reduces the 32 acoustic codebooks.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from .pdl_control import wait,trigger
from .bulk_address import _hint
from .dp4a_norm_pdl import _round,_dot
from .dp4a_layout_pdl_prefetch import _dot as _dot_memory


@gluon.jit
def _finish(sums,xscale,S,base,N:gl.constexpr,K:gl.constexpr,BG:gl.constexpr,R:gl.constexpr,F:gl.constexpr,FACTOR:gl.constexpr):
    if FACTOR:
        paired=gl.reshape(sums*xscale[None,:],(R,BG//2,2))
        combined=gl.sum(paired,2)
        L:gl.constexpr=combined.type.layout
        rows=base+gl.arange(0,R,layout=gl.SliceLayout(1,L))
        groups=gl.arange(0,BG//2,layout=gl.SliceLayout(0,L))
        scales=gl.load(S+rows[:,None]*(K//64)+groups[None,:],(rows[:,None]<N)&(groups[None,:]<K//64),0).to(gl.float32)
        return gl.sum(combined*scales,1)
    else:
        rows=base+gl.arange(0,R,layout=gl.SliceLayout(1,F))
        groups=gl.arange(0,BG,layout=gl.SliceLayout(0,F))
        scales=gl.load(S+rows[:,None]*(K//64)+(groups//2)[None,:],(rows[:,None]<N)&(groups[None,:]<K//32),0).to(gl.float32)
        return gl.sum(sums*(scales*xscale[None,:]),1)


@gluon.jit
def _projection(a,b,xscale,W,S,base,N:gl.constexpr,R:gl.constexpr,I:gl.constexpr,F:gl.constexpr,FACTOR:gl.constexpr):
    rows=base+gl.arange(0,R,layout=gl.SliceLayout(1,gl.SliceLayout(2,I)))
    groups=gl.arange(0,128,layout=gl.SliceLayout(0,gl.SliceLayout(2,I)))
    c=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(0,I)))
    pos=groups[None,:,None]*4+gl.expand_dims(gl.expand_dims(c,0),0)
    w=gl.load(gl.cast(W,gl.pointer_type(gl.uint32))+rows[:,None,None]*512+pos,rows[:,None,None]<N,0)
    dot=_dot(w,a[None,:,:],b[None,:,:])
    sums=gl.convert_layout((gl.sum(dot,2)>>4).to(gl.float32),F)
    return _finish(sums,xscale,S,base,N,4096,128,R,F,FACTOR)


@gluon.jit
def _norm_kernel(X,RES,NW,W,S,SUM,Y,OQ,OS,NY,NQ,NS,N:gl.constexpr,R:gl.constexpr,
            EPS:gl.constexpr,ADD:gl.constexpr,FUSED:gl.constexpr,IG:gl.constexpr,IR:gl.constexpr,
            DEBUG:gl.constexpr,PDL:gl.constexpr,TRIGGER:gl.constexpr,FACTOR:gl.constexpr,DIV:gl.constexpr):
    if DIV:_hint(W,S,N,4096,R,FUSED,DIV,0,0,False,0 if FUSED else 1)
    if PDL:wait()
    if PDL and TRIGGER==1:trigger()
    # Four warps are fixed because the normalization's FP32 order is part
    # of the tested arithmetic contract.
    V:gl.constexpr=gl.BlockedLayout([8],[32],[4],[0])
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
    if PDL and TRIGGER==2:trigger()
    base=gl.program_id(0)*R
    value=_projection(a,b,scale,W,S,base,N*2 if FUSED else N,R,I,F,FACTOR).to(gl.bfloat16)
    if FUSED:
        up=_projection(a,b,scale,W,S,base+N,N*2,R,I,F,FACTOR).to(gl.bfloat16).to(gl.float32)
        gate=value.to(gl.float32)
        silu=(gate/(1+gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value=(silu*up).to(gl.bfloat16)
    if PDL and TRIGGER==3:trigger()
    O:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
    value=gl.convert_layout(value,O)
    rows=base+gl.arange(0,R,layout=O)
    gl.store(Y+rows,value,rows<N)
    if FUSED:
        output=gl.reshape(value.to(gl.float32),(R//32,32))
        output_scale=gl.maximum(gl.div_rn(gl.max(gl.abs(output),1),127.0),1e-8)
        quant=_round(output*gl.div_rn(1.0,output_scale)[:,None])
        gl.store(OQ+rows,gl.reshape(quant,(R,)).to(gl.int8),rows<N)
        groups=gl.program_id(0)*(R//32)+gl.arange(0,R//32,layout=output_scale.type.layout)
        gl.store(OS+groups,output_scale,groups<N//32)


@gluon.jit
def _plain_kernel(Q,XS,W,S,Y,N:gl.constexpr,K:gl.constexpr,BG:gl.constexpr,R:gl.constexpr,
                  WARPS:gl.constexpr,IG:gl.constexpr,IR:gl.constexpr,FACTOR:gl.constexpr,
                  PDL:gl.constexpr,TRIGGER:gl.constexpr,DIV:gl.constexpr):
    if DIV:_hint(W,S,N,K,R,False,DIV,0,0,False,0)
    # Load only immutable weight scales before the producer dependency wait.
    I:gl.constexpr=gl.BlockedLayout([1,IG,4],[1,32,1],[IR,WARPS//IR,1],[2,1,0])
    F:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[WARPS if K==4096 else WARPS//2,1 if K==4096 else 2],[1,0])
    base=gl.program_id(0)*R
    rf=base+gl.arange(0,R,layout=gl.SliceLayout(1,F));gf=gl.arange(0,BG,layout=gl.SliceLayout(0,F))
    scales=gl.load(S+rf[:,None]*(K//64)+(gf//2)[None,:],(rf[:,None]<N)&(gf[None,:]<K//32),0).to(gl.float32)
    if PDL:wait()
    if PDL and TRIGGER==1:trigger()
    ri=base+gl.arange(0,R,layout=gl.SliceLayout(1,gl.SliceLayout(2,I)))
    gi=gl.arange(0,BG,layout=gl.SliceLayout(0,gl.SliceLayout(2,I)))
    c=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(0,I)))
    pos=gi[None,:,None]*4+gl.expand_dims(gl.expand_dims(c,0),0)
    w=gl.load(gl.cast(W,gl.pointer_type(gl.uint32))+ri[:,None,None]*(K//8)+pos,(ri[:,None,None]<N)&(pos<K//8),0)
    dot=_dot_memory(w,gl.cast(Q,gl.pointer_type(gl.uint64))+pos,pos<K//8)
    sums=gl.convert_layout((gl.sum(dot,2)>>4).to(gl.float32),F);xs=gl.load(XS+gf,gf<K//32,0)
    if FACTOR:
        combined=gl.sum(gl.reshape(sums*xs[None,:],(R,BG//2,2)),2)
        # Both repeated scales are identical; take the first without arithmetic.
        even,_=gl.split(gl.reshape(scales,(R,BG//2,2)))
        value=gl.sum(combined*even,1).to(gl.bfloat16)
    else:value=gl.sum(sums*(scales*xs[None,:]),1).to(gl.bfloat16)
    if PDL and TRIGGER==3:trigger()
    O:gl.constexpr=gl.BlockedLayout([1],[32],[WARPS],[0]);value=gl.convert_layout(value,O)
    rows=base+gl.arange(0,R,layout=O);gl.store(Y+rows,value,rows<N)


def _validate_weights(x,w,s,k):
    if not x.is_cuda or x.dtype!=torch.bfloat16 or x.numel()!=k or k not in (4096,12288):raise ValueError('One BF16 CUDA projection row required')
    if w.ndim!=2 or w.dtype!=torch.uint8 or w.shape[1]!=k//2 or s.shape!=(w.shape[0],k//64) or s.dtype!=torch.bfloat16:
        raise ValueError('Packed G64 weights and BF16 scales required')
    if any(t.device!=x.device or not t.is_contiguous() or t.data_ptr()%16 for t in (x,w,s)):
        raise ValueError('Contiguous aligned buffers on one CUDA device required')


def norm_linear(x,residual,norm_weight,eps,w,s,*,fused=False,rows=8,integer_groups=1,integer_rows=1,
                debug=False,pdl=True,trigger_mode=1,factor=0,divisor=16,return_kernel=False):
    _validate_weights(x,w,s,4096);n2=w.shape[0];n=n2//2 if fused else n2
    if factor not in (0,1) or divisor not in (0,16) or trigger_mode not in (1,3):raise ValueError('Unsupported G64 norm schedule')
    if rows not in ((32,64) if fused else (4,8,16)) or integer_groups not in (1,2,4) or integer_rows not in (1,2,4):raise ValueError('Unsupported norm tile')
    if fused and n2%64:raise ValueError('Complete paired activation groups required')
    tensors=(norm_weight,)+(() if residual is None else (residual,))
    if norm_weight.numel()!=4096 or any(t.device!=x.device or t.dtype!=torch.bfloat16 or not t.is_contiguous() for t in tensors):raise ValueError('BF16 norm and residual buffers required')
    if residual is not None and residual.shape!=x.shape:raise ValueError('Residual shape mismatch')
    summed=torch.empty_like(x) if residual is not None else x
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    oq=torch.empty(n if fused else 0,device=x.device,dtype=torch.int8);os=torch.empty(n//32 if fused else 0,device=x.device)
    ny=torch.empty_like(x) if debug else torch.empty(0,device=x.device,dtype=x.dtype)
    nq=torch.empty(4096 if debug else 0,device=x.device,dtype=torch.int8);ns=torch.empty(128 if debug else 0,device=x.device)
    compiled=_norm_kernel[(triton.cdiv(n,rows),)](x,residual,norm_weight,w,s,summed,y,oq,os,ny,nq,ns,n,rows,eps,residual is not None,fused,integer_groups,integer_rows,debug,pdl,trigger_mode,factor,divisor,num_warps=4,launch_pdl=pdl)
    result=(summed,y,(oq,os)) if fused else (summed,y)
    value=(result,(ny,nq,ns)) if debug else result
    return (value,compiled) if return_kernel else value


def linear(x,w,s,*,prequantized,rows=4,warps=4,integer_groups=1,integer_rows=1,
           pdl=True,trigger_mode=3,factor=0,divisor=0,return_kernel=False):
    k=x.numel();_validate_weights(x,w,s,k);n=w.shape[0];q,sx=prequantized
    if factor not in (0,1) or divisor not in (0,16) or trigger_mode not in (1,3):raise ValueError('Unsupported G64 projection schedule')
    if rows not in (2,4,8,16) or warps not in (2,4,8) or integer_groups not in (1,2,4) or integer_rows not in (1,2,4,8) or warps%integer_rows:
        raise ValueError('Unsupported projection tile')
    if q.dtype!=torch.int8 or q.numel()!=k or sx.dtype!=torch.float32 or sx.numel()!=k//32:raise ValueError('G32 activation buffers required')
    if any(t.device!=x.device or not t.is_contiguous() for t in (q,sx)):raise ValueError('Contiguous CUDA activation buffers required')
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    compiled=_plain_kernel[(triton.cdiv(n,rows),)](q,sx,w,s,y,n,k,triton.next_power_of_2(k//32),rows,warps,integer_groups,integer_rows,factor,pdl,trigger_mode,divisor,num_warps=warps,launch_pdl=pdl)
    return (y,compiled) if return_kernel else y
