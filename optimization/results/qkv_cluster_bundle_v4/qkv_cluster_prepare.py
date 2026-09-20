"""Experimental G32 norm/QKV + head norm/RoPE/cache fusion using SM90 clusters.

Each program computes one 128-channel Q/K/V head. CTAs shard its projection
rows, then gather Q/K heads before preserving the original head reduction.
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from .dp4a_norm_pdl import _round,_projection,_dot
from .pdl_control import wait,trigger
from .bulk_address import _bulk


@gluon.jit
def _mul(x,y):
    return gl.inline_asm_elementwise('mul.rn.f32 $0,$1,$2;','=f,f,f',[x,y],dtype=gl.float32,is_pure=True,pack=1)


@gluon.jit
def _add(x,y):
    return gl.inline_asm_elementwise('add.rn.f32 $0,$1,$2;','=f,f,f',[x,y],dtype=gl.float32,is_pure=True,pack=1)


@gluon.jit
def _four(x):
    even,odd=gl.split(gl.reshape(x,(128,32,2,2)))
    x0,x2=gl.split(even);x1,x3=gl.split(odd)
    return x0,x1,x2,x3


@gluon.jit
def _norm_sum_legacy(x):
    # V distributes 8 consecutive elements per lane, then strides by 1024.
    # Keep the selected compiler's sequential 32-value local FMA chain before
    # the unchanged warp and four-warp butterfly reductions.
    lanes=gl.reshape(gl.permute(gl.reshape(x,(4,128,8)),(1,0,2)),(128,32))
    idx=gl.full((128,1),1,gl.int32,lanes.type.layout)
    value=gl.reshape(gl.gather(lanes,idx,1),(128,))
    acc=_mul(value,value)
    idx=gl.full((128,1),0,gl.int32,lanes.type.layout)
    value=gl.reshape(gl.gather(lanes,idx,1),(128,))
    acc=gl.fma(value,value,acc)
    for j in gl.static_range(2,32):
        idx=gl.full((128,1),j,gl.int32,lanes.type.layout)
        value=gl.reshape(gl.gather(lanes,idx,1),(128,))
        acc=gl.fma(value,value,acc)
    return gl.sum(acc,0)


@gluon.jit
def _projection_legacy(a,b,xscale,W,S,base,I:gl.constexpr,F:gl.constexpr):
    ri=base+gl.arange(0,128,layout=gl.SliceLayout(1,gl.SliceLayout(2,I)))
    gi=gl.arange(0,128,layout=gl.SliceLayout(0,gl.SliceLayout(2,I)))
    c=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(0,I)))
    pos=gi[None,:,None]*4+gl.expand_dims(gl.expand_dims(c,0),0)
    w=gl.load(gl.cast(W,gl.pointer_type(gl.uint32))+ri[:,None,None]*512+pos)
    dot=_dot(w,a[None,:,:],b[None,:,:])
    sums=gl.convert_layout((gl.sum(dot,2)>>4).to(gl.float32),F)
    rf=base+gl.arange(0,128,layout=gl.SliceLayout(1,F))
    gf=gl.arange(0,128,layout=gl.SliceLayout(0,F))
    scale=gl.load(S+rf[:,None]*128+gf[None,:]).to(gl.float32)
    d0,d1,d2,d3=_four(sums);s0,s1,s2,s3=_four(_mul(scale,xscale[None,:]))
    # Triton 3.7 folds the four local products from left to right, with the
    # first pair contracted as d0*s0 + rounded(d1*s1). Triton 3.8 changed
    # this to a balanced tree. Keep the selected arithmetic explicitly.
    partial=gl.fma(d0,s0,_mul(d1,s1))
    partial=gl.fma(d2,s2,partial);partial=gl.fma(d3,s3,partial)
    return gl.sum(partial,1)


@gluon.jit
def _kernel(X,RES,NW,W,S,SUM,Y,QW,KW,COS,SIN,Q,KC,VC,POS,
            EPS:gl.constexpr,HEPS:gl.constexpr,L:gl.constexpr,ADD:gl.constexpr,
            CTAS:gl.constexpr,IG:gl.constexpr,IR:gl.constexpr,DIV:gl.constexpr,
            TRIGGER:gl.constexpr,DEBUG:gl.constexpr,LEGACY:gl.constexpr,LEGACY_NORM:gl.constexpr,
            V:gl.constexpr,I:gl.constexpr,F:gl.constexpr,O:gl.constexpr):
    h=gl.program_id(0)
    if DIV:
        rank=gl.inline_asm_elementwise('mov.u32 $0,%cluster_ctarank;','=r',[],dtype=gl.int32,is_pure=False,pack=1)
        pointer=gl.cast(W,gl.pointer_type(gl.uint8))+(h*128+rank*(128//CTAS))*2048
        size=h*0+(128//CTAS)*2048//DIV
        _bulk(pointer,size,0)
    wait()
    if TRIGGER==1:trigger()
    i=gl.arange(0,4096,layout=V)
    x=gl.load(X+i).to(gl.float32)
    if ADD:
        x=(x+gl.load(RES+i).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
        gl.store(SUM+i,x,h==0)
    if LEGACY_NORM:square_sum=_norm_sum_legacy(x)
    else:square_sum=gl.sum(x*x,0)
    inv=gl.rsqrt(square_sum/4096+EPS)
    normalized=((x*inv).to(gl.bfloat16).to(gl.float32)*gl.load(NW+i).to(gl.float32)).to(gl.bfloat16)
    grouped=gl.reshape(normalized.to(gl.float32),(128,32))
    scale=gl.maximum(gl.div_rn(gl.max(gl.abs(grouped),1),127.0),1e-8)
    quant=_round(grouped*gl.div_rn(1.0,scale)[:,None])
    q4=gl.reshape(quant,(1024,4));c=gl.arange(0,4,layout=gl.SliceLayout(0,q4.type.layout))
    words=gl.sum((q4&255)<<(c[None,:]*8),1)
    a,b=gl.split(gl.reshape(words,(512,2)))
    a=gl.convert_layout(gl.reshape(a,(128,4)),gl.SliceLayout(0,I))
    b=gl.convert_layout(gl.reshape(b,(128,4)),gl.SliceLayout(0,I))
    scale=gl.convert_layout(scale,gl.SliceLayout(0,F))
    if LEGACY:value=_projection_legacy(a,b,scale,W,S,h*128,I,F).to(gl.bfloat16)
    else:value=_projection(a,b,scale,W,S,h*128,6144,128,I,F).to(gl.bfloat16)
    if DEBUG:
        rows=gl.arange(0,128,layout=value.type.layout)
        gl.store(Y+h*128+rows,value)
    if TRIGGER==2:trigger()
    p=gl.load(POS)
    if h<40:
        # Replicate the full head, preserving the original four-warp sum tree.
        head=gl.convert_layout(value,O).to(gl.float32)
        d=gl.arange(0,128,layout=O);swap=(d+64)%128
        swapped=gl.gather(head,swap,0)
        norm=QW if h<32 else KW
        weight=gl.load(norm+d).to(gl.float32);weight_swap=gl.load(norm+swap).to(gl.float32)
        r=gl.rsqrt(_add(_mul(gl.sum(_mul(head,head),0),1.0/128),HEPS))
        head=((head*r).to(gl.bfloat16).to(gl.float32)*weight).to(gl.bfloat16).to(gl.float32)
        swapped=((swapped*r).to(gl.bfloat16).to(gl.float32)*weight_swap).to(gl.bfloat16).to(gl.float32)
        cosine=gl.load(COS+d).to(gl.float32);sine=gl.load(SIN+d).to(gl.float32)
        # Explicit sign flip preserves -0 across the BF16 rotary boundary.
        signed=(swapped.to(gl.uint32,bitcast=True)^gl.where(d<64,0x80000000,0)).to(gl.float32,bitcast=True)
        rotated=_add((head*cosine).to(gl.bfloat16).to(gl.float32),(signed*sine).to(gl.bfloat16).to(gl.float32))
        if TRIGGER==3:trigger()
        if h<32:gl.store(Q+h*128+d,rotated)
        else:gl.store(KC+((h-32)*L+p)*128+d,rotated)
    else:
        if TRIGGER==3:trigger()
        rows=gl.arange(0,128,layout=value.type.layout)
        gl.store(VC+((h-40)*L+p)*128+rows,value)


def launch(x,residual,norm_weight,eps,w,s,q_weight,k_weight,cos,sin,kc,vc,position,head_eps,
           *,ctas=4,integer_groups=1,integer_rows=1,divisor=0,trigger_mode=1,legacy_projection=False,legacy_norm=False,debug=False,return_kernel=False,compiled=None):
    if torch.cuda.get_device_capability()!=(9,0):raise ValueError('SM90 experiment required')
    if ctas not in (1,2,4,8) or integer_groups not in (1,2,4) or integer_rows not in (1,2,4) or divisor not in (0,16) or trigger_mode not in (0,1,2,3):
        raise ValueError('Unsupported cluster layout')
    if x.dtype!=torch.bfloat16 or x.numel()!=4096 or norm_weight.numel()!=4096 or w.shape!=(6144,2048) or w.dtype!=torch.uint8 or s.shape!=(6144,128):
        raise ValueError('Fixed K4096/N6144 G32 projection required')
    if kc.shape!=vc.shape or kc.shape[-1]!=128 or kc.numel()!=8*kc.shape[-2]*128:
        raise ValueError('Eight contiguous cache heads required')
    tensors=(x,norm_weight,w,s,q_weight,k_weight,cos,sin,kc,vc,position)+(() if residual is None else (residual,))
    if not x.is_cuda or any(t.device!=x.device or not t.is_contiguous() for t in tensors):raise ValueError('Contiguous same-device CUDA tensors required')
    if any(t.dtype!=torch.bfloat16 for t in (norm_weight,q_weight,k_weight,cos,sin,kc,vc)) or s.dtype not in (torch.bfloat16,torch.float32):raise ValueError('BF16 states and original scale precision required')
    if residual is not None and (residual.shape!=x.shape or residual.dtype!=x.dtype):raise ValueError('Matching residual required')
    if q_weight.numel()!=128 or k_weight.numel()!=128 or cos.numel()!=128 or sin.numel()!=128 or position.numel()!=1 or position.dtype!=torch.int64:raise ValueError('One token and head dimensions required')
    summed=torch.empty_like(x) if residual is not None else x
    projected=torch.empty((1,6144) if debug else (0,),device=x.device,dtype=x.dtype)
    q=torch.empty((32,128),device=x.device,dtype=x.dtype)
    bits=ctas.bit_length()-1;replicated=[[0] for _ in range(bits)]
    vlayout=gl.BlockedLayout([8],[32],[4],[0],cga_layout=replicated)
    ilayout=gl.BlockedLayout([1,integer_groups,4],[1,32,1],[integer_rows,4//integer_rows,1],[2,1,0],cga_layout=[[1<<i,0,0] for i in range(bits)])
    flayout=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0],cga_layout=[[1<<i,0] for i in range(bits)])
    olayout=gl.BlockedLayout([1],[32],[4],[0],cga_layout=replicated)
    if compiled is None:
        kernel=_kernel[(48,)](x,residual,norm_weight,w,s,summed,projected,q_weight,k_weight,cos,sin,q,kc,vc,position,
            eps,head_eps,kc.shape[-2],residual is not None,ctas,integer_groups,integer_rows,divisor,trigger_mode,debug,legacy_projection,legacy_norm,
            vlayout,ilayout,flayout,olayout,num_warps=4,num_ctas=ctas,launch_pdl=True)
    else:
        compiled(dict(X=x,RES=residual,NW=norm_weight,W=w,S=s,SUM=summed,Y=projected,QW=q_weight,KW=k_weight,
            COS=cos,SIN=sin,Q=q,KC=kc,VC=vc,POS=position),48,eps,head_eps,kc.shape[-2])
        kernel=compiled
    value=(summed,projected,q)
    return (value,kernel) if return_kernel else value
