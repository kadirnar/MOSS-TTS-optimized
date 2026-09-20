"""Standalone exact G32 normalization producer with CUDA dependent launch."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait,gdc_launch_dependents


@triton.jit
def _store_quantized(v,Q,S,i,K:tl.constexpr,B:tl.constexpr,GROUP:tl.constexpr):
    if GROUP:
        values=tl.reshape(v,(B//GROUP,GROUP))
        scale=tl.maximum(tl.div_rn(tl.max(tl.abs(values),1),127.0),1e-8)
        inv=tl.div_rn(1.0,scale)
        quantized=tl.reshape(tl.extra.cuda.libdevice.nearbyint(values*inv[:,None]),(B,))
        g=tl.arange(0,B//GROUP)
        tl.store(S+g,scale,g<K//GROUP)
    else:
        scale=tl.maximum(tl.div_rn(tl.max(tl.abs(v),0),127.0),1e-8)
        inv=tl.div_rn(1.0,scale)
        quantized=tl.extra.cuda.libdevice.nearbyint(v*inv)
        tl.store(S,scale)
    tl.store(Q+i,quantized.to(tl.int8),i<K)


@triton.jit
def _norm_quant(X,R,W,SUM,Y,Q,S,K:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr,ADD:tl.constexpr,GROUP:tl.constexpr,LAYOUT_LIMIT:tl.constexpr,TRIGGER:tl.constexpr):
    gdc_wait()
    if TRIGGER==1:gdc_launch_dependents()
    i=tl.arange(0,B)
    if LAYOUT_LIMIT:i=tl.max_contiguous(i,LAYOUT_LIMIT)
    x=tl.load(X+i,i<K,0).to(tl.float32)
    if ADD:
        r=tl.load(R+i,i<K,0).to(tl.float32)
        summed=(x+r).to(tl.bfloat16)
        tl.store(SUM+i,summed,i<K)
        x=summed.to(tl.float32)
    if TRIGGER==2:gdc_launch_dependents()
    inv=tl.rsqrt(tl.sum(x*x,0)/K+EPS)
    norm=(x*inv).to(tl.bfloat16).to(tl.float32)
    w=tl.load(W+i,i<K,0).to(tl.float32)
    y=(norm*w).to(tl.bfloat16)
    if TRIGGER==3:gdc_launch_dependents()
    tl.store(Y+i,y,i<K)
    _store_quantized(y.to(tl.float32),Q,S,i,K,B,GROUP)


def norm_quant(x,residual,weight,eps,*,trigger=1,return_kernel=False):
    if trigger not in (0,1,2,3):raise ValueError('Invalid trigger')
    k=x.shape[-1]
    if k!=4096 or x.dtype!=torch.bfloat16 or x.numel()!=k:raise ValueError('One BF16 K4096 row required')
    y=torch.empty_like(x);summed=torch.empty_like(x) if residual is not None else x
    q=torch.empty(k,device=x.device,dtype=torch.int8);scale=torch.empty(k//32,device=x.device,dtype=torch.float32)
    kernel=_norm_quant[(1,)](x,residual,weight,summed,y,q,scale,k,eps,4096,residual is not None,32,8,trigger,num_warps=4,launch_pdl=True)
    value=summed,y,(q,scale)
    return (value,kernel) if return_kernel else value
