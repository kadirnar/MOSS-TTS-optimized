"""Opt-in producer/INT8 quantizer fusions, preserving intermediate BF16 rounds."""
import torch
import triton
import triton.language as tl


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
def _norm_quant(X,R,W,SUM,Y,Q,S,K:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr,ADD:tl.constexpr,GROUP:tl.constexpr,LAYOUT_LIMIT:tl.constexpr):
    i=tl.arange(0,B)
    if LAYOUT_LIMIT:i=tl.max_contiguous(i,LAYOUT_LIMIT)
    x=tl.load(X+i,i<K,0).to(tl.float32)
    if ADD:
        r=tl.load(R+i,i<K,0).to(tl.float32)
        summed=(x+r).to(tl.bfloat16)
        tl.store(SUM+i,summed,i<K)
        x=summed.to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/K+EPS)
    norm=(x*inv).to(tl.bfloat16).to(tl.float32)
    w=tl.load(W+i,i<K,0).to(tl.float32)
    y=(norm*w).to(tl.bfloat16)
    tl.store(Y+i,y,i<K)
    _store_quantized(y.to(tl.float32),Q,S,i,K,B,GROUP)


@triton.jit
def _silu_quant(X,Y,Q,S,K:tl.constexpr,B:tl.constexpr,GROUP:tl.constexpr):
    i=tl.arange(0,B)
    a=tl.load(X+i,i<K,0).to(tl.float32)
    b=tl.load(X+K+i,i<K,0).to(tl.float32)
    silu=(a/(1+tl.exp(-a))).to(tl.bfloat16).to(tl.float32)
    y=(silu*b).to(tl.bfloat16)
    tl.store(Y+i,y,i<K)
    _store_quantized(y.to(tl.float32),Q,S,i,K,B,GROUP)


def norm_quant(x,residual,weight,eps,group=0,layout_limit=0):
    k=x.shape[-1]
    if x.dtype!=torch.bfloat16 or x.numel()!=k:raise ValueError('One BF16 activation row required')
    y=torch.empty_like(x)
    summed=torch.empty_like(x) if residual is not None else x
    q=torch.empty(k,device=x.device,dtype=torch.int8)
    scale=torch.empty(k//group if group else 1,device=x.device,dtype=torch.float32)
    _norm_quant[(1,)](x,residual,weight,summed,y,q,scale,k,eps,triton.next_power_of_2(k),residual is not None,group,layout_limit,num_warps=4)
    return summed,y,(q,scale)


def silu_quant(x,group=0):
    k=x.shape[-1]//2
    if x.dtype!=torch.bfloat16 or x.numel()!=2*k:raise ValueError('One BF16 gate/up row required')
    y=torch.empty((*x.shape[:-1],k),device=x.device,dtype=x.dtype)
    q=torch.empty(k,device=x.device,dtype=torch.int8)
    scale=torch.empty(k//group if group else 1,device=x.device,dtype=torch.float32)
    _silu_quant[(1,)](x,y,q,scale,k,triton.next_power_of_2(k),group,num_warps=4)
    return y,(q,scale)


def enable_fusions(llm,*,silu=False,layout_limit=0):
    from .calibrated_backend import enable_dp4a_reciprocal
    if not llm.fused_residual:raise ValueError('Producer quantizer fusions require fused residual path')
    enable_dp4a_reciprocal(llm)
    llm.fused_dp4a=True
    llm.dp4a_norm_layout_limit=layout_limit
    # The one-block SiLU quantizer is correct but slower on this H200.
    for layer in llm.model.language_model.layers:layer.mlp._fused_dp4a=silu
