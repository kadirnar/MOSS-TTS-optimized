"""Experimental FP8 gate/up projection fused with BF16-rounded SiLU multiply."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_silu_gemv(X,W,S,Y,N:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,HAS_SCALE:tl.constexpr):
    row=tl.program_id(0)
    k=tl.arange(0,BK)
    x=tl.load(X+k,k<K,0).to(tl.float32)
    g=tl.load(W+row*K+k,k<K,0.0).to(tl.float32)
    u=tl.load(W+(row+N)*K+k,k<K,0.0).to(tl.float32)
    if HAS_SCALE:
        gs=tl.load(S+row)
        us=tl.load(S+row+N)
    else:
        gs=1.0
        us=1.0
    gate=(tl.sum(g*x,0)*gs).to(tl.bfloat16).to(tl.float32)
    up=(tl.sum(u*x,0)*us).to(tl.bfloat16).to(tl.float32)
    activated=(gate/(1+tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+row,activated*up)


def fp8_silu_decode(x,weight,scale=None,warps=4):
    n,k=weight.shape
    out=torch.empty((*x.shape[:-1],n//2),device=x.device,dtype=x.dtype)
    _fp8_silu_gemv[(n//2,)](x,weight,scale,out,n//2,k,triton.next_power_of_2(k),scale is not None,num_warps=warps)
    return out
