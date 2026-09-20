"""Experimental fused Q/K normalization, RoPE, KV write and split attention.

Current-position K/V are computed locally by every reader. Only one CTA per KV
head writes them to the persistent cache; no reader depends on that write in
the same launch. Older positions are read from the cache as usual.
"""
import torch
import triton
import triton.language as tl
from .kernels import _decode_attn_reduce


@triton.jit
def _normalize_rotate(X,W,COS,SIN,BASE,EPS:tl.constexpr):
    d=tl.arange(0,128)
    swap=(d+64)%128
    x=tl.load(X+BASE+d).to(tl.float32)
    xs=tl.load(X+BASE+swap).to(tl.float32)
    r=tl.rsqrt(tl.sum(x*x,0)/128+EPS)
    w=tl.load(W+d).to(tl.float32)
    ws=tl.load(W+swap).to(tl.float32)
    x=((x*r).to(tl.bfloat16).to(tl.float32)*w).to(tl.bfloat16).to(tl.float32)
    xs=((xs*r).to(tl.bfloat16).to(tl.float32)*ws).to(tl.bfloat16).to(tl.float32)
    c=tl.load(COS+d).to(tl.float32)
    s=tl.load(SIN+d).to(tl.float32)
    return ((x*c).to(tl.bfloat16).to(tl.float32)+(tl.where(d<64,-xs,xs)*s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _fused_qk_attention(X,QW,KW,COS,SIN,K,V,POS,PART,LSE,
                         L:tl.constexpr,SPLITS:tl.constexpr,BLOCK:tl.constexpr,EPS:tl.constexpr):
    h=tl.program_id(0)
    split=tl.program_id(1)
    p=tl.load(POS)
    d=tl.arange(0,128)
    if split*BLOCK<=p:
        q=_normalize_rotate(X,QW,COS,SIN,h*128,EPS)
        current_k=_normalize_rotate(X,KW,COS,SIN,4096+(h//4)*128,EPS)
        current_v=tl.load(X+5120+(h//4)*128+d).to(tl.float32)
        if (split==0)&(h%4==0):
            tl.store(K+((h//4)*L+p)*128+d,current_k)
            tl.store(V+((h//4)*L+p)*128+d,current_v)
        t=split*BLOCK+tl.arange(0,BLOCK)
        k=tl.load(K+((h//4)*L+t[:,None])*128+d[None,:],t[:,None]<p,0).to(tl.float32)
        k=tl.where(t[:,None]==p,current_k[None,:],k)
        logits=tl.sum(k*q[None,:],1)*0.08838834764831845
        logits=tl.where(t<=p,logits,float('-inf'))
        m=tl.max(logits,0)
        prob=tl.exp(logits-m)
        denom=tl.sum(prob,0)
        v=tl.load(V+((h//4)*L+t[:,None])*128+d[None,:],t[:,None]<p,0).to(tl.float32)
        v=tl.where(t[:,None]==p,current_v[None,:],v)
        result=tl.sum(prob[:,None]*v,0)/denom
        logsum=m+tl.log(denom)
    else:
        result=tl.full((128,),0,tl.float32)
        logsum=float('-inf')
    tl.store(PART+(h*SPLITS+split)*128+d,result)
    tl.store(LSE+h*SPLITS+split,logsum)


def fused_qk_attention(qkv,qw,kw,cos,sin,k,v,position,eps,block=32,warps=4):
    length=k.shape[-2]
    splits=triton.cdiv(length,block)
    partial=torch.empty((32,splits,128),device=qkv.device,dtype=torch.float32)
    lse=torch.empty((32,splits),device=qkv.device,dtype=torch.float32)
    out=torch.empty((1,1,4096),device=qkv.device,dtype=qkv.dtype)
    _fused_qk_attention[(32,splits)](qkv,qw,kw,cos,sin,k,v,position,partial,lse,
        length,splits,block,eps,num_warps=warps,enable_fp_fusion=False)
    _decode_attn_reduce[(32,)](partial,lse,out,splits,triton.next_power_of_2(splits),num_warps=4)
    return out
