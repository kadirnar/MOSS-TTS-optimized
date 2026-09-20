"""Experimental split-K FP8 GEMV with FP32 partials and a separate reduction."""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(X,W,P,N:tl.constexpr,K:tl.constexpr,R:tl.constexpr,BK:tl.constexpr,
             SPLITS:tl.constexpr,CACHE:tl.constexpr):
    rows=tl.program_id(0)*R+tl.arange(0,R)
    part=tl.program_id(1)
    cols=part*BK+tl.arange(0,BK)
    x=tl.load(X+cols,cols<K,0).to(tl.float32)
    w=tl.load(W+rows[:,None]*K+cols[None,:],
              (rows[:,None]<N)&(cols[None,:]<K),0.0,cache_modifier=CACHE).to(tl.float32)
    y=tl.sum(w*x[None,:],1)
    tl.store(P+part*N+rows,y,rows<N)


@triton.jit
def _finish(P,S,Y,N:tl.constexpr,SPLITS:tl.constexpr,BS:tl.constexpr,B:tl.constexpr):
    rows=tl.program_id(0)*B+tl.arange(0,B)
    parts=tl.arange(0,BS)
    v=tl.load(P+parts[:,None]*N+rows[None,:],
              (parts[:,None]<SPLITS)&(rows[None,:]<N),0)
    y=tl.sum(v,0)*tl.load(S+rows,rows<N,0)
    tl.store(Y+rows,y,rows<N)


def split_gemv(x,weight,scale,*,block_k=2048,rows=1,warps=4,cache=''):
    n,k=weight.shape
    splits=triton.cdiv(k,block_k)
    partial=torch.empty((splits,n),device=x.device,dtype=torch.float32)
    out=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    _partial[(triton.cdiv(n,rows),splits)](x,weight,partial,n,k,rows,block_k,splits,cache,num_warps=warps)
    _finish[(triton.cdiv(n,256),)](partial,scale,out,n,splits,triton.next_power_of_2(splits),256,num_warps=4)
    return out
