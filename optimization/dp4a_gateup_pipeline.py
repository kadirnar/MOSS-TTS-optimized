"""Register-lifetime experiment: compute smaller row tiles before quantization."""
import torch
import triton
import triton.language as tl
from .dp4a_scale import _projection
from .int4_dp4a import _int8_activation_grouped


@triton.jit
def _rows(Q,XS,W,S,rows,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,TILE:tl.constexpr):
    gate=_projection(Q,XS,W,S,rows,N*2,K,BG,TILE,4).to(tl.bfloat16).to(tl.float32)
    up=_projection(Q,XS,W,S,rows+N,N*2,K,BG,TILE,4).to(tl.bfloat16).to(tl.float32)
    silu=(gate/(1+tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    return (silu*up).to(tl.bfloat16)


@triton.jit
def _gateup_pipeline(Q,XS,W,S,Y,OQ,OS,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,TILE:tl.constexpr):
    rows=tl.program_id(0)*32+tl.arange(0,TILE)
    a=_rows(Q,XS,W,S,rows,N,K,BG,TILE)
    b=_rows(Q,XS,W,S,rows+TILE,N,K,BG,TILE)
    value=tl.cat(a,b)
    if TILE==8:
        c=_rows(Q,XS,W,S,rows+16,N,K,BG,TILE)
        d=_rows(Q,XS,W,S,rows+24,N,K,BG,TILE)
        value=tl.cat(value,tl.cat(c,d))
    dest=tl.program_id(0)*32+tl.arange(0,32)
    tl.store(Y+dest,value,dest<N)
    f=value.to(tl.float32)
    scale=tl.maximum(tl.div_rn(tl.max(tl.abs(f),0),127.0),1e-8)
    inv=tl.div_rn(1.0,scale)
    quant=tl.extra.cuda.libdevice.nearbyint(f*inv).to(tl.int8)
    tl.store(OQ+dest,quant,dest<N);tl.store(OS+tl.program_id(0),scale)


def gateup_pipeline(x,packed,scales,*,tile=16,warps=4,prequantized=None):
    n2,k2=packed.shape;n=n2//2;k=k2*2
    if n%32 or k%32 or tile not in (8,16):raise ValueError('G32 dimensions and an 8/16-row tile required')
    if prequantized is None:
        q=torch.empty(k,device=x.device,dtype=torch.int8);sx=torch.empty(k//32,device=x.device)
        _int8_activation_grouped[(triton.cdiv(k//32,4),)](x,q,sx,k,32,4,num_warps=4)
    else:q,sx=prequantized
    out=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    oq=torch.empty(n,device=x.device,dtype=torch.int8);os=torch.empty(n//32,device=x.device)
    _gateup_pipeline[(n//32,)](q,sx,packed,scales,out,oq,os,n,k,triton.next_power_of_2(k//32),tile,num_warps=warps)
    return out,(oq,os)
