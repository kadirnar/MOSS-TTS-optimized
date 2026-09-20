"""Experimental DP4A activation loads in the weight's register layout.

Inline PTX loads avoid a compiler-inserted shared-memory layout conversion.
All masked K lanes initialize their activation words to zero before DP4A.
"""
import torch
import triton
import triton.language as tl
from .int4_dp4a import _int8_activation_grouped


@triton.jit
def _direct_dot(w,p,valid):
    return tl.inline_asm_elementwise("""{
        .reg .b32 a, b, lo, hi, sign;
        .reg .pred valid;
        mov.b32 a, 0;
        mov.b32 b, 0;
        setp.ne.s32 valid, $3, 0;
        @valid ld.global.v2.u32 {a, b}, [$2];
        and.b32 lo, $1, 0x0f0f0f0f;
        shr.u32 hi, $1, 4;
        and.b32 hi, hi, 0x0f0f0f0f;
        and.b32 sign, lo, 0x08080808;
        mad.lo.u32 lo, sign, 30, lo;
        and.b32 sign, hi, 0x08080808;
        mad.lo.u32 hi, sign, 30, hi;
        dp4a.s32.s32 $0, lo, a, 0;
        dp4a.s32.s32 $0, hi, b, $0;
    }""",constraints='=r,r,l,r',args=[w,p,valid.to(tl.int32)],
        dtype=tl.int32,is_pure=True,pack=1)


@triton.jit
def _projection(Q,XS,W,S,rows,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,R:tl.constexpr):
    g=tl.arange(0,BG)
    chunk=tl.arange(0,4)
    pos=g[:,None]*4+chunk[None,:]
    w=tl.load(tl.cast(W,tl.pointer_type(tl.uint32))+rows[:,None,None]*(K//8)+pos[None,:,:],
        (rows[:,None,None]<N)&(pos[None,:,:]<K//8),0)
    p=tl.cast(Q,tl.pointer_type(tl.uint64))+pos
    dot=_direct_dot(w,p[None,:,:],pos[None,:,:]<K//8)
    sums=tl.sum(dot,2).to(tl.float32)
    scale=tl.load(S+rows[:,None]*(K//32)+g[None,:],(rows[:,None]<N)&(g[None,:]<K//32),0).to(tl.float32)
    xscale=tl.load(XS+g,g<K//32,0)
    return tl.sum(sums*(scale*xscale[None,:]),1)


@triton.jit
def _direct_gemv(Q,XS,W,S,Y,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,R:tl.constexpr,PAIRED:tl.constexpr):
    rows=tl.program_id(0)*R+tl.arange(0,R)
    if PAIRED:
        gate=_projection(Q,XS,W,S,rows,N*2,K,BG,R).to(tl.bfloat16).to(tl.float32)
        up=_projection(Q,XS,W,S,rows+N,N*2,K,BG,R).to(tl.bfloat16).to(tl.float32)
        silu=(gate/(1+tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        value=silu*up
    else:value=_projection(Q,XS,W,S,rows,N,K,BG,R)
    tl.store(Y+rows,value,rows<N)


def linear(x,packed,scales,*,rows=2,warps=2,paired=False,prequantized=None):
    n,k2=packed.shape;k=k2*2
    if x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.numel()!=k:raise ValueError('One contiguous BF16 input row required')
    if k%32:raise ValueError('G32 weights required')
    if prequantized is None:
        q=torch.empty(k,device=x.device,dtype=torch.int8)
        sx=torch.empty(k//32,device=x.device,dtype=torch.float32)
        _int8_activation_grouped[(triton.cdiv(k//32,4),)](x,q,sx,k,32,4,num_warps=4)
    else:q,sx=prequantized
    out_n=n//2 if paired else n
    out=torch.empty((*x.shape[:-1],out_n),device=x.device,dtype=x.dtype)
    _direct_gemv[(triton.cdiv(out_n,rows),)](q,sx,packed,scales,out,out_n,k,
        triton.next_power_of_2(k//32),rows,paired,num_warps=warps)
    return out
