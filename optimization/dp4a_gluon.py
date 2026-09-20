"""Explicit Gluon layouts for grouped integer-dot projections."""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from .int4_dp4a import _int8_activation_grouped


@gluon.jit
def _dot(w,p,valid):
    return gl.inline_asm_elementwise("""{
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
    }""",constraints='=r,r,l,r',args=[w,p,valid.to(gl.int32)],
        dtype=gl.int32,is_pure=True,pack=1)


@gluon.jit
def _projection(Q,XS,W,S,rows,N:gl.constexpr,K:gl.constexpr,BG:gl.constexpr,R:gl.constexpr,L:gl.constexpr):
    g=gl.arange(0,BG,layout=gl.SliceLayout(0,gl.SliceLayout(2,L)))
    c=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(0,L)))
    pos=g[None,:,None]*4+gl.expand_dims(gl.expand_dims(c,0),0)
    w=gl.load(gl.cast(W,gl.pointer_type(gl.uint32))+rows[:,None,None]*(K//8)+pos,
        (rows[:,None,None]<N)&(pos<K//8),0)
    ptr=gl.cast(Q,gl.pointer_type(gl.uint64))+pos
    dot=_dot(w,ptr,pos<K//8)
    sums=gl.sum(dot,2).to(gl.float32)
    scale=gl.load(S+rows[:,None]*(K//32)+g[None,:],(rows[:,None]<N)&(g[None,:]<K//32),0).to(gl.float32)
    xs=gl.load(XS+g,g<K//32,0)
    return gl.sum(sums*(scale*xs[None,:]),1)


@gluon.jit
def _gluon_gemv(Q,XS,W,S,Y,N:gl.constexpr,K:gl.constexpr,BG:gl.constexpr,R:gl.constexpr,PAIRED:gl.constexpr,GCONT:gl.constexpr,WARPS:gl.constexpr):
    layout:gl.constexpr=gl.BlockedLayout([1,GCONT,4],[1,32,1],[1,WARPS,1],[2,1,0])
    rows=gl.program_id(0)*R+gl.arange(0,R,layout=gl.SliceLayout(1,gl.SliceLayout(2,layout)))
    if PAIRED:
        gate=_projection(Q,XS,W,S,rows,N*2,K,BG,R,layout).to(gl.bfloat16).to(gl.float32)
        up=_projection(Q,XS,W,S,rows+N,N*2,K,BG,R,layout).to(gl.bfloat16).to(gl.float32)
        silu=(gate/(1+gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value=silu*up
    else:value=_projection(Q,XS,W,S,rows,N,K,BG,R,layout)
    gl.store(Y+rows,value,rows<N)


def linear(x,packed,scales,*,rows=2,warps=2,paired=False,prequantized=None,group_contiguous=4):
    n,k2=packed.shape;k=k2*2
    if x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.numel()!=k:raise ValueError('One contiguous BF16 row required')
    if k%32 or group_contiguous not in (1,2,4,8):raise ValueError('Invalid G32 layout')
    if prequantized is None:
        q=torch.empty(k,device=x.device,dtype=torch.int8)
        sx=torch.empty(k//32,device=x.device,dtype=torch.float32)
        _int8_activation_grouped[(triton.cdiv(k//32,4),)](x,q,sx,k,32,4,num_warps=4)
    else:q,sx=prequantized
    out_n=n//2 if paired else n
    out=torch.empty((*x.shape[:-1],out_n),device=x.device,dtype=x.dtype)
    _gluon_gemv[(triton.cdiv(out_n,rows),)](q,sx,packed,scales,out,out_n,k,triton.next_power_of_2(k//32),rows,paired,group_contiguous,warps,num_warps=warps)
    return out
