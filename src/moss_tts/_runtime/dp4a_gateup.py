"""Paired INT4 gate/up integer dot products with a BF16 SiLU epilogue."""
import torch
import triton
import triton.language as tl
from .int4_dp4a import _int8_activation_grouped


@triton.jit
def _dot_row(Q,XS,W,S,row,K:tl.constexpr,GROUP:tl.constexpr,BG:tl.constexpr):
    group=tl.arange(0,BG)
    chunk=tl.arange(0,GROUP//4)
    pos=group[:,None]*(GROUP//4)+chunk[None,:]
    w=tl.load(tl.cast(W,tl.pointer_type(tl.uint16))+row*(K//4)+pos,pos<K//4,0).to(tl.uint32)
    lanes=(w&15)|((w&240)<<4)|((w&3840)<<8)|((w&61440)<<12)
    lanes=lanes|((lanes&0x08080808)*30)
    x=tl.load(tl.cast(Q,tl.pointer_type(tl.int32))+pos,pos<K//4,0)
    dot=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, 0;',constraints='=r,r,r',
        args=[lanes.to(tl.int32),x],dtype=tl.int32,is_pure=True,pack=1)
    sums=tl.sum(dot,1).to(tl.float32)
    scale=tl.load(S+row*(K//GROUP)+group,group<K//GROUP,0)
    xscale=tl.load(XS+group,group<K//GROUP,0)
    return tl.sum(sums*(scale*xscale),0).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _gateup(Q,XS,W,S,Y,N:tl.constexpr,K:tl.constexpr,GROUP:tl.constexpr,BG:tl.constexpr):
    row=tl.program_id(0)
    gate=_dot_row(Q,XS,W,S,row,K,GROUP,BG)
    up=_dot_row(Q,XS,W,S,row+N,K,GROUP,BG)
    silu=(gate/(1+tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+row,silu*up)


def gateup(x,packed,scales,group=32,warps=1,prequantized=None):
    twice_n,k2=packed.shape
    n,k=twice_n//2,k2*2
    if x.dtype!=torch.bfloat16 or x.numel()!=k or twice_n%2:raise ValueError('Invalid gate/up inputs')
    if prequantized is None:
        q=torch.empty(k,device=x.device,dtype=torch.int8)
        sx=torch.empty(k//group,device=x.device,dtype=torch.float32)
        _int8_activation_grouped[(triton.cdiv(k//group,4),)](x,q,sx,k,group,4,num_warps=4)
    else:q,sx=prequantized
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    _gateup[(n,)](q,sx,packed,scales,y,n,k,group,triton.next_power_of_2(k//group),num_warps=warps)
    return y


def enable_gateup(llm):
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Enable gate/up fusion before capture')
    if not all(getattr(layer.mlp,'_dp4a_grouped_activation',False) for layer in llm.model.language_model.layers):
        raise ValueError('Paired gate/up requires grouped DP4A activations')
    for layer in llm.model.language_model.layers:layer.mlp._fused_dp4a_gateup=True
