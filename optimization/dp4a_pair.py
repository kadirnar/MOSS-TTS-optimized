"""Experimental shared activation loads for paired gate/up integer dot products."""
import torch
import triton
import triton.language as tl
from .benchmark_gateup_quant import quantize


@triton.jit
def _dot_pair(wg,wu,p,valid):
    return tl.inline_asm_elementwise("""{
        .reg .b32 a, b, gl, gh, ul, uh, gate, up;
        .reg .pred valid;
        mov.b32 gate, $2;
        mov.b32 up, $3;
        mov.b32 a, 0;
        mov.b32 b, 0;
        setp.ne.s32 valid, $5, 0;
        @valid ld.global.v2.u32 {a, b}, [$4];
        shl.b32 gl, gate, 4;
        and.b32 gl, gl, 0xf0f0f0f0;
        and.b32 gh, gate, 0xf0f0f0f0;
        shl.b32 ul, up, 4;
        and.b32 ul, ul, 0xf0f0f0f0;
        and.b32 uh, up, 0xf0f0f0f0;
        dp4a.s32.s32 $0, gl, a, 0;
        dp4a.s32.s32 $0, gh, b, $0;
        dp4a.s32.s32 $1, ul, a, 0;
        dp4a.s32.s32 $1, uh, b, $1;
    }""",constraints='=r,=r,r,r,l,r',args=[wg,wu,p,valid.to(tl.int32)],
        dtype=(tl.int32,tl.int32),is_pure=True,pack=1)


@triton.jit
def _pair_gemv(Q,XS,W,S,Y,OQ,OS,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,R:tl.constexpr):
    rows=tl.program_id(0)*R+tl.arange(0,R)
    g=tl.arange(0,BG);pos=g[:,None]*4+tl.arange(0,4)[None,:]
    wbase=tl.cast(W,tl.pointer_type(tl.uint32))+rows[:,None,None]*(K//8)+pos[None,:,:]
    valid=(rows[:,None,None]<N)&(pos[None,:,:]<K//8)
    wg=tl.load(wbase,valid,0);wu=tl.load(wbase+N*(K//8),valid,0)
    p=tl.cast(Q,tl.pointer_type(tl.uint64))+pos
    gd,ud=_dot_pair(wg,wu,p[None,:,:],pos[None,:,:]<K//8)
    gs=(tl.sum(gd,2)>>4).to(tl.float32);us=(tl.sum(ud,2)>>4).to(tl.float32)
    g=tl.max_contiguous(g,4)
    sg=tl.load(S+rows[:,None]*(K//32)+g[None,:],(rows[:,None]<N)&(g[None,:]<K//32),0).to(tl.float32)
    su=tl.load(S+(rows[:,None]+N)*(K//32)+g[None,:],(rows[:,None]<N)&(g[None,:]<K//32),0).to(tl.float32)
    xs=tl.load(XS+g,g<K//32,0)
    gate=tl.sum(gs*(sg*xs[None,:]),1).to(tl.bfloat16).to(tl.float32)
    up=tl.sum(us*(su*xs[None,:]),1).to(tl.bfloat16).to(tl.float32)
    silu=(gate/(1+tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    value=(silu*up).to(tl.bfloat16)
    tl.store(Y+rows,value,rows<N)
    grouped=tl.reshape(value.to(tl.float32),(R//32,32))
    scale=tl.maximum(tl.div_rn(tl.max(tl.abs(grouped),1),127.0),1e-8)
    inv=tl.div_rn(1.0,scale)
    quant=tl.reshape(tl.extra.cuda.libdevice.nearbyint(grouped*inv[:,None]),(R,)).to(tl.int8)
    tl.store(OQ+rows,quant,rows<N)
    groups=tl.program_id(0)*(R//32)+tl.arange(0,R//32)
    tl.store(OS+groups,scale,groups<N//32)


def gateup(x,w,s,*,rows=32,warps=4,prequantized=None):
    n2,k2=w.shape;n=n2//2;k=k2*2
    if n2%64 or rows not in (32,64) or k%32 or x.numel()!=k or x.dtype!=torch.bfloat16:raise ValueError('G32 paired shape required')
    q,sx=quantize(x) if prequantized is None else prequantized
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    oq=torch.empty(n,device=x.device,dtype=torch.int8);os=torch.empty(n//32,device=x.device)
    _pair_gemv[(triton.cdiv(n,rows),)](q,sx,w,s,y,oq,os,n,k,triton.next_power_of_2(k//32),rows,num_warps=warps)
    return y,(oq,os)
