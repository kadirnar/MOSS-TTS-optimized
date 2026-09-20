"""Experimental control of group-scale and integer-sum register layouts."""
import torch
import triton
import triton.language as tl
from .int4_dp4a import _int8_activation_grouped
from .dp4a_direct import _direct_dot


@triton.jit
def _load_scales(sums,sp,xp,valid,SHORT:tl.constexpr):
    # Triton disallows broadcasting a floating tensor with pointer operands;
    # pass integer addresses to PTX's 64-bit address constraints instead.
    sp=sp.to(tl.uint64)
    xp=xp.to(tl.uint64)
    if SHORT:
        return tl.inline_asm_elementwise("""{
            .reg .b16 short_scale;
            .reg .pred valid;
            mov.b32 $0, $3;
            mov.b16 short_scale, 0;
            mov.b32 $2, 0;
            setp.ne.s32 valid, $6, 0;
            @valid ld.global.b16 short_scale, [$4];
            @valid ld.global.b32 $2, [$5];
            cvt.f32.bf16 $1, short_scale;
        }""",constraints='=f,=f,=f,f,l,l,r',args=[sums,sp,xp,valid.to(tl.int32)],
            dtype=(tl.float32,tl.float32,tl.float32),is_pure=True,pack=1)
    else:
        return tl.inline_asm_elementwise("""{
            .reg .pred valid;
            mov.b32 $0, $3;
            mov.b32 $1, 0;
            mov.b32 $2, 0;
            setp.ne.s32 valid, $6, 0;
            @valid ld.global.b32 $1, [$4];
            @valid ld.global.b32 $2, [$5];
        }""",constraints='=f,=f,=f,f,l,l,r',args=[sums,sp,xp,valid.to(tl.int32)],
            dtype=(tl.float32,tl.float32,tl.float32),is_pure=True,pack=1)


@triton.jit
def _projection(Q,XS,W,S,rows,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,R:tl.constexpr,MODE:tl.constexpr):
    g=tl.arange(0,BG)
    pos=g[:,None]*4+tl.arange(0,4)[None,:]
    w=tl.load(tl.cast(W,tl.pointer_type(tl.uint32))+rows[:,None,None]*(K//8)+pos[None,:,:],
        (rows[:,None,None]<N)&(pos[None,:,:]<K//8),0)
    p=tl.cast(Q,tl.pointer_type(tl.uint64))+pos
    dot=_direct_dot(w,p[None,:,:],pos[None,:,:]<K//8)
    sums=tl.sum(dot,2).to(tl.float32)
    if MODE>0:g=tl.max_contiguous(g,MODE)
    sp=S+rows[:,None]*(K//32)+g[None,:]
    valid=(rows[:,None]<N)&(g[None,:]<K//32)
    if MODE==-1:
        sums,scale,xscale=_load_scales(sums,sp,XS+g[None,:],valid,S.dtype.element_ty==tl.bfloat16)
        values=sums*(scale*xscale)
    else:
        scale=tl.load(sp,valid,0).to(tl.float32)
        xscale=tl.load(XS+g,g<K//32,0)
        values=sums*(scale*xscale[None,:])
    return tl.sum(values,1)


@triton.jit
def _scale_gemv(Q,XS,W,S,Y,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,R:tl.constexpr,PAIRED:tl.constexpr,MODE:tl.constexpr):
    rows=tl.program_id(0)*R+tl.arange(0,R)
    if PAIRED:
        gate=_projection(Q,XS,W,S,rows,N*2,K,BG,R,MODE).to(tl.bfloat16).to(tl.float32)
        up=_projection(Q,XS,W,S,rows+N,N*2,K,BG,R,MODE).to(tl.bfloat16).to(tl.float32)
        silu=(gate/(1+tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        value=silu*up
    else:value=_projection(Q,XS,W,S,rows,N,K,BG,R,MODE)
    tl.store(Y+rows,value,rows<N)


def linear(x,packed,scales,*,rows=2,warps=2,paired=False,prequantized=None,scale_mode=1):
    n,k2=packed.shape;k=k2*2
    if x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.numel()!=k:raise ValueError('One contiguous BF16 input row required')
    if k%32 or scale_mode not in (-1,0,1,2,4):raise ValueError('Invalid G32 scale-load configuration')
    if prequantized is None:
        q=torch.empty(k,device=x.device,dtype=torch.int8)
        sx=torch.empty(k//32,device=x.device,dtype=torch.float32)
        _int8_activation_grouped[(triton.cdiv(k//32,4),)](x,q,sx,k,32,4,num_warps=4)
    else:q,sx=prequantized
    out_n=n//2 if paired else n
    out=torch.empty((*x.shape[:-1],out_n),device=x.device,dtype=x.dtype)
    _scale_gemv[(triton.cdiv(out_n,rows),)](q,sx,packed,scales,out,out_n,k,
        triton.next_power_of_2(k//32),rows,paired,scale_mode,num_warps=warps)
    return out
