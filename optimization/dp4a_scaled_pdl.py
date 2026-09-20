"""Experimental exact INT4 expansion using signed bytes scaled by sixteen.

Moving each nibble into the high four bits of an INT8 byte replaces signed
extension. DP4A then returns 16 times the desired integer dot product. Eight
terms fit in INT32 even for weight -8 and activation -128. Removing the factor
before FP32 conversion preserves every integer sum and subsequent arithmetic.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait,gdc_launch_dependents
from .int4_dp4a import _int8_activation_grouped


SELECTED = {
    'up': {'rows': 32, 'warps': 4, 'mode': 2},
    'qkv': {'rows': 4, 'warps': 4, 'mode': 2},
    'out': {'rows': 4, 'warps': 4, 'mode': 1},
    'down': {'rows': 4, 'warps': 2, 'mode': 1},
}


@triton.jit
def _dot(w,p,valid,MODE:tl.constexpr):
    scaled=tl.inline_asm_elementwise("""{
        .reg .b32 a, b, lo, hi;
        .reg .pred valid;
        mov.b32 a, 0;
        mov.b32 b, 0;
        setp.ne.s32 valid, $3, 0;
        @valid ld.global.v2.u32 {a, b}, [$2];
        shl.b32 lo, $1, 4;
        and.b32 lo, lo, 0xf0f0f0f0;
        and.b32 hi, $1, 0xf0f0f0f0;
        dp4a.s32.s32 $0, lo, a, 0;
        dp4a.s32.s32 $0, hi, b, $0;
    }""",constraints='=r,r,l,r',args=[w,p,valid.to(tl.int32)],
        dtype=tl.int32,is_pure=True,pack=1)
    if MODE==1:return scaled>>4
    return scaled


@triton.jit
def _projection(Q,XS,W,S,rows,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,R:tl.constexpr,
                MODE:tl.constexpr,SCALE_MODE:tl.constexpr):
    g=tl.arange(0,BG)
    pos=g[:,None]*4+tl.arange(0,4)[None,:]
    w=tl.load(tl.cast(W,tl.pointer_type(tl.uint32))+rows[:,None,None]*(K//8)+pos[None,:,:],
        (rows[:,None,None]<N)&(pos[None,:,:]<K//8),0)
    p=tl.cast(Q,tl.pointer_type(tl.uint64))+pos
    dot=_dot(w,p[None,:,:],pos[None,:,:]<K//8,MODE)
    isum=tl.sum(dot,2)
    if MODE==2:isum=isum>>4
    sums=isum.to(tl.float32)
    if SCALE_MODE:g=tl.max_contiguous(g,SCALE_MODE)
    scale=tl.load(S+rows[:,None]*(K//32)+g[None,:],(rows[:,None]<N)&(g[None,:]<K//32),0).to(tl.float32)
    xscale=tl.load(XS+g,g<K//32,0)
    return tl.sum(sums*(scale*xscale[None,:]),1)


@triton.jit
def _gemv(Q,XS,W,S,Y,OQ,OS,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,R:tl.constexpr,
          MODE:tl.constexpr,PAIRED:tl.constexpr,FUSED:tl.constexpr,SCALE_MODE:tl.constexpr,TRIGGER:tl.constexpr):
    gdc_wait()
    if TRIGGER==1:gdc_launch_dependents()
    rows=tl.program_id(0)*R+tl.arange(0,R)
    if PAIRED:
        gate=_projection(Q,XS,W,S,rows,N*2,K,BG,R,MODE,SCALE_MODE).to(tl.bfloat16).to(tl.float32)
        if TRIGGER==2:gdc_launch_dependents()
        up=_projection(Q,XS,W,S,rows+N,N*2,K,BG,R,MODE,SCALE_MODE).to(tl.bfloat16).to(tl.float32)
        silu=(gate/(1+tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        value=(silu*up).to(tl.bfloat16)
    else:value=_projection(Q,XS,W,S,rows,N,K,BG,R,MODE,SCALE_MODE).to(tl.bfloat16)
    if TRIGGER==3:gdc_launch_dependents()
    tl.store(Y+rows,value,rows<N)
    if FUSED:
        grouped=tl.reshape(value.to(tl.float32),(R//32,32))
        scale=tl.maximum(tl.div_rn(tl.max(tl.abs(grouped),1),127.0),1e-8)
        inv=tl.div_rn(1.0,scale)
        quant=tl.reshape(tl.extra.cuda.libdevice.nearbyint(grouped*inv[:,None]),(R,)).to(tl.int8)
        tl.store(OQ+rows,quant,rows<N)
        groups=tl.program_id(0)*(R//32)+tl.arange(0,R//32)
        tl.store(OS+groups,scale,groups<N//32)


def linear(x,packed,scales,*,rows=4,warps=4,mode=2,paired=False,fused=False,prequantized=None,scale_mode=0,trigger=1,return_kernel=False):
    if trigger not in (0,1,2,3):raise ValueError("Invalid trigger")
    n,k2=packed.shape;k=k2*2
    if x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.numel()!=k or k%32 or mode not in (1,2):
        raise ValueError('One contiguous BF16 row and grouped G32 weights required')
    if fused and (not paired or rows not in (32,64) or n%64):raise ValueError('Fused output quantization requires complete groups of 32')
    if prequantized is None:
        q=torch.empty(k,device=x.device,dtype=torch.int8);sx=torch.empty(k//32,device=x.device)
        _int8_activation_grouped[(triton.cdiv(k//32,4),)](x,q,sx,k,32,4,num_warps=4)
    else:q,sx=prequantized
    out_n=n//2 if paired else n
    out=torch.empty((*x.shape[:-1],out_n),device=x.device,dtype=x.dtype)
    oq=torch.empty(out_n if fused else 0,device=x.device,dtype=torch.int8)
    os=torch.empty(out_n//32 if fused else 0,device=x.device)
    kernel=_gemv[(triton.cdiv(out_n,rows),)](q,sx,packed,scales,out,oq,os,out_n,k,triton.next_power_of_2(k//32),rows,mode,paired,fused,scale_mode,trigger,num_warps=warps,launch_pdl=True)
    value=(out,(oq,os)) if fused else out
    return (value,kernel) if return_kernel else value
