"""Experimental W4A8 integer-dot GEMV; FP32 group scales, BF16 output."""
import torch
import triton
import triton.language as tl


@triton.jit
def _int8_activation(X,Q,S,K:tl.constexpr,B:tl.constexpr,RECIPROCAL:tl.constexpr=False):
    i=tl.arange(0,B)
    x=tl.load(X+i,i<K,0).to(tl.float32)
    scale=tl.maximum(tl.div_rn(tl.max(tl.abs(x),0),127.0),1e-8)
    if RECIPROCAL:
        inv=tl.div_rn(1.0,scale)
        q=tl.extra.cuda.libdevice.nearbyint(x*inv)
    else:
        q=tl.extra.cuda.libdevice.nearbyint(tl.div_rn(x,scale))
    tl.store(Q+i,q.to(tl.int8),i<K)
    tl.store(S,scale)


@triton.jit
def _int8_activation_grouped(X,Q,S,K:tl.constexpr,GROUP:tl.constexpr,R:tl.constexpr):
    g=tl.program_id(0)*R+tl.arange(0,R)
    i=g[:,None]*GROUP+tl.arange(0,GROUP)[None,:]
    x=tl.load(X+i,i<K,0).to(tl.float32)
    scale=tl.maximum(tl.div_rn(tl.max(tl.abs(x),1),127.0),1e-8)
    inv=tl.div_rn(1.0,scale)
    q=tl.extra.cuda.libdevice.nearbyint(x*inv[:,None])
    tl.store(Q+i,q.to(tl.int8),i<K)
    tl.store(S+g,scale,g<K//GROUP)


@triton.jit
def _dp4a_gemv(Q,XS,W,S,Y,K:tl.constexpr,GROUP:tl.constexpr,BG:tl.constexpr,GROUPED_X:tl.constexpr=False):
    row=tl.program_id(0)
    group=tl.arange(0,BG)
    chunk=tl.arange(0,GROUP//4)
    pos=group[:,None]*(GROUP//4)+chunk[None,:]
    weight=tl.load(tl.cast(W,tl.pointer_type(tl.uint16))+row*(K//4)+pos,pos<K//4,0).to(tl.uint32)
    # Four signed 4-bit values become four signed 8-bit lanes, without carries
    # between lanes. Multiplication spreads each nibble's sign into its byte.
    lanes=(weight&15)|((weight&240)<<4)|((weight&3840)<<8)|((weight&61440)<<12)
    lanes=lanes|((lanes&0x08080808)*30)
    x=tl.load(tl.cast(Q,tl.pointer_type(tl.int32))+pos,pos<K//4,0)
    dot=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, 0;',constraints='=r,r,r',
        args=[lanes.to(tl.int32),x],dtype=tl.int32,is_pure=True,pack=1)
    sums=tl.sum(dot,1).to(tl.float32)
    scales=tl.load(S+row*(K//GROUP)+group,group<K//GROUP,0)
    if GROUPED_X:
        xscales=tl.load(XS+group,group<K//GROUP,0)
        value=tl.sum(sums*(scales*xscales),0)
    else:value=tl.sum(sums*scales,0)*tl.load(XS)
    tl.store(Y+row,value)


def int4_dp4a(x,packed,scales,group=128,warps=4,reciprocal=False,prequantized=None,grouped_activation=False):
    n,k2=packed.shape
    k=k2*2
    if x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.numel()!=k:
        raise ValueError('Expected one contiguous BF16 activation row')
    if prequantized is None:
        quantized=torch.empty(k,device=x.device,dtype=torch.int8)
        scale=torch.empty(k//group if grouped_activation else 1,device=x.device,dtype=torch.float32)
        if grouped_activation:
            if not reciprocal:raise ValueError('Grouped activation uses explicit reciprocal rounding')
            _int8_activation_grouped[(triton.cdiv(k//group,4),)](x,quantized,scale,k,group,4,num_warps=4)
        else:_int8_activation[(1,)](x,quantized,scale,k,triton.next_power_of_2(k),reciprocal)
    else:
        quantized,scale=prequantized
        if quantized.numel()!=k or quantized.dtype!=torch.int8 or scale.numel()!=(k//group if grouped_activation else 1) or scale.dtype!=torch.float32:
            raise ValueError('Invalid prequantized activation')
    out=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    _dp4a_gemv[(n,)](quantized,scale,packed,scales,out,k,group,triton.next_power_of_2(k//group),grouped_activation,num_warps=warps)
    return out
