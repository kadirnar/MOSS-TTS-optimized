"""SIMT INT4 trial: apply group scales after each group's dot-product."""
import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_int4(X,W,S,Y,K:tl.constexpr,GROUP:tl.constexpr,BG:tl.constexpr,VECTOR_X:tl.constexpr=False):
    row=tl.program_id(0)
    group=tl.arange(0,BG)
    pair=tl.arange(0,GROUP//2)
    pos=group[:,None]*(GROUP//2)+pair[None,:]
    packed=tl.load(W+row*(K//2)+pos,pos<K//2,0).to(tl.int32)
    lo=(((packed&15)^8)-8).to(tl.float32)
    hi=(((packed>>4)^8)-8).to(tl.float32)
    if VECTOR_X:
        words=tl.load(tl.cast(X,tl.pointer_type(tl.uint32))+pos,pos<K//2,0)
        x0=words.to(tl.uint16).to(tl.bfloat16,bitcast=True).to(tl.float32)
        x1=(words>>16).to(tl.uint16).to(tl.bfloat16,bitcast=True).to(tl.float32)
    else:
        x0=tl.load(X+2*pos,pos<K//2,0).to(tl.float32)
        x1=tl.load(X+2*pos+1,pos<K//2,0).to(tl.float32)
    group_dot=tl.sum(lo*x0+hi*x1,1)
    scales=tl.load(S+row*(K//GROUP)+group,group<K//GROUP,0)
    tl.store(Y+row,tl.sum(group_dot*scales,0))


def grouped_int4(x,packed,scales,group=128,warps=4,vector_x=False):
    n,k2=packed.shape
    out=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    _grouped_int4[(n,)](x,packed,scales,out,k2*2,group,triton.next_power_of_2(k2*2//group),vector_x,num_warps=warps)
    return out
