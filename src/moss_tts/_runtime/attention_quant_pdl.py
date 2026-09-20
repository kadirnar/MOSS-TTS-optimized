"""Experimental exact split reduction/quantization with PDL."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait,gdc_launch_dependents


@triton.jit
def _reduce_quant(PART,LSE,OUT,Q,SCALE,SPLITS:tl.constexpr,BS:tl.constexpr,PDL:tl.constexpr,TRIGGER:tl.constexpr):
    if PDL:gdc_wait()
    if PDL and TRIGGER==1:gdc_launch_dependents()
    h=tl.program_id(0)
    s=tl.arange(0,BS)
    d=tl.arange(0,128)
    lse=tl.load(LSE+h*SPLITS+s,s<SPLITS,float('-inf'))
    a=tl.exp(lse-tl.max(lse,0))
    a=a/tl.sum(a,0)
    val=tl.load(PART+(h*SPLITS+s[:,None])*128+d[None,:],s[:,None]<SPLITS,0)
    out=tl.sum(a[:,None]*val,0).to(tl.bfloat16)
    if PDL and TRIGGER==2:gdc_launch_dependents()
    tl.store(OUT+h*128+d,out)
    values=tl.reshape(out.to(tl.float32),(4,32))
    scale=tl.maximum(tl.div_rn(tl.max(tl.abs(values),1),127.0),1e-8)
    inv=tl.div_rn(1.0,scale)
    quant=tl.reshape(tl.extra.cuda.libdevice.nearbyint(values*inv[:,None]),(128,))
    if PDL and TRIGGER==3:gdc_launch_dependents()
    tl.store(Q+h*128+d,quant.to(tl.int8))
    tl.store(SCALE+h*4+tl.arange(0,4),scale)


def reduce_quant(partial,lse,warps=4,*,pdl=True,trigger=1,return_kernel=False):
    if trigger not in (0,1,2,3):raise ValueError("Invalid PDL trigger")
    splits=partial.shape[1]
    out=torch.empty((1,1,4096),device=partial.device,dtype=torch.bfloat16)
    q=torch.empty(4096,device=partial.device,dtype=torch.int8)
    scale=torch.empty(128,device=partial.device,dtype=torch.float32)
    kernel=_reduce_quant[(32,)](partial,lse,out,q,scale,splits,triton.next_power_of_2(splits),pdl,trigger,num_warps=warps,launch_pdl=pdl)
    value=out,(q,scale)
    return (value,kernel) if return_kernel else value

