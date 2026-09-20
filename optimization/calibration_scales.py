"""Optional static group-scale search before GPTQ error feedback.

Searches BF16-rounded scales and reconstructions over a fixed clipping grid.
The diagonal mode weights each squared weight error by calibration activation
energy. This is a diagonal proxy, not a full output-error optimum or AWQ.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _search(W,H,S,COUNT:tl.constexpr,NG:tl.constexpr,GROUP:tl.constexpr,
            R:tl.constexpr,STEPS:tl.constexpr,DIAGONAL:tl.constexpr):
    group=tl.program_id(0)*R+tl.arange(0,R)
    column=tl.arange(0,GROUP)
    w=tl.load(W+group[:,None]*GROUP+column[None,:],group[:,None]<COUNT,0).to(tl.float32)
    if DIAGONAL:
        importance=tl.load(H+(group%NG)[:,None]*GROUP+column[None,:])
    else:importance=tl.full((R,GROUP),1,tl.float32)
    maximum=tl.maximum(tl.max(tl.abs(w),1),1e-8)
    best=tl.full((R,),float('inf'),tl.float32)
    chosen=tl.div_rn(maximum,7.)
    for step in range(STEPS):
        alpha=1.-step*(.5/(STEPS-1))
        scale=tl.div_rn(maximum*alpha,7.).to(tl.bfloat16).to(tl.float32)
        code=tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(tl.div_rn(w,scale[:,None])),-7.),7.)
        error=(code*scale[:,None]).to(tl.bfloat16).to(tl.float32)-w
        loss=tl.sum((error*error)*importance,1)
        take=loss<best
        chosen=tl.where(take,scale,chosen);best=tl.minimum(best,loss)
    tl.store(S+group,chosen,group<COUNT)


@torch.inference_mode()
def search_scales(weight,inputs,group=128,mode='mse',steps=33):
    n,k=weight.shape
    if weight.dtype!=torch.bfloat16 or not weight.is_cuda or not weight.is_contiguous() or k%group or group not in (32,64,128):raise ValueError('Contiguous BF16 CUDA weights and supported group required')
    if inputs.ndim!=2 or inputs.shape[1]!=k or inputs.device!=weight.device or mode not in ('mse','diagonal') or steps<2:raise ValueError('Invalid scale-search inputs')
    energy=inputs.float().square().mean(0)
    scales=torch.empty(n,k//group,device=weight.device,dtype=torch.bfloat16)
    _search[(triton.cdiv(n*(k//group),4),)](weight,energy,scales,n*(k//group),k//group,group,4,steps,mode=='diagonal',num_warps=4,enable_fp_fusion=False)
    return scales
