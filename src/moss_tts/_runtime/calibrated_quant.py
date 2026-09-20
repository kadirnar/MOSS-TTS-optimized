"""GPTQ-style block error feedback with static BF16 group scales.

Algorithm reference: IST-DASLab/gptq, Apache-2.0, commit
2d65066eeb06a5c9ff5184d8cebdf33662c67faf (see ../_licenses/GPTQ-LICENSE).
Adaptations: explicit symmetric [-7,7] codes matching existing RTN controls,
BF16-rounded dequantization in feedback, static groups, optional activation
ordering, and a fused Triton column quantization/update kernel. Original model
weights are never overwritten. Prefill remains BF16 in inference adapters.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['start','column'])
def _column(W,E,Q,S,H,P,start,column,N:tl.constexpr,K:tl.constexpr,
            GROUP:tl.constexpr,BLOCK:tl.constexpr,R:tl.constexpr):
    rows=tl.program_id(0)*R+tl.arange(0,R)
    cols=tl.arange(0,BLOCK)
    original=tl.load(P+start+column)
    scale=tl.load(S+rows*(K//GROUP)+original//GROUP,rows<N,1).to(tl.float32)
    value=tl.load(W+rows*BLOCK+column,rows<N,0)
    code=tl.extra.cuda.libdevice.nearbyint(tl.div_rn(value,scale))
    code=tl.minimum(tl.maximum(code,-7),7)
    quantized=(code*scale).to(tl.bfloat16).to(tl.float32)
    diagonal=tl.load(H+(start+column)*K+start+column)
    error=tl.div_rn(value-quantized,diagonal)
    tl.store(Q+rows*K+original,code.to(tl.int8),rows<N)
    tl.store(E+rows*BLOCK+column,error,rows<N)
    h=tl.load(H+(start+column)*K+start+cols)
    old=tl.load(W+rows[:,None]*BLOCK+cols[None,:],rows[:,None]<N,0)
    updated=old-error[:,None]*h[None,:]
    tl.store(W+rows[:,None]*BLOCK+cols[None,:],updated,(rows[:,None]<N)&(cols[None,:]>=column))


def static_scales(weight,group=32):
    n,k=weight.shape
    return (weight.float().view(n,k//group,group).abs().amax(-1).clamp_min(1e-8)/7).bfloat16()


def dequantize(signed,scales,group):
    return (signed.float().view(signed.shape[0],-1,group)*scales.float()[:,:,None]).reshape_as(signed).bfloat16()


@torch.inference_mode()
def gptq(weight,inputs,group=32,damping=.01,actorder=True,block=128,backend='triton',scales_override=None):
    n,k=weight.shape
    if k%block or k%group:raise ValueError('Dimensions must be divisible by group/block sizes')
    if inputs.ndim!=2 or inputs.shape[1]!=k:raise ValueError('Calibration input shape mismatch')
    scales=static_scales(weight,group) if scales_override is None else scales_override
    if scales.shape!=(n,k//group) or scales.dtype!=torch.bfloat16 or scales.device!=weight.device or not scales.is_contiguous() or not torch.isfinite(scales).all() or not (scales>0).all():raise ValueError('Invalid static group scales')
    x=inputs.float()
    h=(x.T@x)*(2/x.shape[0])
    dead=h.diagonal()==0
    h[dead,dead]=1
    w=weight.float().clone()
    w[:,dead]=0
    perm=torch.argsort(h.diagonal(),descending=True) if actorder else torch.arange(k,device=w.device)
    w=w[:,perm].contiguous()
    h=h[perm][:,perm].contiguous()
    h.diagonal().add_(damping*h.diagonal().mean())
    hinv=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)),upper=True).contiguous()
    del h,x
    signed=torch.empty((n,k),device=w.device,dtype=torch.int8)
    for start in range(0,k,block):
        end=start+block
        chunk=w[:,start:end].contiguous().clone()
        errors=torch.zeros_like(chunk)
        for column in range(block):
            if backend=='triton':
                _column[(triton.cdiv(n,16),)](chunk,errors,signed,scales,hinv,perm,
                    start,column,n,k,group,block,16,num_warps=4,enable_fp_fusion=False)
            elif backend=='torch':
                original=perm[start+column]
                scale=scales[:,original//group].float()
                value=chunk[:,column].clone()
                code=(value/scale).round().clamp(-7,7)
                quantized=(code*scale).bfloat16().float()
                error=(value-quantized)/hinv[start+column,start+column]
                signed[:,original]=code.to(torch.int8)
                chunk[:,column:]-=error[:,None]*hinv[start+column,start+column:end][None,:]
                errors[:,column]=error
            else:raise ValueError('Unknown GPTQ backend')
        if end<k:w[:,end:]-=errors@hinv[start:end,end:]
    return signed,scales
