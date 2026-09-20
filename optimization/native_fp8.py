"""Hopper FP8 tensor-core projection with fused dynamic activation quantization."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _quantize_row(X,Y,S,K:tl.constexpr,B:tl.constexpr):
    c=tl.arange(0,B)
    x=tl.load(X+c,c<K,0).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x),0)/448,1e-8)
    tl.store(Y+c,x/scale,c<K)
    tl.store(S,scale)


@triton.jit
def _quantize_rows(X,Y,S,K:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    c=tl.arange(0,B)
    x=tl.load(X+row*K+c,c<K,0).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x),0)/448,1e-8)
    tl.store(Y+row*K+c,x/scale,c<K)
    tl.store(S+row,scale)


def fp8_prefill_linear(x,weight,weight_scale,backend='torch'):
    """Experimental W8A8 prefill; dynamic per-token activation scaling."""
    n,k=weight.shape
    m=x.numel()//k
    quantized=torch.empty((m,k),device=x.device,dtype=torch.float8_e4m3fn)
    scales=torch.empty((m,1),device=x.device,dtype=torch.float32)
    _quantize_rows[(m,)](x,quantized,scales,k,triton.next_power_of_2(k))
    if backend=='torch':
        out=torch._scaled_mm(quantized,weight.T,scale_a=scales,scale_b=weight_scale.view(1,n),out_dtype=x.dtype,use_fast_accum=False)
    elif backend=='vllm':
        try:import vllm._C
        except ModuleNotFoundError:import vllm._C_stable_libtorch
        out=torch.empty((m,n),device=x.device,dtype=x.dtype)
        torch.ops._C.cutlass_scaled_mm(out,quantized,weight.T,scales,weight_scale,None)
    else:raise ValueError('Unsupported FP8 prefill backend')
    return out.reshape(*x.shape[:-1],n)


class ScaledFP8Linear:
    @torch.inference_mode()
    def __init__(self,weight):
        try:
            import vllm._C
        except ModuleNotFoundError:
            import vllm._C_stable_libtorch
        self.original=weight
        self.n,self.k=weight.shape
        self.scales=weight.float().abs().amax(-1).clamp_min(1e-8)/448
        self.packed=(weight.float()/self.scales[:,None]).to(torch.float8_e4m3fn)

    def __call__(self,x):
        if x.numel()!=self.k:
            return F.linear(x,self.original)
        q=torch.empty((1,self.k),device=x.device,dtype=torch.float8_e4m3fn)
        scale=torch.empty(1,device=x.device,dtype=torch.float32)
        _quantize_row[(1,)](x,q,scale,self.k,triton.next_power_of_2(self.k))
        out=torch.empty((1,self.n),device=x.device,dtype=x.dtype)
        torch.ops._C.cutlass_scaled_mm(out,q,self.packed.T,scale,self.scales,None)
        return out.reshape(*x.shape[:-1],self.n)
