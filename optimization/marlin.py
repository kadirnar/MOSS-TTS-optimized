"""Optional vLLM Marlin weight-only projection adapter (BF16 prefill retained)."""
import torch
import torch.nn.functional as F


class MarlinLinear:
    @torch.inference_mode()
    def __init__(self, weight, bits=4, group_size=128, *, signed=None, scales=None):
        # New vLLM releases use the stable libtorch extension and renamed GEMM.
        try:
            import vllm._C
        except ModuleNotFoundError:
            import vllm._C_stable_libtorch
        from vllm.scalar_type import scalar_types
        assert bits in (4,8) and weight.dtype==torch.bfloat16
        self.original=weight
        self.n,self.k=weight.shape
        self.bits=bits
        self.gemm=getattr(torch.ops._C,'marlin_gemm',None)
        if self.gemm is None:
            self.gemm=torch.ops._C.gptq_marlin_gemm
        self.group=self.k if group_size==-1 else group_size
        assert self.k%self.group==0
        self.quant_type=(scalar_types.uint4b8 if bits==4 else scalar_types.uint8b128).id
        maximum=2**(bits-1)-1
        if signed is None:
            if scales is not None:raise ValueError('Scales require supplied integer codes')
            groups=weight.float().reshape(self.n,-1,self.group)
            # BF16 scales match the native dequantization type.
            scales=(groups.abs().amax(-1).clamp_min(1e-8)/maximum).bfloat16()
            signed=(groups/scales.float()[:,:,None]).round().clamp(-maximum,maximum).to(torch.int32)
        else:
            if scales is None or tuple(scales.shape)!=(self.n,self.k//self.group):
                raise ValueError('Calibrated scale shape mismatch')
            if tuple(signed.shape)!=(self.n,self.k):raise ValueError('Calibrated code shape mismatch')
            if bool((signed < -maximum-1).any()|(signed>maximum).any()):raise ValueError('Integer code outside format')
            signed=signed.to(device=weight.device,dtype=torch.int32).reshape(self.n,-1,self.group)
            scales=scales.to(device=weight.device,dtype=torch.bfloat16).contiguous()
        biased=(signed.reshape(self.n,self.k)+2**(bits-1)).T.contiguous()
        factor=32//bits
        packed=torch.zeros((self.k//factor,self.n),device=weight.device,dtype=torch.int32)
        for i in range(factor):
            packed.bitwise_or_(biased[i::factor] << (bits*i))
        self.empty=torch.empty(0,device=weight.device,dtype=torch.int32)
        self.packed=torch.ops._C.gptq_marlin_repack(packed,self.empty,self.k,self.n,bits,False)
        # Marlin scale layout is specified by vLLM's marlin_permute_scales.
        if self.group<self.k:
            perm=[i+8*j for i in range(8) for j in range(8)]
        else:
            perm=[2*i+j for i in range(4) for j in (0,1,8,9,16,17,24,25)]
        unpermuted=scales.T.contiguous()
        self.scales=unpermuted.reshape(-1,len(perm))[:,perm].reshape(-1,self.n).contiguous()
        self.workspace=torch.zeros((self.n//64)*16,device=weight.device,dtype=torch.int32)
        self.reference_weight=(signed.float()*scales.float()[:,:,None]).reshape_as(weight).bfloat16()

    def release_reference(self):
        self.reference_weight=None

    def __call__(self,x):
        if x.numel()!=self.k:
            return F.linear(x,self.original)
        out=self.gemm(x.reshape(1,self.k),None,self.packed,None,
            self.scales,None,None,None,self.empty,self.empty,self.workspace,
            self.quant_type,1,self.n,self.k,True,False,True,False)
        return out.reshape(*x.shape[:-1],self.n)
