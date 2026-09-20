"""Optional GemLite weight-only decode; original BF16 weights serve prefill."""
import torch
import torch.nn.functional as F


class GemLiteProjection:
    @torch.inference_mode()
    def __init__(self, weight, bits=4, group_size=32, asymmetric=False):
        from gemlite import DType, GemLiteLinear
        from gemlite.core import set_fast_gemv_acc, set_native_atomic_bfp16
        set_fast_gemv_acc(False)
        # Split-K BF16 atomics round after every partial sum. Accumulate partial
        # outputs in FP32 too; a FP32 local accumulator alone is insufficient.
        set_native_atomic_bfp16(False)
        self.original = weight
        n, k = weight.shape
        grouped = weight.float().view(n, k // group_size, group_size)
        if asymmetric:
            lo, hi = grouped.amin(-1), grouped.amax(-1)
            scale = ((hi-lo)/(2**bits-1)).clamp_min(1e-8).bfloat16()
            zero = (-lo / scale.float()).bfloat16()
            q = (grouped/scale.float()[..., None]+zero.float()[..., None]).round().clamp(0, 2**bits-1)
        else:
            scale = (grouped.abs().amax(-1).clamp_min(1e-8)/(2**(bits-1)-1)).bfloat16()
            zero = 2**(bits-1)
            q = (grouped/scale.float()[..., None]).round().clamp(1-zero, zero-1)+zero
        self.reference_weight = ((q-(zero.float()[..., None] if torch.is_tensor(zero) else zero))*scale.float()[..., None]).reshape(n,k).bfloat16()
        self.kernel = GemLiteLinear(bits, group_size=group_size, in_features=k,
            out_features=n, input_dtype=DType.BF16, output_dtype=DType.BF16,
            acc_dtype=DType.FP32, scaled_activations=False)
        self.kernel.pack(q.reshape(n,k).to(torch.uint8), scale, zero, fma_mode=False)

    def release_reference(self):
        del self.reference_weight

    def __call__(self, x):
        if x.numel() != x.shape[-1]:
            return F.linear(x, self.original)
        return self.kernel(x)
