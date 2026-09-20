"""CUTLASS FP8 GEMM versus the current FP8 GEMV at real backbone shapes."""
import importlib.metadata
import json
import torch
from .common import RESULTS
from .kernels import quantized_linear_decode
from .native_fp8 import ScaledFP8Linear
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(52)
    result={'torch':torch.__version__,'vllm':importlib.metadata.version('vllm'),'cases':[]}
    for n,k in ((24576,4096),(4096,12288),(6144,4096),(4096,4096)):
        x=torch.randn(1,1,k,device='cuda',dtype=torch.bfloat16)
        weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*0.02
        layer=ScaledFP8Linear(weight)
        sx=x.float().abs().max()/448
        xq=(x.float()/sx).to(torch.float8_e4m3fn).float()*sx
        wq=layer.packed.float()*layer.scales[:,None]
        expected=torch.nn.functional.linear(xq,wq)
        actual=layer(x)
        rel=((actual.float()-expected).square().mean().sqrt()/expected.square().mean().sqrt()).item()
        assert rel<0.005,(n,k,rel)
        matrices=[layer.packed]+[layer.packed.clone() for _ in range(7)]
        def native(w):
            layer.packed=w
            return layer(x)
        def current(w):
            return quantized_linear_decode(x,w,layer.scales,weight)
        d={'shape':[n,k],'native_fp8_us':measure(native,matrices),
           'current_weight_only_us':measure(current,matrices),'relative_rms_vs_dequantized_fp32':rel}
        result['cases'].append(d)
        print(d,flush=True)
        (RESULTS/'native_fp8_kernels.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
