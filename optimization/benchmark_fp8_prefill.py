"""Low-token-count FP8 tensor-core prefill versus BF16 GEMM."""
import json
import torch
from .common import RESULTS
from .native_fp8 import fp8_prefill_linear
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(287)
    results={'torch':torch.__version__,'method':'Eight distinct matrices, graph replay timing. Dynamic per-row activation quantization included. Synthetic values, actual MOSS projection sizes.','cases':[]}
    for n,k in ((24576,4096),(4096,12288),(6144,4096),(4096,4096)):
        original=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
        scale=original.float().abs().amax(-1).clamp_min(1e-8)/448
        weight=(original.float()/scale[:,None]).to(torch.float8_e4m3fn)
        weights=[weight]+[weight.clone() for _ in range(7)]
        originals=[original]+[original.clone() for _ in range(7)]
        for m in (16,32,64,128,256):
            x=torch.randn(1,m,k,device='cuda',dtype=torch.bfloat16)
            sx=x.float().abs().amax(-1,keepdim=True)/448
            xq=(x.float()/sx).to(torch.float8_e4m3fn).float()*sx
            expected=torch.nn.functional.linear(xq,weight.float()*scale[:,None])
            row={'shape':[m,n,k],'bf16_us':measure(lambda w:torch.nn.functional.linear(x,w),originals),'backends':{}}
            for backend in ('torch','vllm'):
                fn=lambda w:fp8_prefill_linear(x,w,scale,backend)
                actual=fn(weight)
                rel=((actual.float()-expected).square().mean().sqrt()/expected.square().mean().sqrt()).item()
                assert rel<.005,(m,n,k,backend,rel)
                row['backends'][backend]={'us':measure(fn,weights),'relative_rms_vs_dequantized_fp32':rel}
            results['cases'].append(row)
            print(row,flush=True)
            (RESULTS/'fp8_prefill_kernels.json').write_text(json.dumps(results,indent=2)+'\n')
        del original,originals,weight,weights,x,xq,expected,actual


if __name__=='__main__':main()
