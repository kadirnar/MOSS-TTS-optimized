"""Actual projection shapes, packed GemLite weights, and HBM-scale rotation."""
import importlib.metadata
import json
import torch
import gemlite
from .common import RESULTS
from .gemlite_linear import GemLiteProjection
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(888)
    results = {'gemlite': importlib.metadata.version('gemlite'),
        'torch': torch.__version__, 'method': 'Eight distinct packed matrices rotated in a CUDA graph; FP32 accumulation; symmetric RTN weights', 'cases': []}
    for bits, group in ((4,128), (4,32), (8,128)):
        for n,k in ((24576,4096),(4096,12288),(6144,4096),(4096,4096)):
            w = torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*0.02
            x = torch.randn(1,1,k,device='cuda',dtype=torch.bfloat16)
            layer = GemLiteProjection(w,bits,group)
            expected = torch.nn.functional.linear(x,layer.reference_weight)
            actual = layer(x)
            rel = ((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
            assert rel < 0.005, (bits,group,n,k,rel)
            matrices = [layer.kernel.W_q]+[torch.nn.Parameter(layer.kernel.W_q.detach().clone(),requires_grad=False) for _ in range(7)]
            def fn(packed):
                layer.kernel.W_q = packed
                return layer(x)
            us = measure(fn,matrices)
            case = {'bits':bits,'group':group,'shape':[n,k],'us':us,'relative_rms_vs_dequantized_bf16':rel}
            print(case,flush=True)
            results['cases'].append(case)
            (RESULTS/'gemlite_kernels.json').write_text(json.dumps(results,indent=2)+'\n')
            del layer, matrices, w, x, expected, actual
    gemlite.cache_config(str(RESULTS/'gemlite_autotune.json'))


if __name__ == '__main__':
    main()
