"""Marlin projection checks and graph timing before complete-model integration."""
import importlib.metadata
import argparse
import json
import torch
from .common import RESULTS
from .marlin import MarlinLinear
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    name='marlin_kernels'+('_'+args.tag if args.tag else '')+'.json'
    torch.set_num_threads(4)
    torch.manual_seed(888)
    results={'vllm':importlib.metadata.version('vllm'),'torch':torch.__version__,'cases':[]}
    for bits,group in ((4,128),(4,32),(8,128)):
        for n,k in ((24576,4096),(4096,12288),(6144,4096),(4096,4096)):
            w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*0.02
            x=torch.randn(1,1,k,device='cuda',dtype=torch.bfloat16)
            layer=MarlinLinear(w,bits,group)
            expected=torch.nn.functional.linear(x,layer.reference_weight)
            actual=layer(x)
            rel=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
            assert rel<0.005,(bits,group,n,k,rel)
            # Rotate eight distinct packed matrices; the total exceeds H200 L2.
            matrices=[layer.packed]+[layer.packed.clone() for _ in range(7)]
            def fn(packed):
                layer.packed=packed
                return layer(x)
            us=measure(fn,matrices)
            d={'bits':bits,'group':group,'shape':[n,k],'us':us,'relative_rms_vs_dequantized_bf16':rel}
            results['cases'].append(d)
            print(d,flush=True)
            (RESULTS/name).write_text(json.dumps(results,indent=2)+'\n')
            del layer,matrices,w,x,expected,actual


if __name__=='__main__':main()
