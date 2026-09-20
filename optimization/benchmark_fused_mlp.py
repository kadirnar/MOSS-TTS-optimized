"""Cold weight rotation: same projection plus SiLU, fused versus separate."""
import json
import argparse
import torch
from .common import RESULTS
from .kernels import quantized_linear_decode,linear_decode,silu_mul
from .fused_mlp import fp8_silu_decode
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dtype',choices=('fp8','bf16'),default='fp8')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(444)
    weight=(torch.randn(24576,4096,device='cuda')*(24 if args.dtype=='fp8' else 0.02)).to(torch.float8_e4m3fn if args.dtype=='fp8' else torch.bfloat16)
    fallback=torch.empty_like(weight,dtype=torch.bfloat16)
    scale=torch.rand(24576,device='cuda')*0.005 if args.dtype=='fp8' else None
    x=torch.randn(1,1,4096,device='cuda',dtype=torch.bfloat16)
    weights=[weight]+[weight.clone() for _ in range(7)]
    def original(w):return silu_mul(quantized_linear_decode(x,w,scale,fallback) if scale is not None else linear_decode(x,w))
    expected=original(weight)
    result={'separate_us':measure(original,weights),'fused':{}}
    for warps in (2,4,8,16):
        actual=fp8_silu_decode(x,weight,scale,warps)
        rel=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
        assert rel<0.001,(warps,rel)
        result['fused'][str(warps)]={'us':measure(lambda w:fp8_silu_decode(x,w,scale,warps),weights),
            'relative_rms_vs_existing':rel,'bitwise_equal':bool(torch.equal(actual,expected))}
    print(json.dumps(result,indent=2),flush=True)
    (RESULTS/f'fused_{args.dtype}_mlp.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
