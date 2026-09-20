"""Compare split-K FP8 projection kernels with an eight-matrix HBM rotation."""
import json
import torch
from .common import RESULTS
from .kernels import quantized_linear_decode
from .split_gemv import split_gemv
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    results={'torch':torch.__version__,'method':'Eight distinct FP8 matrices, 24 kernels per CUDA graph; nine replays; non-default stream warmup; FP32 partial and final accumulation. Actual projection shapes, synthetic values.','cases':{}}
    for n,k in ((4096,12288),(6144,4096),(4096,4096)):
        weights=[(torch.randn(n,k,device='cuda')*25).to(torch.float8_e4m3fn) for _ in range(8)]
        x=torch.randn(1,1,k,device='cuda',dtype=torch.bfloat16)
        scale=torch.rand(n,device='cuda')*.005
        fallback=torch.empty((n,k),device='cuda',dtype=torch.bfloat16)
        original=lambda w:quantized_linear_decode(x,w,scale,fallback)
        expected=original(weights[0])
        trials=[]
        case={'baseline_us':measure(original,weights),'trials':trials}
        for bk in (1024,2048,4096):
            for rows in (1,2,4):
                for warps in (2,4,8):
                    for cache in ('','.cg'):
                        config={'block_k':bk,'rows':rows,'warps':warps,'cache':cache}
                        actual=split_gemv(x,weights[0],scale,**config)
                        error=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
                        assert error<.001,(n,k,config,error)
                        trials.append({**config,'us':measure(lambda w:split_gemv(x,w,scale,**config),weights),
                            'relative_rms':error,'bitwise_equal':bool(torch.equal(actual,expected))})
        case['best']=min(trials,key=lambda t:t['us'])
        results['cases'][f'{n}x{k}']=case
        (RESULTS/'split_fp8_gemv.json').write_text(json.dumps(results,indent=2)+'\n')
        print(n,k,'baseline',case['baseline_us'],'best',case['best'],flush=True)
        del weights,fallback,x,expected,actual


if __name__=='__main__':main()
