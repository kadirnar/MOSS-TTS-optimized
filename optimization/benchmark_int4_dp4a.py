"""DP4A W4A8: exact integer-dot reference and cold-weight timings."""
import argparse
import json
import numpy as np
import torch
from .common import RESULTS
from .int4_dp4a import int4_dp4a
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reciprocal',action='store_true')
    parser.add_argument('--grouped-activation',action='store_true')
    args=parser.parse_args()
    if args.grouped_activation and not args.reciprocal:raise ValueError('Grouped quantization uses reciprocal rounding')
    torch.set_num_threads(4)
    torch.manual_seed(471)
    result={'torch':torch.__version__,'reciprocal':args.reciprocal,'grouped_activation':args.grouped_activation,'method':'Eight packed matrix rotation, dynamic activation quantization included. Synthetic signed INT4 weights; reference uses the same rounded INT8 activations and FP32 dequantized weights. Reciprocal variant uses an independent NumPy FP32 reference for scaling/rounding. Kernel correctness only, not a model quality test.','cases':[]}
    for n,k in ((24576,4096),(4096,12288),(6144,4096),(4096,4096)):
        for group in (32,128):
            signed=torch.randint(-8,8,(n,k),device='cuda',dtype=torch.int8)
            scales=torch.rand(n,k//group,device='cuda')*.02
            unsigned=signed.to(torch.uint8)&15
            packed=(unsigned[:,::2]|(unsigned[:,1::2]<<4)).contiguous()
            weights=[packed]+[packed.clone() for _ in range(7)]
            x=torch.randn(1,1,k,device='cuda',dtype=torch.bfloat16)
            sx=x.float().abs().max()/127
            q=(x.float()/sx).round().clamp(-127,127)
            if args.reciprocal:
                x_cpu=x.float().cpu().numpy()
                scale_cpu=np.float32(np.max(np.abs(x_cpu))/np.float32(127))
                inv=np.float32(np.float32(1)/scale_cpu)
                q=torch.from_numpy(np.rint(x_cpu*inv).clip(-127,127)).cuda()
                sx=torch.tensor(scale_cpu,device='cuda')
            if args.grouped_activation:
                x_cpu=x.float().cpu().numpy().reshape(-1,group)
                scale_cpu=np.maximum(np.max(np.abs(x_cpu),axis=1,keepdims=True)/np.float32(127),np.float32(1e-8))
                inv=np.float32(1)/scale_cpu
                q=torch.from_numpy(np.rint(x_cpu*inv).clip(-127,127)).cuda()
                sx=torch.from_numpy(scale_cpu).cuda()
            expected=torch.nn.functional.linear((q*sx).reshape(1,1,k),(signed.float().view(n,-1,group)*scales[:,:,None]).view(n,k)).bfloat16()
            trials=[]
            for warps in (1,2,4,8):
                fn=lambda w:int4_dp4a(x,w,scales,group,warps,args.reciprocal,grouped_activation=args.grouped_activation)
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):actual=fn(packed)
                torch.cuda.current_stream().wait_stream(stream)
                rel=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
                assert rel<.001,(n,k,group,warps,rel)
                trials.append({'warps':warps,'us':measure(fn,weights),'relative_rms_vs_reference':rel})
            case={'shape':[n,k],'group':group,'trials':trials,'best':min(trials,key=lambda t:t['us'])}
            result['cases'].append(case)
            print(case,flush=True)
            (RESULTS/('int4_dp4a_kernels'+('_reciprocal' if args.reciprocal else '')+('_grouped' if args.grouped_activation else '')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            del signed,unsigned,packed,weights,scales,x,q,expected,actual


if __name__=='__main__':main()
