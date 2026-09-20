"""Group-reordered INT4 projection correctness and HBM timing."""
import argparse
import json
import torch
from .common import RESULTS
from .grouped_int4 import grouped_int4
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--vector-x',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(119)
    result={'torch':torch.__version__,'vector_x':args.vector_x,'method':'Eight packed matrices; graph timing; synthetic integer weights at actual projection sizes. FP32 group dot products then group scaling. This changes summation order, not quantization.','cases':[]}
    for n,k in ((24576,4096),(4096,12288),(6144,4096),(4096,4096)):
        for group in (32,128):
            signed=torch.randint(-7,8,(n,k),device='cuda',dtype=torch.int8)
            scale=torch.rand(n,k//group,device='cuda')*.02
            unsigned=signed.to(torch.uint8)&15
            packed=(unsigned[:,::2]|(unsigned[:,1::2]<<4)).contiguous()
            weights=[packed]+[packed.clone() for _ in range(7)]
            x=torch.randn(1,1,k,device='cuda',dtype=torch.bfloat16)
            expected=torch.nn.functional.linear(x.float(),(signed.float().view(n,-1,group)*scale[:,:,None]).view(n,k)).bfloat16()
            trials=[]
            for warps in (1,2,4,8,16):
                fn=lambda w:grouped_int4(x,w,scale,group,warps,args.vector_x)
                actual=fn(packed)
                rel=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
                assert rel<.001,(n,k,group,warps,rel)
                trials.append({'warps':warps,'us':measure(fn,weights),'relative_rms_vs_fp32':rel})
            case={'shape':[n,k],'group':group,'trials':trials,'best':min(trials,key=lambda t:t['us'])}
            result['cases'].append(case)
            print(case,flush=True)
            (RESULTS/('grouped_int4_kernels'+('_vector' if args.vector_x else '')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            del signed,unsigned,packed,weights,scale,x,expected,actual


if __name__=='__main__':main()
