"""Changed-input private-stream prefill activation and residual graph checks."""
import argparse
import json

import torch
import torch.nn.functional as F

from .common import RESULTS
from .kernels import add_rmsnorm,rmsnorm
from .prefill_pointwise import silu_mul


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'prefill_pointwise_validation_{args.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.manual_seed(92039);torch.set_num_threads(4)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());checks=[]
    with torch.cuda.stream(stream):
        weight=torch.randn(4096,device='cuda',dtype=torch.bfloat16)
        for n in (2,3,6,127,128,145,160,256,511,512,1023):
            x=torch.empty((1,n,24576),device='cuda',dtype=torch.bfloat16)
            a=torch.empty((1,n,4096),device='cuda',dtype=torch.bfloat16);residual=torch.empty_like(a)
            x.normal_();a.normal_();residual.normal_()
            for _ in range(2):silu_mul(x);add_rmsnorm(a,residual,weight,1e-6)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                actual=silu_mul(x);summed,normalized=add_rmsnorm(a,residual,weight,1e-6)
            for probe in ('random','zero','spike'):
                if probe=='random':x.normal_();a.normal_();residual.normal_()
                else:
                    x.zero_();a.zero_();residual.zero_()
                    if probe=='spike':x[...,0]=32;x[...,12288]=-16;a[...,0]=3;residual[...,1]=-2
                graph.replay();gate,up=x.chunk(2,-1);ref=F.silu(gate)*up
                rs=a+residual;rn=rmsnorm(rs,weight,1e-6)
                counts=[int((u.view(torch.int16)!=v.view(torch.int16)).sum()) for u,v in ((actual,ref),(summed,rs),(normalized,rn))]
                checks.append({'tokens':n,'probe':probe,'bit_mismatches':counts});assert not any(counts),checks[-1]
            if n==6:
                index=torch.arange(65536,device='cuda');bits=index.to(torch.int16).view(torch.bfloat16)
                x.zero_();x[0,index//12288,index%12288]=bits
                for value in (1.0,-2.0,0.0):
                    x[...,12288:]=value;graph.replay();gate,up=x.chunk(2,-1);ref=F.silu(gate)*up
                    count=int((actual.view(torch.int16)!=ref.view(torch.int16)).sum())
                    checks.append({'tokens':n,'probe':'all_bf16_bits','up_value':value,'bit_mismatches':[count]});assert count==0
            print('CHECKED',n,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'all_exact':True,'cases':len(checks),'checks':checks,
        'scope':'Private-stream CUDA graphs replayed with changed random/zero/spike inputs at eleven lengths through 1023; all 65536 BF16 gate patterns with three up values. Bitwise activation product, residual sum and normalized output, including zero signs and NaN bits.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(checks),flush=True)


if __name__=='__main__':main()
