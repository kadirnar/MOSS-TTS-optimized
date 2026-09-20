"""Private-stream changing-input checks for experimental split PDL chains."""
import argparse
import json

import torch

from .common import RESULTS
from .benchmark_split_norm_pdl import Chain
from .benchmark_projection_pdl import load_layer
from .benchmark_norm_projection import flatten


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'split_norm_pdl_validation_{args.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    configs={'qkv_t1':{'qkv':{}},'up_t1':{'up':{}},'both_t1':{'qkv':{},'up':{}},
        'both_t3':{'qkv':{'norm_trigger':3},'up':{'norm_trigger':3}},
        'up_r160':{'up':{'registers':160}},'up_r128':{'up':{'registers':128}}}
    control=Chain({})
    with torch.cuda.stream(stream):
        for index in (0,17,35):
            entry=load_layer(index);data=entry['raw'];x=data['x'][0];original=x.clone()
            residual=data['residual'][0] if data['has_residual'] else None
            saved=None if residual is None else residual.clone()
            for name,options in configs.items():
                candidate=Chain(options)
                for _ in range(2):candidate(entry)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):actual=candidate(entry)
                for probe in ('real','zero','spike'):
                    if probe=='real':x.copy_(original)
                    else:x.zero_()
                    if residual is not None:
                        if probe=='real':residual.copy_(saved)
                        else:residual.zero_()
                    if probe=='spike':x.reshape(-1)[0]=3
                    graph.replay();expected=control(entry)
                    exact=all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(flatten(actual),flatten(expected),strict=True))
                    checks.append({'layer':index,'config':name,'probe':probe,'all_bits_exact':exact});assert exact,checks[-1]
                print('CHECKED',index,name,flush=True)
            x.copy_(original)
            if residual is not None:residual.copy_(saved)
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'cases':len(checks),'all_exact':True,'checks':checks,'scope':'Six split-PDL schedules, three layers, private-stream graphs replayed with real/zero/spike inputs. Entire returned projection/residual/quantization chains match selected fused-normalization control bitwise. Includes two register caps with local spills.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(checks),flush=True)


if __name__=='__main__':main()
