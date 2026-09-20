"""Changing-input private-stream graph checks for experimental G64 kernels."""
import argparse
import json

import torch

from .common import RESULTS
from .benchmark_group64_a64 import Chain,load
from .benchmark_group64_pdl import Chain as A32Chain
from .benchmark_norm_projection import flatten
from .benchmark_group128 import quantize
from .dp4a_packing import pack_interleaved
from .dp4a_group64_a64 import linear


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'group64_validation_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    variants={
        'a32_nominal':A32Chain('g64'),
        'a32_factored':A32Chain('g64',factors={n:1 for n in ('up','down','qkv')}),
        'selected_candidate':Chain({'up':{'factor':1},'qkv':{'factor':1}}),
        'norm_a64':Chain({'up':{'activation_group':64},'qkv':{'activation_group':64}}),
        'all_a64':Chain({'up':{'activation_group':64,'rows':64},'down':{'activation_group':64},'qkv':{'activation_group':64}}),
    }
    def replay_case(fn,x,res,label):
        original=x.clone();saved=None if res is None else res.clone()
        for _ in range(2):fn()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):actual=fn()
        for probe in ('real','zero','spike'):
            x.copy_(original) if probe=='real' else x.zero_()
            if res is not None:res.copy_(saved) if probe=='real' else res.zero_()
            if probe=='spike':x.reshape(-1)[0]=3
            graph.replay();expected=fn()
            exact=all(torch.equal(u.view(torch.uint8),v.view(torch.uint8)) for u,v in zip(flatten(actual),flatten(expected),strict=True))
            finite=all(bool(torch.isfinite(u).all()) for u in flatten(actual))
            checks.append({**label,'probe':probe,'own_eager_graph_bits_exact':exact,'finite':finite})
            assert exact and finite,checks[-1]
        x.copy_(original)
        if res is not None:res.copy_(saved)
    with torch.cuda.stream(stream):
        for i in (0,17,35):
            e=load(i);d=e['raw'];x=d['x'][0];res=d['residual'][0] if d['has_residual'] else None
            for name,chain in variants.items():
                replay_case(lambda:chain(e),x,res,{'layer':i,'config':name})
                print('CHECKED',i,name,flush=True)
            del e
            saved=torch.load(RESULTS/f'gptq_v1_g64_diag_d10/{i:02d}_out.pt',map_location='cuda',weights_only=True)
            w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16()
            x=torch.load(RESULTS/f'calibration_v1/{i:02d}_out.pt',weights_only=True)[231:232].cuda()
            for ag,factor in ((32,0),(32,1),(64,0)):
                def fn():return linear(x,w,s,prequantized=quantize(x,ag),activation_group=ag,factor=factor)
                replay_case(fn,x,None,{'layer':i,'config':f'out_a{ag}_f{factor}'})
                print('CHECKED',i,'out',ag,factor,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'cases':len(checks),'checks':checks,'all_own_eager_graph_bits_exact':True,
        'scope':'Three actual layers, five chained G64/G32 plans plus three attention-output plans, real/zero/spike changing inputs, private CUDA stream. Graph replay matches each candidate own eager arithmetic. This is memory/concurrency validation, not bit-equivalence to selected G32 or speech acceptance.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(checks),flush=True)


if __name__=='__main__':main()
