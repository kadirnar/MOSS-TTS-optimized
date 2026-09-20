"""Changing-input private-graph sanitizer cases for the resource experiments."""
import argparse
import json

import torch

from .common import RESULTS
from .benchmark_projection_pdl import load_layer,graph_edges
from .benchmark_ptx_resources import Chain
from .benchmark_gateup_warp_specialized import Candidate
from .benchmark_norm_projection import flatten


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    output=RESULTS/f'projection_resources_validation_{a.tag}.json';assert not output.exists(),'Preserve evidence'
    torch.set_num_threads(4);entries=[load_layer(i) for i in (0,17,35)]
    for e in entries:
        if not e['next']['has_residual']:
            e['next']=dict(e['next']);e['next']['has_residual']=True;e['next']['residual']=torch.zeros_like(e['next']['x'])
    control=Chain(entries,None)
    candidates={'static':Chain(entries,{n:{} for n in ('up','down','qkv')}),
                'up_r128_s1':Chain(entries,{'up':{'registers':128,'shared_spilling':True}}),
                'down_r112_s1':Chain(entries,{'down':{'registers':112,'shared_spilling':True}}),
                'qkv_r48_s1':Chain(entries,{'qkv':{'registers':48,'shared_spilling':True}}),
                'parallel_branches':Candidate({'mode':0,'max_registers':192,'worker_registers':160}),
                'parallel_rows':Candidate({'mode':1,'max_registers':128,'worker_registers':128})}
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());checks=[]
    with torch.cuda.stream(stream):
        for entry in entries:
            d=entry['raw'];original_x=d['x'].clone();original_res=d['residual'].clone()
            for name,fn in candidates.items():
                for _ in range(2):fn(entry)
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):actual=fn(entry)
                edges=graph_edges(graph);assert len(edges)==2 and all(e['type']==1 for e in edges)
                for source in (0,10,31):
                    d['x'][:1].copy_(original_x[source:source+1]);d['residual'][:1].copy_(original_res[source:source+1])
                    expected=control(entry);graph.replay();stream.synchronize()
                    counts=[int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]
                    assert not any(counts),(name,entry['layer'],source,counts)
                    assert all(torch.isfinite(t).all() for t in flatten(actual))
                    checks.append({'variant':name,'layer':entry['layer'],'source':source,'mismatches':counts})
                d['x'].copy_(original_x);d['residual'].copy_(original_res)
            print('CHECKED',entry['layer'],flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'all_exact':True,'checks':checks,'cases':len(checks),
            'scope':'Six configurations, layers 0/17/35, three changing inputs per private captured graph. Selected control and all returned residual, projection and quantized buffers compared. PDL graph edge types checked. Rejected performance experiments only; no model promotion or full-quality claim.'}
    output.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(checks),flush=True)


if __name__=='__main__':main()
