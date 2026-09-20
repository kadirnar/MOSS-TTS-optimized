"""Rotate real layer chains to test CUDA 13 register budgets and shared spills."""
import argparse
import json
import statistics

import torch

from .common import RESULTS
from .benchmark_projection_pdl import load_layer,graph_edges
from .benchmark_norm_projection import flatten
from .dp4a_norm_pdl import linear as norm,SELECTED
from .dp4a_layout_pdl_prefetch import linear as projection
from .short_scales import PLAN
from .ptx_resources import NormProjection,DownProjection
from .tune_weight_reads import measure


class Chain:
    def __init__(self,ring,config):
        self.config=config;self.norms={};self.down=None
        if config is None:return
        for entry in ring:
            for name,d in [('up',entry['raw']),('qkv',entry['next'])]:
                key=(name,d['has_residual'])
                if key not in self.norms:
                    c=config.get(name)
                    self.norms[key]=NormProjection(entry,name,**c) if c is not None else None
            if self.down is None and config.get('down') is not None:
                result=self.up(entry,0);self.down=DownProjection(entry,result[1],result[2],**config['down'])

    def normalize(self,entry,name,x,res,d):
        candidate=self.norms.get((name,res is not None))
        if candidate is not None:return candidate(x,res,d['weight'],d['eps'],*entry['weights'][name])
        return norm(x,res,d['weight'],d['eps'],*entry['weights'][name],fused=name=='up',**SELECTED[name])

    def up(self,entry,index):
        d=entry['raw'];res=d['residual'][index:index+1] if d['has_residual'] else None
        return self.normalize(entry,'up',d['x'][index:index+1],res,d)

    def __call__(self,entry,index=0):
        result=self.up(entry,index);summed,hidden,quantized=result
        if self.down is not None:output=self.down(hidden,quantized,*entry['weights']['down'])
        else:output=projection(hidden,*entry['weights']['down'],prequantized=quantized,**PLAN['down'],trigger_mode=3,prefetch=1)
        # The following QKV consumes this chain's summed residual, including
        # at ring wrap. Its normalization therefore always has a residual.
        nd=entry['next'];following=self.normalize(entry,'qkv',output,summed,nd)
        return result,output,following

    def resources(self):
        result={f'{n}_res{int(r)}':v.kernel.resources for (n,r),v in self.norms.items() if v is not None}
        if self.down is not None:result['down']=self.down.kernel.resources
        return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--layers',type=int,choices=(1,36),default=36)
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and at least two rounds required')
    path=RESULTS/f'ptx_resources_{a.tag}.json';assert not path.exists(),'Preserve measurements'
    torch.set_num_threads(4);ring=[load_layer(i) for i in (range(36) if a.layers==36 else [17])]
    # QKV in this synthetic adjacent-layer ring always receives the prior MLP
    # residual. Build its template with that same ADD=True specialization.
    for entry in ring:
        if not entry['next']['has_residual']:
            entry['next']=dict(entry['next']);entry['next']['has_residual']=True
            entry['next']['residual']=torch.zeros_like(entry['next']['x'])
    configs={'control':None,
             'cuda13_dynamic':{n:{'static_shared':False} for n in ('up','down','qkv')},
             'cuda13_static':{n:{} for n in ('up','down','qkv')}}
    for name,caps in [('up',(160,144,128,112,96,80)),('down',(112,96,80,64)),('qkv',(48,40,32))]:
        for cap in caps:
            for spill in (False,True):
                configs[f'{name}_r{cap}_s{int(spill)}']={name:{'registers':cap,'shared_spilling':spill}}
    if a.pilot:configs={k:v for k,v in configs.items() if k in ('control','cuda13_dynamic','cuda13_static','up_r128_s0','up_r128_s1','down_r96_s1','qkv_r40_s1')}
    output={'codebooks':32,'layers':a.layers,'configs':configs,'checks':[],'resources':{},'errors':{},'graph_edges':{},'rows':[],
            'method':'Actual selected MLP/down/following-QKV dependency chains, three saved inputs per layer, rotating all selected weights. CUDA 13.0.88 PTXAS reassembles unchanged Triton instruction bodies with static shared declarations/register limits/optional shared spilling. Selected Triton 3.7/CUDA 12.8 kernels remain the control. No TTFA claim; all timing precedes any profiler initialization.'}
    def save():path.write_text(json.dumps(output,indent=2)+'\n')
    chains={};stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        control=Chain(ring,None)
        for name,config in configs.items():
            try:
                candidate=Chain(ring,config);chains[name]=candidate;output['resources'][name]=candidate.resources()
                for entry in ring:
                    for index in (0,10,31):
                        expected=control(entry,index);actual=candidate(entry,index)
                        counts=[int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]
                        output['checks'].append({'config':name,'layer':entry['layer'],'input':index,'mismatches':counts})
                for _ in range(2):candidate(ring[0])
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):captured=candidate(ring[0])
                edges=graph_edges(graph);assert len(edges)==2 and all(e['type']==1 for e in edges)
                output['graph_edges'][name]=edges;graph.replay();stream.synchronize()
                expected=control(ring[0]);counts=[int((x!=y).sum()) for x,y in zip(flatten(captured),flatten(expected),strict=True)]
                output['checks'].append({'config':name,'layer':ring[0]['layer'],'input':'graph','mismatches':counts})
                print('CHECKED',name,'mismatches',sum(sum(c['mismatches']) for c in output['checks'] if c['config']==name),flush=True)
            except Exception as error:
                output['errors'][name]=repr(error);chains.pop(name,None);print('ERROR',name,repr(error),flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(a.rounds):
        order=list(chains);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        timing={name:measure(chains[name],ring) for name in order}
        output['rows'].append({'round':repeat,'order':order,'us':timing});save();print('ROUND',repeat,timing,flush=True)
    output['summary']={name:{'median_us':statistics.median(r['us'][name] for r in output['rows']),
                             'mismatches':sum(sum(c['mismatches']) for c in output['checks'] if c['config']==name),
                             'paired_gain_us':[r['us']['control']-r['us'][name] for r in output['rows']]}
                       for name in chains}
    save();print('SUMMARY',output['summary'],flush=True)


if __name__=='__main__':main()
