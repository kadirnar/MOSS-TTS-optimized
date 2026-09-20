"""Rotating real-layer chains for single-CTA normalization with PDL."""
import argparse
import hashlib
import json
import statistics

import torch

from .common import RESULTS
from .benchmark_projection_pdl import load_layer,graph_edges
from .benchmark_norm_projection import flatten
from .dp4a_norm_pdl import linear as norm,SELECTED
from .dp4a_layout_pdl_prefetch import linear as projection
from .short_scales import PLAN
from .bulk_prefetch import configured as original
from .dp4a_split_norm_pdl import configured
from .benchmark_bulk_address import Chain as BaseChain
from .tune_weight_reads import measure


class Chain(BaseChain):
    def __init__(self,options):
        super().__init__({'qkv':{'address_mode':1}})
        self.options=options
        for name,settings in options.items():self.functions[name]=configured(**settings)

    def __call__(self,*args,**kwargs):
        value=super().__call__(*args,**kwargs)
        if kwargs.get('audit',False):
            for name in self.options:value[1][name+'_normalizer']=self.functions[name].dispatch.last_normalizer
        return value


def configs(pilot=False,resources=False):
    result={'control':{}}
    if resources:
        result['up_uncapped']={'up':{}}
        for cap in (128,144,160,168,176,192):result[f'up_r{cap}']={'up':{'registers':cap}}
        return result
    for stage in ('qkv','up','both'):
        for mode in (1,2,3):
            names=('qkv','up') if stage=='both' else (stage,)
            result[f'{stage}_t{mode}']={name:{'norm_trigger':mode} for name in names}
    result['both_t1_no_hint']={name:{'norm_trigger':1,'divisor':0} for name in ('qkv','up')}
    if pilot:result={name:result[name] for name in ('control','both_t1')}
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--layers',type=int,choices=(1,36),default=36)
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--pilot',action='store_true');p.add_argument('--resources',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and at least two rounds required')
    path=RESULTS/f'split_norm_pdl_{a.tag}.json';assert not path.exists(),'Preserve measurements'
    torch.set_num_threads(4);ring=[load_layer(i) for i in (range(36) if a.layers==36 else [17])]
    options=configs(a.pilot,a.resources);chains={};result={'codebooks':32,'layers':a.layers,'configs':options,'checks':[],'resources':{},'errors':{},'graph_edges':{},'rows':[],
        'method':'Rotating actual 36-layer MLP/down/following-QKV chain, three saved inputs per layer. Selected bulk-prefetch/QKV-address control versus one norm-quant producer per stage and overlapping projection consumers. Same original arithmetic, explicit programmatic waits. No serving or TTFA claim. All timing before any profiler.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        control=Chain({})
        for name,config in options.items():
            try:
                candidate=Chain(config);chains[name]=candidate
                for e in ring:
                    for index in (0,10,31):
                        expected=control(e,index);actual=candidate(e,index)
                        result['checks'].append({'config':name,'layer':e['layer'],'input':index,
                            'mismatches':[int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]})
                _,kernels=candidate(ring[0],audit=True)
                result['resources'][name]={n:{'registers':k.n_regs,'spills':k.n_spills,'shared_bytes':k.metadata.shared,
                    'bulk_prefetch_ptx_instructions':k.asm['ptx'].count('cp.async.bulk.prefetch'),
                    'runtime_modulo_in_ptx':'rem.' in k.asm['ptx'],
                    'cubin_sha256':hashlib.sha256(k.asm['cubin']).hexdigest()} for n,k in kernels.items()}
                for _ in range(2):candidate(ring[0])
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):actual=candidate(ring[0])
                edges=graph_edges(graph);assert len(edges)==2+len(config) and all(e['type']==1 for e in edges)
                result['graph_edges'][name]=edges;graph.replay();stream.synchronize();expected=control(ring[0])
                result['checks'].append({'config':name,'layer':ring[0]['layer'],'input':'graph',
                    'mismatches':[int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]})
                mismatches=sum(sum(c['mismatches']) for c in result['checks'] if c['config']==name)
                print('CHECKED',name,'mismatches',mismatches,flush=True)
            except Exception as error:
                result['errors'][name]=repr(error);chains.pop(name,None);print('ERROR',name,repr(error),flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(a.rounds):
        order=list(chains);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        timing={name:measure(chains[name],ring) for name in order}
        result['rows'].append({'round':repeat,'order':order,'us':timing});save();print('ROUND',repeat,flush=True)
    result['summary']={name:{'median_us':statistics.median(r['us'][name] for r in result['rows']),
        'mismatches':sum(sum(c['mismatches']) for c in result['checks'] if c['config']==name),
        'paired_gain_us':[r['us']['control']-r['us'][name] for r in result['rows']]} for name in chains}
    save();print('BEST',sorted(result['summary'],key=lambda n:result['summary'][n]['median_us'])[:8],flush=True)


if __name__=='__main__':main()
