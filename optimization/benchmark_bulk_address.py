"""Rotating real-layer chains for bulk-prefetch address specialization."""
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
from .bulk_address import configured
from .tune_weight_reads import measure


class Chain:
    def __init__(self,options):
        self.options=options
        self.functions={n:(configured('projection' if n=='down' else 'norm',**options[n]) if n in options else original('projection' if n=='down' else 'norm',divisor=16)) for n in ('up','down','qkv')}

    def __call__(self,e,index=0,audit=False):
        d=e['raw'];nd=e['next'];w=e['weights'];kernels={}
        up,k=self.functions.get('up',norm)(d['x'][index:index+1],d['residual'][index:index+1] if d['has_residual'] else None,
            d['weight'],d['eps'],*w['up'],fused=True,**SELECTED['up'],return_kernel=True)
        kernels['up']=k;summed,hidden,quantized=up
        down,k=self.functions.get('down',projection)(hidden,*w['down'],prequantized=quantized,
            **PLAN['down'],trigger_mode=3,prefetch=1,return_kernel=True)
        kernels['down']=k
        following,k=self.functions.get('qkv',norm)(down,summed,nd['weight'],nd['eps'],*w['qkv'],
            **SELECTED['qkv'],return_kernel=True)
        kernels['qkv']=k
        return ((up,down,following),kernels) if audit else (up,down,following)


def configs(pilot=False):
    output={'control':{},'wrapper_control':{n:{'address_mode':0} for n in ('up','down','qkv')}}
    for stage in ('up','down','qkv'):
        for mode in (1,2):output[f'{stage}_m{mode}']={stage:{'address_mode':mode}}
    for mode in (1,2):
        for divisor in (8,16,32,64):output[f'all_m{mode}_d{divisor}']={n:{'address_mode':mode,'divisor':divisor} for n in ('up','down','qkv')}
    if pilot:output={k:v for k,v in output.items() if k in ('control','wrapper_control','all_m1_d16','all_m2_d16')}
    return output


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--layers',type=int,choices=(1,36),default=36)
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and at least two rounds required')
    path=RESULTS/f'bulk_address_{a.tag}.json';assert not path.exists(),'Preserve measurements'
    torch.set_num_threads(4);ring=[load_layer(i) for i in (range(36) if a.layers==36 else [17])]
    options=configs(a.pilot);chains={};result={'codebooks':32,'layers':a.layers,'configs':options,'checks':[],'resources':{},'errors':{},'graph_edges':{},'rows':[],
        'method':'Rotating actual 36-layer MLP/down/following-QKV chain, three saved inputs per layer. Selected bulk-prefetch control versus specialized zero-lookahead block index and full-CTA constant span. Same original arithmetic and waits. No serving or TTFA claim. All timing before any profiler.'}
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
                edges=graph_edges(graph);assert len(edges)==2 and all(e['type']==1 for e in edges)
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
