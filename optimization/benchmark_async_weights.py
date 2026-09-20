"""G32 MLP/down/QKV chains with asynchronous down-projection weight staging."""
import argparse
import json
import statistics

import torch
from .common import RESULTS
from .benchmark_projection_pdl import load_layer,graph_edges
from .benchmark_norm_projection import flatten
from .benchmark_bulk_address import Chain as SelectedChain
from .dp4a_async_weights import configured
from .tune_weight_reads import measure


class Chain(SelectedChain):
    def __init__(self,options=None):
        super().__init__({'qkv':{'address_mode':1}})
        if options is not None:
            options=dict(options);tile=options.pop('tile',{})
            fn=configured(**options)
            self.functions['down']=lambda *args,**kwargs:fn(*args,**{**kwargs,**tile})


def configs(pilot):
    choices={'control':None}
    for mode,swizzles in ((1,(1,2,4,8)),(2,(0,32,64,128)),(3,(0,128)),(4,(0,128))):
        for swizzle in swizzles:
            for divisor in (0,16):
                choices[f'm{mode}_s{swizzle}_d{divisor}']={'mode':mode,'swizzle':swizzle,'divisor':divisor}
    for s in (1,4,8):choices[f'm1_s{s}_cg']={'mode':1,'swizzle':s,'cache':1}
    for mode in (2,3,4):
        for rows,warps in ((2,2),(4,4),(8,4)):
            choices[f'm{mode}_s0_r{rows}w{warps}']={'mode':mode,'swizzle':0,'tile':{'rows':rows,'warps':warps}}
    if pilot:choices={n:c for n,c in choices.items() if n in ('control','m1_s1_d0','m2_s0_d0','m3_s0_d0','m3_s128_d0','m4_s0_d0','m4_s128_d0','m3_s0_r2w2','m3_s0_r4w4')}
    return choices


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--layers',type=int,choices=(1,36),default=36);p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'async_weights_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);ring=[load_layer(i) for i in (range(36) if a.layers==36 else (17,))]
    choices=configs(a.pilot);chains={};result={'codebooks':32,'configs':choices,'checks':[],'resources':{},'errors':{},'graph_edges':{},'rows':[],
        'method':'Original G32 weights and exact DP4A arithmetic. Down projection stages immutable packed weights using cp.async or Hopper TMA before PDL wait, then awaits copy completion before use. Actual 36-layer weight ring, numerical/private-stream graph checks and rotating/reversing measurement order. Not TTFA.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        control=Chain()
        for name,options in choices.items():
            try:
                candidate=Chain(options)
                for e in ring:
                    for index in (0,10,31):
                        actual=candidate(e,index);expected=control(e,index)
                        exact=all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(flatten(actual),flatten(expected),strict=True))
                        result['checks'].append({'config':name,'layer':e['layer'],'input':index,'all_bits_exact':exact});assert exact,result['checks'][-1]
                _,kernels=candidate(ring[0],audit=True);k=kernels['down']
                result['resources'][name]={'registers':k.n_regs,'spills':k.n_spills,'shared_bytes':k.metadata.shared,
                    'cp_async_instructions':k.asm['ptx'].count('cp.async'),'tma_instructions':k.asm['ptx'].count('cp.async.bulk.tensor')}
                for _ in range(2):candidate(ring[0])
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):actual=candidate(ring[0])
                result['graph_edges'][name]=graph_edges(graph)
                assert len(result['graph_edges'][name])==2 and all(e['type']==1 for e in result['graph_edges'][name])
                graph.replay();expected=control(ring[0])
                assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(flatten(actual),flatten(expected),strict=True))
                chains[name]=candidate;print('CHECKED',name,result['resources'][name],flush=True)
            except Exception as error:result['errors'][name]=repr(error);print('ERROR',name,repr(error),flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(a.rounds):
        order=list(chains);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        times={n:measure(chains[n],ring) for n in order};result['rows'].append({'round':repeat,'order':order,'us':times})
        save();print('ROUND',repeat,times,flush=True)
    result['summary']={n:{'median_us':statistics.median(r['us'][n] for r in result['rows']),
        'median_paired_gain_us':statistics.median(r['us']['control']-r['us'][n] for r in result['rows'])} for n in chains}
    save();print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
