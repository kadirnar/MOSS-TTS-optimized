"""Check async output staging with its actual attention producer schedule."""
import argparse
import json
import statistics
import types

import torch
from .common import RESULTS
from .benchmark_attention_pdl import load_layer,chain as original,library
from .benchmark_bulk_address import Chain as SelectedChain
from .benchmark_projection_pdl import graph_edges
from .benchmark_norm_projection import flatten
from .dp4a_async_weights import configured
from .tune_weight_reads import measure


class Chain:
    def __init__(self,options=None):
        namespace=dict(original.__globals__)
        namespace['norm']=SelectedChain({'qkv':{'address_mode':1}}).functions['qkv']
        if options is not None:
            options=dict(options);tile=options.pop('tile')
            project=namespace['project'] if options.get('mode')==0 else configured(**options)
            namespace['project']=lambda *args,**kwargs:project(*args,**{**kwargs,**tile})
        self.chain=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
        self.chain.__kwdefaults__=dict(original.__kwdefaults__)
    def __call__(self,e,index=0,debug=False):
        return self.chain(e,{'pdl':True,'qk':1,'attention':2,'reduce':1},index=index,debug=debug)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--ablation',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'async_attention_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);library();ring=[load_layer(i) for i in range(36)]
    options={'control':None}
    for mode,swizzle in ((1,1),(1,8),(2,0)):
        for rows in (8,16):options[f'm{mode}_s{swizzle}_r{rows}w4']={'mode':mode,'swizzle':swizzle,'tile':{'rows':rows,'warps':4}}
    if a.ablation:
        options={n:v for n,v in options.items() if n in ('control','m1_s8_r8w4','m1_s8_r16w4')}
        for rows in (8,16):
            for prefetch in (1,3):
                options[f'plain_r{rows}w4_pre{prefetch}']={'mode':0,'tile':{'rows':rows,'warps':4,'prefetch':prefetch}}
    chains={n:Chain(c) for n,c in options.items()};result={'codebooks':32,'configs':options,'checks':[],'graph_edges':{},'rows':[],
        'method':'Selected QKV bulk hint and QK/attention/reduction PDL schedules with async output weights. Actual 36-layer weights/norm inputs and nearest frozen attention fixtures (0/17/35), capacity256; isolated five-kernel chains, not a full trajectory or TTFA. Rotating/reversing six-round actual-weight ring.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for n,c in chains.items():
            for e in ring:
                for index in (0,10,31):
                    actual=c(e,index,True);expected=chains['control'](e,index,True)
                    exact=all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(flatten(actual),flatten(expected),strict=True))
                    result['checks'].append({'config':n,'layer':e['layer'],'input':index,'all_bits_exact':exact});assert exact,result['checks'][-1]
            for _ in range(2):c(ring[0])
            graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph,stream=stream):actual=c(ring[0])
            result['graph_edges'][n]=graph_edges(graph)
            assert len(result['graph_edges'][n])==4 and all(e['type']==1 for e in result['graph_edges'][n])
            graph.replay();expected=chains['control'](ring[0]);assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
            save();print('CHECKED',n,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(a.rounds):
        order=list(chains);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        times={n:measure(chains[n],ring) for n in order};result['rows'].append({'round':repeat,'order':order,'us':times});save();print('ROUND',repeat,times,flush=True)
    result['summary']={n:{'median_us':statistics.median(r['us'][n] for r in result['rows']),
        'median_paired_gain_us':statistics.median(r['us']['control']-r['us'][n] for r in result['rows'])} for n in chains}
    save();print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
