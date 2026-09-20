"""Actual attention-output weight ring for asynchronous G32 staging."""
import argparse
import json
import statistics

import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .benchmark_group128 import quantize
from .dp4a_layout_pdl_prefetch import linear as control
from .dp4a_async_weights import configured
from .short_scales import PLAN
from .tune_weight_reads import measure


def configs():
    choices={'control':None}
    for mode,swizzle in ((1,1),(1,8),(2,0),(2,128)):
        for rows,warps in ((2,2),(4,4),(8,4),(16,4)):
            choices[f'm{mode}_s{swizzle}_r{rows}w{warps}']={'mode':mode,'swizzle':swizzle,'rows':rows,'warps':warps}
    for rows in (4,8):choices[f'm1_cg_r{rows}w4']={'mode':1,'swizzle':1,'cache':1,'rows':rows,'warps':4}
    return choices


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=6);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'async_output_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);ring=[]
    for i in range(36):
        saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{i:02d}_out.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16()
        states=torch.load(RESULTS/f'calibration_v1/{i:02d}_out.pt',weights_only=True)
        xs=[states[j:j+1].cuda() for j in (0,231,528,1186)]
        zero=torch.zeros_like(xs[0]);spike=zero.clone();spike[0,-1]=3;xs.extend((zero,spike))
        ring.append({'layer':i,'weights':(w,s),'inputs':[(x,quantize(x,32)) for x in xs]})
    choices=configs();functions={};result={'codebooks':32,'configs':choices,'checks':[],'resources':{},'errors':{},'rows':[],
        'method':'Actual 36-layer attention-output weight ring, four frozen calibration inputs plus zero/spike. Selected G32 values and arithmetic. Standalone operator timings, without a preceding attention producer; no full-chain or TTFA claim.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    def invoke(fn,tile,e,index=1,audit=False):
        x,q=e['inputs'][index]
        return fn(x,*e['weights'],prequantized=q,**{**PLAN['out'],**tile},trigger_mode=3,prefetch=1,return_kernel=audit)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name,options in choices.items():
            try:
                tile={} if options is None else {k:options[k] for k in ('rows','warps')}
                fn=control if options is None else configured(**{k:v for k,v in options.items() if k not in tile})
                for e in ring:
                    for index in range(6):
                        actual=invoke(fn,tile,e,index);expected=invoke(control,{},e,index)
                        exact=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
                        result['checks'].append({'config':name,'layer':e['layer'],'input':index,'all_bits_exact':exact});assert exact,result['checks'][-1]
                _,k=invoke(fn,tile,ring[0],audit=True)
                result['resources'][name]={'registers':k.n_regs,'spills':k.n_spills,'shared_bytes':k.metadata.shared}
                for _ in range(2):invoke(fn,tile,ring[0])
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):actual=invoke(fn,tile,ring[0])
                graph.replay();expected=invoke(control,{},ring[0]);assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
                functions[name]=(fn,tile);print('CHECKED',name,result['resources'][name],flush=True)
            except Exception as error:result['errors'][name]=repr(error);print('ERROR',name,repr(error),flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(a.rounds):
        order=list(functions);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        times={n:measure(lambda e:invoke(*functions[n],e),ring) for n in order}
        result['rows'].append({'round':repeat,'order':order,'us':times});save();print('ROUND',repeat,flush=True)
    result['summary']={n:{'median_us':statistics.median(r['us'][n] for r in result['rows']),
        'median_paired_gain_us':statistics.median(r['us']['control']-r['us'][n] for r in result['rows'])} for n in functions}
    save();print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
