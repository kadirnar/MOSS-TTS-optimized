"""Revisit separate normalization now that dependent launches can overlap it."""
import argparse
import json
import statistics

import torch
from .common import RESULTS
from .benchmark_projection_pdl import load_layer,graph_edges
from .benchmark_norm_projection import flatten
from .dp4a_norm_pdl import linear as fused_norm
from .dp4a_norm_projection import SELECTED
from .dp4a_norm_quant_pdl import norm_quant
from .dp4a_scaled_pdl import linear as split_projection
from .dp4a_layout_pdl_prefetch import linear as project
from .short_scales import PLAN
from .tune_weight_reads import measure


def producer(raw,weights,*,index=0,fused=False,debug=False,config=None):
    x=raw['x'][index:index+1];res=raw['residual'][index:index+1] if raw['has_residual'] else None
    if config is None:
        return fused_norm(x,res,raw['weight'],raw['eps'],*weights,
                          fused=fused,debug=debug,**SELECTED['up' if fused else 'qkv'],trigger_mode=1)
    summed,normalized,quantized=norm_quant(x,res,raw['weight'],raw['eps'],trigger=config['norm_trigger'])
    output=split_projection(normalized,*weights,paired=fused,fused=fused,prequantized=quantized,
                            scale_mode=4 if fused else 0,mode=2,rows=config['rows'],warps=config['warps'],trigger=config['trigger'])
    value=(summed,*output) if fused else (summed,output)
    return (value,(normalized,*quantized)) if debug else value


def chain(entry,config=None,*,index=0,debug=False):
    up_config=None if config is None else config.get('up')
    qkv_config=None if config is None else config.get('qkv')
    raw=entry['raw'];nxt=entry['next'];weights=entry['weights']
    result=producer(raw,weights['up'],index=index,fused=True,debug=debug,config=up_config)
    summed,hidden,quantized=result[0] if debug else result
    output=project(hidden,*weights['down'],prequantized=quantized,**PLAN['down'],trigger_mode=3,prefetch=1)
    following_raw={'x':output,'residual':summed,'has_residual':True,'weight':nxt['weight'],'eps':nxt['eps']}
    following=producer(following_raw,weights['qkv'],debug=debug,config=qkv_config)
    return result,output,following


def configurations():
    configs={'control':None}
    for rows,warps in ((32,4),(32,8),(64,4),(64,8)):
        for trigger in (1,2,3):
            configs[f'up_r{rows}_w{warps}_p{trigger}']={'up':{'rows':rows,'warps':warps,'norm_trigger':1,'trigger':trigger}}
    for nt in (0,2,3):
        configs[f'up_n{nt}']={'up':{'rows':32,'warps':4,'norm_trigger':nt,'trigger':1}}
    for rows in (4,8):
        for trigger in (1,3):
            configs[f'qkv_r{rows}_p{trigger}']={'qkv':{'rows':rows,'warps':4,'norm_trigger':1,'trigger':trigger}}
    return configs


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--layers',type=int,choices=(1,36),default=36)
    p.add_argument('--rounds',type=int,default=6);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'split_projection_pdl_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);configs=configurations();ring=[load_layer(i) for i in (range(36) if a.layers==36 else (17,))]
    checks=[];edges={};rows=[];graph_checks={}
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for entry in ring:
            for index in (0,10,31):
                expected=chain(entry,index=index,debug=True)
                for name,config in configs.items():
                    if config is None:continue
                    actual=chain(entry,config,index=index,debug=True)
                    counts=[int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]
                    checks.append({'layer':entry['layer'],'input':index,'config':name,'mismatches':counts})
            print('CHECKED',entry['layer'],flush=True)
        for name,config in configs.items():
            entry=ring[0]
            for _ in range(2):chain(entry,config)
            graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph,stream=stream):actual=chain(entry,config)
            graph.replay();expected=chain(entry)
            graph_checks[name]=[int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]
            edges[name]=graph_edges(graph)
            count=2+(0 if config is None else len(config))
            assert len(edges[name])==count and all(e['type']==1 for e in edges[name]),(name,edges[name])
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'torch':torch.__version__,'configs':configs,'layers':a.layers,'checks':checks,'graph_checks':graph_checks,'graph_edges':edges,'rows':rows,
            'method':'Actual 36-layer MLP -> down -> following norm/QKV ring. Control fuses normalization into each projection with qualified PDL and short scales. '
                     'Candidates split either normalization producer and add a PDL projection consumer, varying tile size, warp count and hint placement. '
                     'Three frozen inputs per layer; intermediate, output, private stream and graph checks. Six rotating/reversed timing rounds; no profiler initialized before timing. '
                     'Isolated chains, not TTFA or a full model trajectory.'}
    path.write_text(json.dumps(result,indent=2)+'\n')
    for repeat in range(a.rounds):
        order=list(configs);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        values={name:measure(lambda e:chain(e,configs[name]),ring) for name in order}
        rows.append({'round':repeat,'order':order,'us':values});print('ROUND',repeat,values,flush=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
    result['summary']={name:{'median_us':statistics.median(r['us'][name] for r in rows),'mismatches':sum(sum(r['mismatches']) for r in checks if r['config']==name)} for name in configs}
    path.write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
