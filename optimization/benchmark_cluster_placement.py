"""Actual-weight ring screen for unchanged cluster cubins with launch policies."""
import argparse
import json
import statistics
import traceback
from pathlib import Path
import torch
from .common import RESULTS
from .benchmark_attention_pdl import load_layer,library
from .benchmark_qkv_cluster import Chain
from .benchmark_norm_projection import flatten
from .benchmark_projection_pdl import graph_edges
from .qkv_cluster_binary import load_bundle
from .cluster_placement import configured
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--layers',type=int,choices=(1,36),default=36);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'cluster_placement_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);library();ring=[load_layer(i) for i in (range(36) if a.layers==36 else (17,))]
    folder=RESULTS/'qkv_cluster_bundle_v6';options,launchers=load_bundle(folder)
    control=Chain(options['c8_t2_exact'],launcher=launchers['c8_t2_exact'])
    choices={'control':None}
    for policy in (0,1,2):
        for carveout in (None,0,25,50,100):choices[f'p{policy}_c{carveout}']={'policy':policy,'carveout':carveout}
    result={'codebooks':32,'configs':choices,'checks':[],'graphs':{},'occupancy':{},'errors':{},'rows':[],
        'method':'Same selected eight-CTA cubins and native attention/output chain. CUDA launch policy DEFAULT/SPREAD/LOAD_BALANCING and per-launch shared-memory carveout preference only. Fresh modules per variant, fixed arithmetic. Actual 36-layer weights with nearest frozen attention fixtures; not TTFA. Balanced rotated/reversed timing.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    def poison(e):
        d=e['attention'];pos=int(d['position'])
        for n in ('k','v'):d[n][:,:,pos,:].fill_(float('nan'))
    chains={};stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name,choice in choices.items():
            try:
                if choice is None:candidate=control;fn=None
                else:
                    opt,fn=configured(folder,**choice);candidate=Chain(opt,launcher=fn)
                for e in ring:
                    for index in (0,10,31):
                        poison(e);expected=control(e,index,True);cache=[e['attention'][n].clone() for n in ('k','v')]
                        poison(e);actual=candidate(e,index,True)
                        exact=all(torch.equal(x.reshape(-1).view(torch.uint8),y.reshape(-1).view(torch.uint8)) for x,y in zip(flatten(actual),flatten(expected),strict=True))
                        exact_cache=all(torch.equal(e['attention'][n].view(torch.uint8),c.view(torch.uint8)) for n,c in zip(('k','v'),cache,strict=True))
                        row={'config':name,'layer':e['layer'],'input':index,'outputs_exact':exact,'poisoned_cache_exact':exact_cache};result['checks'].append(row)
                        assert exact and exact_cache,row
                for _ in range(2):candidate(ring[0])
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):actual=candidate(ring[0])
                edges=graph_edges(graph);assert len(edges)==3 and all(e['type']==1 for e in edges);result['graphs'][name]=edges
                graph.replay();expected=control(ring[0]);assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
                if fn:result['occupancy'][name]={n:k.occupancy for n,k in fn.kernels.items()}
                chains[name]=candidate;print('CHECKED',name,flush=True)
            except Exception as error:
                result['errors'][name]=traceback.format_exc();print('ERROR',name,repr(error),flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(a.rounds):
        order=list(chains);offset=(repeat//2)%len(order);order=order[offset:]+order[:offset]
        if repeat%2:order.reverse()
        times={n:measure(chains[n],ring) for n in order};result['rows'].append({'round':repeat,'order':order,'us':times});save();print('ROUND',repeat,times,flush=True)
    result['summary']={n:{'median_us':statistics.median(r['us'][n] for r in result['rows']),
        'median_paired_gain_us':statistics.median(r['us']['control']-r['us'][n] for r in result['rows']),
        'faster_rounds':sum(r['us']['control']>r['us'][n] for r in result['rows'])} for n in chains}
    save();print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
