"""Real-layer PDL chains with gate/up branches or output rows in parallel warps."""
import argparse
import json
import statistics
import traceback

import torch

from .common import RESULTS
from .benchmark_ptx_resources import Chain
from .benchmark_projection_pdl import load_layer,graph_edges
from .benchmark_norm_projection import flatten
from .dp4a_gateup_warp_specialized import linear
from .tune_weight_reads import measure


class Candidate(Chain):
    def __init__(self,config):super().__init__([],None);self.options=config
    def up(self,entry,index):
        if self.options is None:return super().up(entry,index)
        d=entry['raw']
        return linear(d['x'][index:index+1],d['residual'][index:index+1],d['weight'],d['eps'],*entry['weights']['up'],**self.options)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--layers',type=int,choices=(1,36),default=36)
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'gateup_ws_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);ring=[load_layer(i) for i in (range(36) if a.layers==36 else [17])]
    configs={'control':None}
    for mode in (0,1):
        for cap,worker in ((80,80),(96,96),(112,96),(112,112),(128,96),(128,128),(160,128),(192,160)):
            configs[f'm{mode}_r{cap}_w{worker}']={'mode':mode,'max_registers':cap,'worker_registers':worker}
    if a.pilot:configs={k:v for k,v in configs.items() if k in ('control','m0_r112_w96','m1_r112_w96')}
    result={'codebooks':32,'layers':a.layers,'configs':configs,'checks':[],'resources':{},'errors':{},'graph_edges':{},'rows':[],
            'method':'Four-warp normalization/reduction order is retained separately in both partitions. Either gate/up branches or half the output rows execute concurrently in two four-warp partitions, joined through shared memory and an mbarrier. Complete MLP/down/next-QKV chains; real saved inputs, all-layer rotating weights, private-stream graph checks. Microbenchmark only, not TTFA.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    chains={};control=Candidate(None);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name,config in configs.items():
            try:
                candidate=Candidate(config)
                if config is not None:
                    e=ring[0];d=e['raw'];_,kernel=linear(d['x'][:1],d['residual'][:1],d['weight'],d['eps'],*e['weights']['up'],**config,return_kernel=True)
                    result['resources'][name]={'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared,'num_warps':kernel.metadata.num_warps}
                    for ext in ('ptx','ttgir'):(RESULTS/f'gateup_ws_{a.tag}_{name}.{ext}').write_text(kernel.asm[ext])
                for entry in ring:
                    for index in (0,10,31):
                        expected=control(entry,index);actual=candidate(entry,index)
                        counts=[int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]
                        result['checks'].append({'config':name,'layer':entry['layer'],'input':index,'mismatches':counts})
                for _ in range(2):candidate(ring[0])
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):captured=candidate(ring[0])
                edges=graph_edges(graph);assert len(edges)==2 and all(e['type']==1 for e in edges);result['graph_edges'][name]=edges
                graph.replay();stream.synchronize();expected=control(ring[0])
                counts=[int((x!=y).sum()) for x,y in zip(flatten(captured),flatten(expected),strict=True)]
                result['checks'].append({'config':name,'layer':ring[0]['layer'],'input':'graph','mismatches':counts})
                chains[name]=candidate;print('CHECKED',name,'mismatches',sum(sum(r['mismatches']) for r in result['checks'] if r['config']==name),flush=True)
            except Exception as error:
                result['errors'][name]=traceback.format_exc();print('ERROR',name,repr(error),flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(a.rounds):
        order=list(chains);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        timings={name:measure(chains[name],ring) for name in order};result['rows'].append({'round':repeat,'order':order,'us':timings})
        save();print('ROUND',repeat,timings,flush=True)
    result['summary']={name:{'median_us':statistics.median(r['us'][name] for r in result['rows']),
                            'mismatches':sum(sum(r['mismatches']) for r in result['checks'] if r['config']==name),
                            'paired_gain_us':[r['us']['control']-r['us'][name] for r in result['rows']]}
                       for name in chains}
    save();print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
