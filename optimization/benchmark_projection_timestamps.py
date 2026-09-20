"""Measure probe perturbation before interpreting sampled PDL timestamps."""
import argparse
import hashlib
import json
import statistics
import subprocess

import torch

from .common import RESULTS
from .benchmark_projection_pdl import load_layer,graph_edges
from .benchmark_bulk_address import Chain as SelectedChain
from .benchmark_norm_projection import flatten
from .dp4a_norm_pdl import SELECTED
from .short_scales import PLAN
from .projection_timestamps import configured
from .tune_weight_reads import measure


class Chain:
    def __init__(self,mode,stride=16,compact=False,families=None):
        self.mode=mode;self.stride=stride;self.traces={};self.functions={}
        self.families=('up','down','qkv') if families is None else tuple(families)
        self.selected=SelectedChain({'qkv':{'address_mode':1}})
        self.configured=configured
        if compact:
            from .projection_timestamps_compact import configured as compact_configured
            self.configured=compact_configured

    def __call__(self,e,index=0,audit=False):
        if self.mode is None:return self.selected(e,index,audit=audit)
        d=e['raw'];nd=e['next'];weights=e['weights'];kernels={}
        def fn(name):
            if name not in self.families:return self.selected.functions[name]
            key=(e['layer'],name)
            if key not in self.functions:
                rows=(PLAN if name=='down' else SELECTED)[name]['rows']
                n=weights[name][0].shape[0]//(2 if name=='up' else 1)
                self.traces[key]=torch.zeros((n//rows,4),device='cuda',dtype=torch.int64)
                self.functions[key]=self.configured('projection' if name=='down' else 'norm',self.traces[key],mode=self.mode,stride=self.stride)
            return self.functions[key]
        up,k=fn('up')(d['x'][index:index+1],d['residual'][index:index+1] if d['has_residual'] else None,
            d['weight'],d['eps'],*weights['up'],fused=True,**SELECTED['up'],return_kernel=True)
        kernels['up']=k;summed,hidden,q=up
        down,k=fn('down')(hidden,*weights['down'],prequantized=q,**PLAN['down'],trigger_mode=3,prefetch=1,return_kernel=True)
        kernels['down']=k
        following,k=fn('qkv')(down,summed,nd['weight'],nd['eps'],*weights['qkv'],**SELECTED['qkv'],return_kernel=True)
        kernels['qkv']=k
        value=(up,down,following)
        return (value,kernels) if audit else value


def distributions(values):
    x=torch.tensor(values,dtype=torch.float64)
    return {'samples':len(values),'median_us':float(x.median()),'p95_us':float(x.quantile(.95)),
            'max_us':float(x.max()),'min_us':float(x.min())}


def summarize(traces,mode,stride):
    result={}
    for family in ('up','down','qkv'):
        values={'wait':[]}
        if mode==2:values.update(pre_wait=[],after_wait=[],span=[])
        for (layer,stage),t in traces.items():
            if stage!=family:continue
            t=t.cpu()[::stride]
            assert bool((t[:,1]>0).all()) and bool((t[:,2]>=t[:,1]).all())
            values['wait'].extend(((t[:,2]-t[:,1])/1000).tolist())
            if mode==2:
                assert bool((t[:,0]>0).all()) and bool((t[:,3]>=t[:,2]).all()) and bool((t[:,1]>=t[:,0]).all())
                values['pre_wait'].extend(((t[:,1]-t[:,0])/1000).tolist())
                values['after_wait'].extend(((t[:,3]-t[:,2])/1000).tolist())
                values['span'].extend(((t[:,3]-t[:,0])/1000).tolist())
        if values['wait']:result[family]={n:distributions(v) for n,v in values.items()}
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--layers',type=int,choices=(1,36),default=36);p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--compact',action='store_true');p.add_argument('--targeted',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'projection_timestamps_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);ring=[load_layer(i) for i in (range(36) if a.layers==36 else (17,))]
    options={'control':{'mode':None},'clone_off':{'mode':0},'wait_s16':{'mode':1,'stride':16},
        'full_s16':{'mode':2,'stride':16},'full_s64':{'mode':2,'stride':64},
        'wait_s1':{'mode':1,'stride':1},'full_s1':{'mode':2,'stride':1}}
    if a.targeted:
        options={n:options[n] for n in ('control','clone_off')}
        for family in ('up','down','qkv'):
            for mode in (1,2):
                for stride in (1,16):options[f'{family}_m{mode}_s{stride}']={'mode':mode,'stride':stride,'families':[family]}
    if a.compact:options={name:{**value,'compact':True} for name,value in options.items()}
    chains={};result={'codebooks':32,'configs':options,'checks':[],'resources':{},'errors':{},'graph_edges':{},'rows':[],
        'method':'Selected G32 chain, cloned no-probe control and lane-zero globaltimer probes. Sampling does not imply unchanged register occupancy. Six rounds rotate/reverse all-layer rings; instrumentation overhead measured before reporting lane intervals. Final timestamp is store issue by lane zero, not CTA/grid completion. No TTFA claim.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        control=Chain(None)
        for name,config in options.items():
            candidate=Chain(**config)
            try:
                for e in ring:
                    for index in (0,10,31):
                        actual=candidate(e,index);expected=control(e,index)
                        exact=all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(flatten(actual),flatten(expected),strict=True))
                        result['checks'].append({'config':name,'layer':e['layer'],'input':index,'all_bits_exact':exact});assert exact,result['checks'][-1]
                _,compiled=candidate(ring[0],audit=True)
                result['resources'][name]={n:{'registers':k.n_regs,'spills':k.n_spills,'shared_bytes':k.metadata.shared,
                    'globaltimer_reads_in_ptx':k.asm['ptx'].count('%globaltimer')} for n,k in compiled.items()}
                for _ in range(2):candidate(ring[0])
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):actual=candidate(ring[0])
                result['graph_edges'][name]=graph_edges(graph)
                assert len(result['graph_edges'][name])==2 and all(x['type']==1 for x in result['graph_edges'][name])
                graph.replay();expected=control(ring[0])
                assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(flatten(actual),flatten(expected),strict=True))
                chains[name]=candidate;print('CHECKED',name,flush=True)
            except Exception as error:result['errors'][name]=repr(error);print('ERROR',name,repr(error),flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(a.rounds):
        order=list(chains);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        times={n:measure(chains[n],ring) for n in order};result['rows'].append({'round':repeat,'order':order,'us':times})
        save();print('ROUND',repeat,times,flush=True)
    result['timing']={n:{'median_us':statistics.median(r['us'][n] for r in result['rows']),
        'median_paired_overhead_us':statistics.median(r['us'][n]-r['us']['control'] for r in result['rows'])} for n in chains}
    result['lane_intervals']={n:summarize(c.traces,c.mode,c.stride) for n,c in chains.items() if c.mode}
    torch.save({n:{f'{layer:02d}_{stage}':v.cpu() for (layer,stage),v in c.traces.items()} for n,c in chains.items() if c.mode},RESULTS/f'projection_timestamps_{a.tag}_raw.pt')
    audit=RESULTS/f'projection_timestamps_{a.tag}_audit';audit.mkdir(exist_ok=False)
    for name in ('control','clone_off','full_s16','up_m1_s16','up_m2_s16','down_m1_s16','down_m2_s16','qkv_m1_s16','qkv_m2_s16'):
        if name not in chains:continue
        _,compiled=chains[name](ring[0],audit=True)
        for stage,k in compiled.items():
            prefix=audit/f'{name}_{stage}'
            for suffix in ('ptx','ttgir','cubin'):
                data=k.asm[suffix];prefix.with_suffix('.'+suffix).write_bytes(data if isinstance(data,bytes) else data.encode())
            prefix.with_suffix('.sass').write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(prefix.with_suffix('.cubin'))],text=True))
            result['resources'][name][stage]['cubin_sha256']=hashlib.sha256(k.asm['cubin']).hexdigest()
    save();print('SUMMARY',result['timing'],flush=True)


if __name__=='__main__':main()
