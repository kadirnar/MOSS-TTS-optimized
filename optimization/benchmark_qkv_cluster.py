"""Actual layer-ring screen for clustered QKV/head-preparation fusion."""
import argparse
import hashlib
import json
import statistics
import subprocess
import traceback
from pathlib import Path

import torch
import triton
from .common import RESULTS
from .benchmark_attention_pdl import load_layer,library
from .benchmark_async_attention import Chain as OriginalChain
from .benchmark_projection_pdl import graph_edges
from .benchmark_norm_projection import flatten
from .attention_pdl import launch as attention
from .attention_quant_pdl import reduce_quant
from .async_output import make_dispatch
from .short_scales import PLAN
from .qkv_cluster_prepare import launch
from .tune_weight_reads import measure


class Chain:
    def __init__(self,options=None,launcher=launch):
        self.options=options
        self.launch=launcher
        self.original=OriginalChain({'mode':0,'tile':{'rows':16,'warps':4,'prefetch':3}})
        self.output=make_dispatch(16,register_preload=True)
    def __call__(self,e,index=0,debug=False,audit=False):
        if self.options is None:return self.original(e,index,debug)
        d=e['attention'];raw=e['raw'];w=e['weights']
        result,k=self.launch(raw['x'][index:index+1],raw['residual'][index:index+1] if raw['has_residual'] else None,
            raw['weight'],raw['eps'],*w['qkv'],d['qw'],d['kw'],d['cos'],d['sin'],d['k'],d['v'],d['position'],d['eps'],
            **self.options,debug=debug,return_kernel=True)
        summed,qkv,q=result
        splits=e['capacity']//32;part=torch.empty((32,splits,128),device=q.device,dtype=torch.float32)
        lse=torch.empty((32,splits),device=q.device,dtype=torch.float32)
        attention(q,d['k'],d['v'],d['position'],part,lse,pdl=True,trigger=2)
        hidden,quantized=reduce_quant(part,lse,pdl=True,trigger=1)
        output=self.output(hidden,*w['out'],prequantized=quantized,**PLAN['out'],trigger_mode=3,prefetch=1)
        result=(summed,qkv,q,part,lse,hidden,quantized,output) if debug else output
        return (result,k) if audit else result


def configs(pilot):
    options={'control':None}
    for ctas in ((2,4,8) if pilot else (1,2,4,8)):
        for ir in ((1,) if pilot else (1,2,4)):
            for divisor in (0,16):
                options[f'c{ctas}_ir{ir}_d{divisor}']={'ctas':ctas,'integer_rows':ir,'divisor':divisor}
    if not pilot:
        for ctas in (2,4,8):
            for ig in (2,4):
                for ir in (1,2,4):
                    options[f'c{ctas}_ig{ig}_ir{ir}']={'ctas':ctas,'integer_groups':ig,'integer_rows':ir,'divisor':16}
        for ctas in (2,4,8):
            for trigger in (0,2,3):
                options[f'c{ctas}_t{trigger}']={'ctas':ctas,'divisor':16,'trigger_mode':trigger}
        options['c8_t2_legacy']={'ctas':8,'divisor':16,'trigger_mode':2,'legacy_projection':True}
        options['c8_legacy']={'ctas':8,'divisor':16,'legacy_projection':True}
        options['c8_t2_exact']={'ctas':8,'divisor':16,'trigger_mode':2,'legacy_projection':True,'legacy_norm':True}
        options['c8_exact']={'ctas':8,'divisor':16,'legacy_projection':True,'legacy_norm':True}
        for ctas in (4,8):
            options[f'c{ctas}_t2_leader']={'ctas':ctas,'divisor':16,'trigger_mode':2,'legacy_projection':True,'legacy_norm':True,'leader_only':True}
        for divisor in (8,32):
            options[f'c8_t2_exact_d{divisor}']={'ctas':8,'divisor':divisor,'trigger_mode':2,'legacy_projection':True,'legacy_norm':True}
    return options


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--layers',type=int,choices=(1,36),default=36)
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--pilot',action='store_true')
    p.add_argument('--binary-bundle',type=Path);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and at least two rounds required')
    path=RESULTS/f'qkv_cluster_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);library();ring=[load_layer(i) for i in (range(36) if a.layers==36 else (17,))]
    choices=configs(a.pilot);launchers={}
    if a.binary_bundle:
        from .qkv_cluster_binary import load_bundle
        choices,launchers=load_bundle(a.binary_bundle)
    chains={};result={'codebooks':32,'torch':torch.__version__,'triton':triton.__version__,'binary_bundle':str(a.binary_bundle) if a.binary_bundle else None,'configs':choices,'checks':[],'graph_edges':{},'errors':{},'resources':{},'rows':[],
        'method':'One cluster per 128-channel Q/K/V head: exact G32 norm/projection followed by per-head norm/RoPE/cache writes, then selected native attention/reduction and register-preload output. Four vs five kernels. Actual layer weights/normalization inputs, nearest 0/17/35 frozen attention fixtures; not a full trajectory or TTFA. Changed cache positions poisoned before checks.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    def poison(e):
        d=e['attention'];pos=int(d['position'])
        for name in ('k','v'):d[name].view(8,-1,128)[:,pos,:].fill_(float('nan'))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());control=Chain()
    with torch.cuda.stream(stream):
        for name,options in choices.items():
            candidate=Chain(options,launcher=launchers.get(name,launch))
            try:
                for e in ring:
                    for index in (0,10,31):
                        poison(e);expected=control(e,index,True)
                        cache=[e['attention'][n].clone() for n in ('k','v')]
                        poison(e);actual=candidate(e,index,True)
                        mismatches=[int((x.reshape(-1).view(torch.uint8)!=y.reshape(-1).view(torch.uint8)).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]
                        caches_exact=all(torch.equal(e['attention'][n].view(torch.uint8),saved.view(torch.uint8)) for n,saved in zip(('k','v'),cache,strict=True))
                        check={'config':name,'layer':e['layer'],'input':index,'byte_mismatches':mismatches,'full_poisoned_caches_exact':caches_exact}
                        result['checks'].append(check);save();assert not any(mismatches) and caches_exact,check
                if options is not None:
                    _,k=candidate(ring[0],audit=True)
                    result['resources'][name]={'registers':k.n_regs,'spills':k.n_spills,'shared_bytes':k.metadata.shared,'ctas':options['ctas'],
                        'cubin_sha256':hashlib.sha256(k.asm['cubin']).hexdigest()}
                    folder=RESULTS/f'qkv_cluster_{a.tag}_audit';folder.mkdir(exist_ok=True);prefix=folder/name
                    for suffix in ('ptx','ttgir','cubin'):
                        v=k.asm[suffix];prefix.with_suffix('.'+suffix).write_bytes(v if isinstance(v,bytes) else v.encode())
                    prefix.with_suffix('.sass').write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(prefix.with_suffix('.cubin'))],text=True))
                for _ in range(2):candidate(ring[0])
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):actual=candidate(ring[0])
                edges=graph_edges(graph);result['graph_edges'][name]=edges
                assert len(edges)==(4 if options is None else 3) and all(e['type']==1 for e in edges)
                graph.replay();expected=control(ring[0]);assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
                chains[name]=candidate;print('CHECKED',name,result['resources'].get(name),flush=True)
            except Exception as error:
                trace=traceback.format_exc();result['errors'][name]=trace
                (RESULTS/f'qkv_cluster_{a.tag}_{name}_error.txt').write_text(trace)
                print('ERROR',name,type(error).__name__,str(error).splitlines()[-1],flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(a.rounds):
        order=list(chains);offset=(repeat//2)%len(order);order=order[offset:]+order[:offset]
        if repeat%2:order.reverse()
        times={n:measure(chains[n],ring) for n in order};result['rows'].append({'round':repeat,'order':order,'us':times});save();print('ROUND',repeat,times,flush=True)
    result['summary']={n:{'median_us':statistics.median(r['us'][n] for r in result['rows']),
        'median_paired_gain_us':statistics.median(r['us']['control']-r['us'][n] for r in result['rows'])} for n in chains}
    save();print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
