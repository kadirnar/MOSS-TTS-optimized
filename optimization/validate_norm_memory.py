"""Repeated timings, all-layer arithmetic and padded graph checks for QKV loads."""
import argparse
import json
import statistics

import torch

from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_norm_projection import linear as reference,SELECTED
from .dp4a_norm_memory import linear
from .benchmark_norm_projection import flatten
from .tune_weight_reads import measure


CANDIDATE={'eviction':'evict_first','scale_cache':'.cg'}


def compare(actual,expected):
    return [int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('repeat','layer_ring','all','memory'))
    p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'norm_memory_{a.mode}_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve previous results')
    torch.set_num_threads(4);torch.manual_seed(712);rows=[]
    base=SELECTED['qkv'];folder=RESULTS/'norm_projection_capture_v1'
    if a.mode in ('repeat','layer_ring'):
        d=torch.load(folder/'00_qkv.pt',map_location='cuda',weights_only=True)
        saved=torch.load(RESULTS/'gptq_v1_g32_d10/00_qkv.pt',map_location='cuda',weights_only=True)
        w,s=pack_interleaved(saved['packed']),saved['scales'].bfloat16()
        ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
        if a.mode=='layer_ring':
            ring=[]
            for layer in range(36):
                raw=torch.load(folder/f'{layer:02d}_qkv.pt',map_location='cuda',weights_only=True)
                saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{layer:02d}_qkv.pt',map_location='cuda',weights_only=True)
                ring.append((raw,pack_interleaved(saved['packed']),saved['scales'].bfloat16()))
        def call(pair,config):
            fn=reference if config is None else linear
            raw=d
            if a.mode=='layer_ring':raw,*pair=pair
            residual=raw['residual'][:1] if raw['has_residual'] else None
            return fn(raw['x'][:1],residual,raw['weight'],raw['eps'],*pair,**base,**({} if config is None else config))
        configs={'control':None,'evict_first':{'eviction':'evict_first'},'evict_first_scale_cg':CANDIDATE}
        for i in range(12):
            order=list(configs);order=order[i%3:]+order[:i%3]
            if i%2:order.reverse()
            timing={name:measure(lambda pair:call(pair,configs[name]),ring) for name in order}
            rows.append({'round':i,'order':order,'us':timing});print(rows[-1],flush=True)
        result={'rows':rows,'median_us':{name:statistics.median(r['us'][name] for r in rows) for name in configs},
                'median_paired_gain_us':{name:statistics.median(r['us']['control']-r['us'][name] for r in rows) for name in configs if name!='control'}}
    elif a.mode=='all':
        for layer in range(36):
            d=torch.load(folder/f'{layer:02d}_qkv.pt',map_location='cuda',weights_only=True)
            saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{layer:02d}_qkv.pt',map_location='cuda',weights_only=True)
            w,s=pack_interleaved(saved['packed']),saved['scales'].bfloat16()
            inputs=[(str(i),d['x'][i:i+1],d['residual'][i:i+1] if d['has_residual'] else None) for i in range(d['x'].shape[0])]
            zero=torch.zeros(1,4096,device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
            inputs.extend([('zero',zero,zero if d['has_residual'] else None),('spike',spike,zero if d['has_residual'] else None)])
            for label,x,residual in inputs:
                expected=reference(x,residual,d['weight'],d['eps'],w,s,debug=True,**base)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):actual=linear(x,residual,d['weight'],d['eps'],w,s,debug=True,**base,**CANDIDATE)
                torch.cuda.current_stream().wait_stream(stream)
                counts=compare(actual,expected);rows.append({'layer':layer,'input':label,'mismatches':counts})
            print('LAYER',layer,'mismatches',sum(sum(r['mismatches']) for r in rows if r['layer']==layer),flush=True)
        result={'cases':rows,'all_exact':all(not any(r['mismatches']) for r in rows),'private_stream':True}
        assert result['all_exact']
    else:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for n in (1,5,37,6143,6144,6145):
                w=pack_interleaved(torch.randint(0,256,(n,2048),device='cuda',dtype=torch.uint8))
                s=(torch.rand(n,128,device='cuda')*.01).bfloat16()
                nw=torch.randn(4096,device='cuda',dtype=torch.bfloat16)
                x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16)
                for add in (False,True):
                    residual=torch.randn_like(x) if add else None
                    for _ in range(3):linear(x,residual,nw,1e-6,w,s,**base,**CANDIDATE)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):actual=linear(x,residual,nw,1e-6,w,s,**base,**CANDIDATE)
                    graph.replay();expected=reference(x,residual,nw,1e-6,w,s,**base)
                    counts=compare(actual,expected)
                    rows.append({'n':n,'add':add,'mismatches':counts});assert not any(counts)
        torch.cuda.current_stream().wait_stream(stream)
        result={'cases':rows,'all_exact':True,'private_stream':True,'cuda_graph':True}
    result.update({'mode':a.mode,'config':CANDIDATE,'codebooks':32})
    path.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('rows','cases')},flush=True)


if __name__=='__main__':main()
