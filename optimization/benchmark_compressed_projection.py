"""Identical kernels with ordinary and losslessly compressible actual weights."""
import argparse
import gc
import json
import statistics

import torch

from .common import RESULTS
from .compressed_alloc import clone,library
from .dp4a_packing import pack_interleaved
from .dp4a_norm_projection import linear as norm_linear,SELECTED as NORM
from .dp4a_scaled import linear as scaled_linear,SELECTED as SCALED
from .benchmark_group128 import quantize
from .benchmark_norm_projection import flatten
from .tune_weight_reads import measure


@torch.inference_mode()
def family(name,rounds):
    ring=[];allocations=[];checks=[]
    for layer in range(36):
        saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(saved['packed']);s=saved['scales'].to(torch.bfloat16 if name in NORM else torch.float32)
        variants={'torch':(w,s)};buffers={}
        for compressed in (False,True):
            pair=[]
            for label,t in (('weights',w),('scales',s)):
                value,meta=clone(t,compressed=compressed)
                assert torch.equal(value.view(torch.uint8),t.view(torch.uint8))
                pair.append(value);allocations.append({'layer':layer,'buffer':label,'requested_compression':compressed,**meta})
            buffers[compressed]=tuple(pair)
        variants.update({'vmm_plain':buffers[False],'compressed_both':buffers[True],
                         'compressed_weights':(buffers[True][0],s),'compressed_scales':(w,buffers[True][1])})
        if name in NORM:
            raw=torch.load(RESULTS/f'norm_projection_capture_v1/{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
            data=(raw['x'][:1],raw['residual'][:1] if raw['has_residual'] else None,raw['weight'],raw['eps'])
        else:
            x=torch.load(RESULTS/f'calibration_v1/{layer:02d}_{name}.pt',weights_only=True)[231:232].cuda()
            data=(x,quantize(x,32))
        ring.append((data,variants))
    def call(entry,variant):
        data,variants=entry;w,s=variants[variant]
        if name in NORM:return norm_linear(*data,w,s,fused=name=='up',**NORM[name])
        x,qx=data
        return scaled_linear(x,w,s,prequantized=qx,**SCALED[name])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for layer,entry in enumerate(ring):
            expected=call(entry,'torch')
            for variant in variants:
                if variant=='torch':continue
                actual=call(entry,variant)
                mismatch=[int((a!=b).sum()) for a,b in zip(flatten(actual),flatten(expected),strict=True)]
                checks.append({'layer':layer,'variant':variant,'mismatches':mismatch})
                assert not any(mismatch),(name,layer,variant,mismatch)
    torch.cuda.current_stream().wait_stream(stream)
    rows=[]
    for repeat in range(rounds):
        order=list(variants);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        timing={variant:measure(lambda e:call(e,variant),ring) for variant in order}
        rows.append({'round':repeat,'order':order,'us':timing});print(name,rows[-1],flush=True)
    return {'rounds':rows,'checks':checks,'allocations':allocations,'all_exact':True,
            'median_us':{v:statistics.median(r['us'][v] for r in rows) for v in variants},
            'median_paired_gain_us':{v:statistics.median(r['us']['torch']-r['us'][v] for r in rows) for v in variants if v!='torch'}}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=8);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and at least two rounds required')
    path=RESULTS/f'compressed_projection_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);result={'codebooks':32,'torch':torch.__version__,'families':{},
        'method':'Thirty-six actual layer weights/scales and distinct layer inputs/norm buffers. Identical selected kernels, five allocation variants, rotating/reversed timing order, private-stream numerical comparisons. Copy time and allocation setup excluded.'}
    for name in ('qkv','up','out','down'):
        result['families'][name]=family(name,a.rounds)
        gc.collect();result['allocator_after_family']=library().counters();assert result['allocator_after_family']['live_allocations']==0
        path.write_text(json.dumps(result,indent=2)+'\n')
        print('DONE',name,result['families'][name]['median_us'],result['allocator_after_family'],flush=True)
    print('ALL PASS',flush=True)


if __name__=='__main__':main()
