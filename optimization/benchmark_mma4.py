"""Native INT4 MMA screening on every actual G32 projection layer."""
import argparse
import gc
import json
import statistics

import torch

from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_scaled import linear as reference,SELECTED
from .benchmark_group128 import quantize
from .benchmark_norm_projection import flatten
from .mma4_projection import linear,pack_weight,unpack_weight,library
from .tune_weight_reads import measure


@torch.inference_mode()
def family(name,rounds):
    configs=[{'m':m,'warps':w,'unroll':u} for m in (8,16) for w in (4,8) for u in (1,4)]
    ring=[];checks=[]
    for layer in range(36):
        saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
        raw=saved['packed'];s=saved['scales'].to(torch.bfloat16 if name in ('qkv','up') else torch.float32)
        weights={'reference':pack_interleaved(raw)}
        for m in (8,16):
            weights[m]=pack_weight(raw,m)
            assert torch.equal(unpack_weight(weights[m],raw.shape[0],raw.shape[1]*2,m),raw)
        states=torch.load(RESULTS/f'calibration_v1/{layer:02d}_{name}.pt',weights_only=True)
        x=states[231:232].cuda();ring.append((x,quantize(x,32),weights,s))
        zero=torch.zeros_like(x);spike=zero.clone();spike[0,-1]=100
        inputs=[(str(i),states[i:i+1].cuda()) for i in (0,231,528,1186)]+[('zero',zero),('spike',spike)]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for label,x in inputs:
                qx=quantize(x,32)
                expected=reference(x,weights['reference'],s,**SELECTED[name],paired=name=='up',fused=name=='up',scale_mode=4 if name=='up' else 0,prequantized=qx)
                for config in configs:
                    actual=linear(x,weights[config['m']],s,prequantized=qx,fused=name=='up',**config)
                    mismatch=[int((a!=b).sum()) for a,b in zip(flatten(actual),flatten(expected),strict=True)]
                    checks.append({'layer':layer,'input':label,'config':config,'mismatches':mismatch})
        torch.cuda.current_stream().wait_stream(stream)
        print('CHECK',name,layer,'mismatches',sum(sum(r['mismatches']) for r in checks if r['layer']==layer),flush=True)
    def call(entry,config):
        x,qx,w,s=entry
        if config is None:return reference(x,w['reference'],s,**SELECTED[name],paired=name=='up',fused=name=='up',scale_mode=4 if name=='up' else 0,prequantized=qx)
        return linear(x,w[config['m']],s,prequantized=qx,fused=name=='up',**config)
    options={'reference':None,**{f'm{c["m"]}_w{c["warps"]}_u{c["unroll"]}':c for c in configs}}
    rows=[]
    for repeat in range(rounds):
        order=list(options);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        timing={key:measure(lambda entry:call(entry,options[key]),ring) for key in order}
        rows.append({'round':repeat,'order':order,'us':timing});print('TIMING',name,rows[-1],flush=True)
    median={key:statistics.median(r['us'][key] for r in rows) for key in options}
    exact={key:all(not any(c['mismatches']) for c in checks if c['config']==config) for key,config in options.items() if config is not None}
    valid=[key for key in exact if exact[key]]
    return {'checks':checks,'all_exact':all(exact.values()),'exact_by_config':exact,'rows':rows,'median_us':median,
            'best_exact':min(valid,key=lambda key:median[key]) if valid else None,'packing_roundtrip_all_36_layers':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=4);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and at least two rounds required')
    path=RESULTS/f'mma4_projection_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);library()
    result={'codebooks':32,'torch':torch.__version__,'families':{},
            'method':'Actual 36-layer weight/scale/input ring, prequantized G32 activations, unchanged selected scaled-DP4A reference. M8/M16 INT4 MMA, four/eight warps, loop unroll one/four. Four recorded inputs plus zero/spike per layer; private-stream checks. Rotating/reversed timing order. Gate/up includes SiLU and next activation quantization. Producer normalization excluded from both sides.'}
    for name in ('qkv','out','up','down'):
        result['families'][name]=family(name,a.rounds);torch.cuda.synchronize();gc.collect()
        path.write_text(json.dumps(result,indent=2)+'\n')
        v=result['families'][name];print('DONE',name,v['median_us'],'BEST',v['best_exact'],'EXACT',v['all_exact'],flush=True)


if __name__=='__main__':main()
