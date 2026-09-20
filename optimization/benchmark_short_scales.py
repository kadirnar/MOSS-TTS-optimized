"""Revisit exact BF16 storage under the current projection compiler/layouts."""
import argparse
import gc
import json
import statistics
import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_scaled import linear as reference,SELECTED
from .dp4a_layout import linear as explicit
from .benchmark_group128 import quantize
from .compressed_alloc import clone
from .tune_weight_reads import measure


@torch.inference_mode()
def family(name,rounds):
    options={'fp32':{'backend':'reference'},'compressed_fp32':{'backend':'compressed'}}
    for mode in (0,1,2,4,8,16):options[f'triton_s{mode}']={'backend':'triton','scale_mode':mode}
    for r in (4,8):
        for nw in ((4,) if name=='out' else (2,4)):
            for ig in (1,2):
                for ir in (1,nw):
                    options[f'gluon_r{r}w{nw}ig{ig}ir{ir}']={'backend':'gluon','rows':r,'warps':nw,'integer_groups':ig,'integer_rows':ir}
    ring=[];checks=[]
    def call(entry,config):
        x,q,w,s,short,compressed=entry
        backend=config['backend']
        if backend in ('reference','compressed'):return reference(x,w,s if backend=='reference' else compressed,**SELECTED[name],prequantized=q)
        if backend=='triton':return reference(x,w,short,**SELECTED[name],scale_mode=config['scale_mode'],prequantized=q)
        return explicit(x,w,short,prequantized=q,**{k:v for k,v in config.items() if k!='backend'})
    for layer in range(36):
        saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(saved['packed']);short=saved['scales'].bfloat16();s=saved['scales'].float()
        assert torch.equal(short.float(),s)
        compressed,_=clone(s)
        states=torch.load(RESULTS/f'calibration_v1/{layer:02d}_{name}.pt',weights_only=True)
        x=states[231:232].cuda();entry=(x,quantize(x,32),w,s,short,compressed);ring.append(entry)
        if layer not in (0,17,35):continue
        zero=torch.zeros_like(x);spike=zero.clone();spike[0,-1]=100
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for label,x in [(str(i),states[i:i+1].cuda()) for i in (0,231,528,1186)]+[('zero',zero),('spike',spike)]:
                current=(x,quantize(x,32),w,s,short,compressed);expected=call(current,options['fp32'])
                for key,config in options.items():
                    if key=='fp32':continue
                    actual=call(current,config);count=int((actual!=expected).sum())
                    checks.append({'layer':layer,'input':label,'variant':key,'mismatches':count})
        torch.cuda.current_stream().wait_stream(stream)
    rows=[]
    for repeat in range(rounds):
        order=list(options);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        timing={key:measure(lambda e:call(e,options[key]),ring) for key in order}
        rows.append({'round':repeat,'order':order,'us':timing});print(name,rows[-1],flush=True)
    summary={key:{'config':config,'median_us':statistics.median(r['us'][key] for r in rows),'mismatches':sum(c['mismatches'] for c in checks if c['variant']==key),
                  'checks':sum(c['variant']==key for c in checks)} for key,config in options.items()}
    return {'rows':rows,'checks':checks,'summary':summary,'all_scale_values_exact':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=4);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and at least two rounds required')
    path=RESULTS/f'short_scales_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);result={'codebooks':32,'torch':torch.__version__,'families':{},
        'method':'36 distinct actual weight/scale/input layers; current selected FP32 scaled-DP4A reference, lossless compressed FP32 control, BF16 scale storage with Triton contiguity hints or explicit Gluon layouts. Three layers x four real vectors plus zero/spike, private-stream exactness checks. Rotating/reversed timing order; all scale values unchanged.'}
    for name in ('out','down'):
        result['families'][name]=family(name,a.rounds);torch.cuda.synchronize();gc.collect()
        path.write_text(json.dumps(result,indent=2)+'\n');print('DONE',name,result['families'][name]['summary'],flush=True)


if __name__=='__main__':main()
