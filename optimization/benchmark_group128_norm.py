"""Fuse normalization into actual calibrated projections; retain failed trials."""
import argparse
import json
import re
import torch
from .common import RESULTS
from .compiler_audit import cuda
from .dp4a_packing import pack_interleaved
from .dp4a_fusions import norm_quant
from .dp4a_group128 import linear as reference
from .dp4a_group128_norm import linear
from .tune_weight_reads import measure
from .benchmark_group128 import quantize


def flatten(x):
    if isinstance(x,tuple):return tuple(t for child in x for t in flatten(child))
    return (x,)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--activation-group',type=int,choices=(32,128),required=True);p.add_argument('--output-quant',action='store_true');args=p.parse_args()
    ag=args.activation_group
    plan=json.loads((RESULTS/f'group128_a{ag}_plan_v1.json').read_text())
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    outfile=RESULTS/f'group128_norm_a{ag}_{args.tag}.json'
    if outfile.exists():raise FileExistsError('Preserve previous results')
    torch.set_num_threads(4)
    folder=RESULTS/'compiler_capture_v1';manifest=json.loads((folder/'manifest.json').read_text())
    result={'method':'Eight distinct packed weight/scale pairs, complete residual/norm/quant plus projection vs fused producer-consumer. Nine real frozen inputs from layers 0/17/35 and steps 0/10/31 for each projection family; private-stream output and intermediate comparisons. Debug stores excluded from timing.','cases':{}}
    for name in ('qkv','up'):
        fused=name=='up';inputs=[];weights={}
        for record in manifest['records']:
            if record['kind']!='norm_quant':continue
            index=int(re.search(r'norm(\d+)',record['label'])[1])
            if bool(index%2)!=fused:continue
            d=cuda(torch.load(folder/record['file'],weights_only=True));inputs.append((record['label'],index//2,d))
            if index//2 not in weights:
                saved=torch.load(RESULTS/f'gptq_v1_g128_d10/{index//2:02d}_{name}.pt',map_location='cuda',weights_only=True)
                weights[index//2]=(pack_interleaved(saved['packed']),saved['scales'].bfloat16())
        def old(d,pair,debug=False):
            summed,x,q=norm_quant(d['x'],d['residual'],d['weight'],d['eps'],ag,8)
            cfg=dict(plan['projections'][name]);cfg['fused']=False
            output=reference(x,*pair,**cfg,paired=fused,prequantized=q,activation_group=ag)
            value=(summed,output)
            if fused and args.output_quant:value=(summed,output,quantize(output,ag))
            return (value,(x,*q)) if debug else value
        def new(d,pair,c,debug=False):return linear(d['x'],d['residual'],d['weight'],d['eps'],*pair,fused=fused,debug=debug,activation_group=ag,**c)
        label,layer,d=inputs[0];w,s=weights[layer];ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
        configs=[{'rows':r,'integer_groups':ig,'integer_rows':ir} for r in ((8,16,32) if fused else (4,8,16)) for ig in (1,2) for ir in (1,4)]
        if fused and args.output_quant:
            configs=[{**c,'output_quant':True} for c in configs if c['rows']%ag==0]
        case={'reference_us':measure(lambda pair:old(d,pair),ring),'input_labels':[r[0] for r in inputs],'candidates':[]};result['cases'][name]=case
        for config in configs:
            row={'config':config,'checks':0,'exact_checks':0,'mismatches':0,'records':[],'max_relative_rms':0.}
            try:
                for label,layer,d in inputs:
                    pair=weights[layer];expected=old(d,pair,True)
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):actual=new(d,pair,config,True)
                    torch.cuda.current_stream().wait_stream(stream)
                    counts=[int((a!=b).sum()) for a,b in zip(flatten(actual),flatten(expected),strict=True)]
                    aa,bb=flatten(actual),flatten(expected)
                    error=((aa[1].float()-bb[1].float()).square().mean()/bb[1].float().square().mean().clamp_min(1e-20)).sqrt().item()
                    row['max_relative_rms']=max(row['max_relative_rms'],error)
                    assert counts[0]==0 and not any(counts[4 if fused and args.output_quant else 2:]),'Producer intermediates differ'
                    row['records'].append({'input':label,'mismatches':counts});row['checks']+=1;row['exact_checks']+=not any(counts);row['mismatches']+=sum(counts)
                d=inputs[0][2];row['us']=measure(lambda pair:new(d,pair,config),ring)
            except Exception as e:row['error']=str(e)
            case['candidates'].append(row)
            print(name,{k:v for k,v in row.items() if k!='records'},flush=True);outfile.write_text(json.dumps(result,indent=2)+'\n')
        valid=[r for r in case['candidates'] if r['checks']==len(inputs) and r['max_relative_rms']<.001 and 'us' in r]
        case['best']=min(valid,key=lambda r:r['us']) if valid else None
        valid=[r for r in valid if r['exact_checks']==len(inputs)]
        case['best_exact']=min(valid,key=lambda r:r['us']) if valid else None
        print('DONE',name,'REF',case['reference_us'],'BEST',None if not valid else {k:v for k,v in case['best_exact'].items() if k!='records'},flush=True)
        outfile.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
