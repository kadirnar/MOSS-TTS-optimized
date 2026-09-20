"""Fuse normalization into actual calibrated projections; retain failed trials."""
import argparse
import json
import re
import torch
from .common import RESULTS
from .compiler_audit import cuda
from .dp4a_packing import pack_interleaved
from .dp4a_fusions import norm_quant
from .dp4a_scaled import linear as reference,SELECTED
from .dp4a_norm_projection import linear
from .tune_weight_reads import measure


def flatten(x):
    if isinstance(x,tuple):return tuple(t for child in x for t in flatten(child))
    return (x,)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    outfile=RESULTS/f'norm_projection_{args.tag}.json'
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
                saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{index//2:02d}_{name}.pt',map_location='cuda',weights_only=True)
                weights[index//2]=(pack_interleaved(saved['packed']),saved['scales'].bfloat16())
        def old(d,pair,debug=False):
            summed,x,q=norm_quant(d['x'],d['residual'],d['weight'],d['eps'],32,8)
            output=reference(x,*pair,**SELECTED[name],paired=fused,fused=fused,scale_mode=4 if fused else 0,prequantized=q)
            value=(summed,*output) if fused else (summed,output)
            return (value,(x,*q)) if debug else value
        def new(d,pair,c,debug=False):return linear(d['x'],d['residual'],d['weight'],d['eps'],*pair,fused=fused,debug=debug,**c)
        label,layer,d=inputs[0];w,s=weights[layer];ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
        configs=[{'rows':r,'integer_groups':ig,'integer_rows':ir} for r in ((32,64) if fused else (4,8,16)) for ig in (1,2) for ir in (1,4)]
        case={'reference_us':measure(lambda pair:old(d,pair),ring),'input_labels':[r[0] for r in inputs],'candidates':[]};result['cases'][name]=case
        for config in configs:
            row={'config':config,'checks':0,'exact_checks':0,'mismatches':0,'records':[]}
            try:
                for label,layer,d in inputs:
                    pair=weights[layer];expected=old(d,pair,True)
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):actual=new(d,pair,config,True)
                    torch.cuda.current_stream().wait_stream(stream)
                    counts=[int((a!=b).sum()) for a,b in zip(flatten(actual),flatten(expected),strict=True)]
                    row['records'].append({'input':label,'mismatches':counts});row['checks']+=1;row['exact_checks']+=not any(counts);row['mismatches']+=sum(counts)
                d=inputs[0][2];row['us']=measure(lambda pair:new(d,pair,config),ring)
            except Exception as e:row['error']=str(e)
            case['candidates'].append(row)
            print(name,{k:v for k,v in row.items() if k!='records'},flush=True);outfile.write_text(json.dumps(result,indent=2)+'\n')
        valid=[r for r in case['candidates'] if r['exact_checks']==len(inputs) and 'us' in r]
        case['best_exact']=min(valid,key=lambda r:r['us']) if valid else None
        print('DONE',name,'REF',case['reference_us'],'BEST',None if not valid else {k:v for k,v in case['best_exact'].items() if k!='records'},flush=True)
        outfile.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
