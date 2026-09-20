"""All-layer exact comparison and repeated ring timing for scaled INT4 unpacking."""
import argparse
import hashlib
import json
import torch
import triton
from .common import RESULTS
from .dp4a_packing import pack_interleaved,linear as packed
from .dp4a_direct import linear as direct
from .dp4a_gateup_quant import gateup_quant
from .dp4a_scaled import linear,_gemv
from .benchmark_gateup_quant import quantize
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v1');parser.add_argument('--sweep-tag',default='v1');args=parser.parse_args()
    if any(not tag or not all(c.isalnum() or c=='_' for c in tag) for tag in (args.tag,args.sweep_tag)):raise ValueError('Safe nonempty tags required')
    if (RESULTS/f'dp4a_scaled_validation_{args.tag}.json').exists():raise FileExistsError('Choose a fresh tag to preserve results')
    torch.set_num_threads(4);records=[];resources={};timings={};selected={}
    sweep=json.loads((RESULTS/f'dp4a_scaled_kernels_{args.sweep_tag}.json').read_text())
    plan=json.loads((RESULTS/'dp4a_direct_exact_plan.json').read_text())
    for name in ('up','qkv','out','down'):
        cfg=plan['projections'][name];best=sweep['cases'][name]['best_exact']['config'];selected[name]=best
        def old(x,w,s,qx):
            if name=='up':return gateup_quant(x,w,s,prequantized=qx)
            fn=direct if cfg.get('activation_load')=='direct' else packed
            return fn(x,w,s,rows=cfg['rows'],warps=cfg['warps'],prequantized=qx,**({} if fn==direct else {'interleaved':True}))
        def new(x,w,s,qx):
            return linear(x,w,s,prequantized=qx,paired=name=='up',fused=name=='up',scale_mode=4 if name=='up' else 0,**best)
        for layer in range(36):
            saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
            w=pack_interleaved(saved['packed']);s=saved['scales'].to(getattr(torch,cfg['scale_dtype']))
            xs=torch.load(RESULTS/'calibration_v1'/f'{layer:02d}_{name}.pt',weights_only=True)
            zero=torch.zeros(1,xs.shape[-1],device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
            inputs=[(str(i),xs[i:i+1].cuda()) for i in (0,37,116,231,352,463,528,671,829,1007,1186,1275)]+[('zero',zero),('spike',spike)]
            for label,x in inputs:
                qx=quantize(x);expected=old(x,w,s,qx)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):actual=new(x,w,s,qx)
                torch.cuda.current_stream().wait_stream(stream)
                aa=(actual[0],*actual[1]) if name=='up' else (actual,)
                bb=(expected[0],*expected[1]) if name=='up' else (expected,)
                counts=[int((a!=b).sum()) for a,b in zip(aa,bb)]
                records.append({'projection':name,'layer':layer,'input':label,'mismatches':counts})
            if layer==0:
                y=actual[0] if name=='up' else actual;oq,os=actual[1] if name=='up' else (torch.empty(0,device='cuda',dtype=torch.int8),torch.empty(0,device='cuda'))
                n=y.numel();k=x.numel()
                kernel=_gemv[(triton.cdiv(n,best['rows']),)](*qx,w,s,y,oq,os,n,k,triton.next_power_of_2(k//32),best['rows'],best['mode'],name=='up',name=='up',4 if name=='up' else 0,num_warps=best['warps'])
                resources[name]={'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared}
                for ext in ('ptx','ttgir'):(RESULTS/f'dp4a_scaled_{name}_{args.tag}.{ext}').write_text(kernel.asm[ext])
                ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
                timings[name]={'reference_us':measure(lambda pair:old(x,*pair,qx),ring),'scaled_us':measure(lambda pair:new(x,*pair,qx),ring)}
                del ring,y,oq,os
            print(name,layer,'mismatches',sum(sum(r['mismatches']) for r in records if r['projection']==name and r['layer']==layer),flush=True)
            del saved,w,s,xs,zero,spike,inputs,x,qx,actual,expected,aa,bb
    exact={name:all(not any(r['mismatches']) for r in records if r['projection']==name) for name in selected}
    result={'checks':len(records),'records':records,'selected':selected,'resources':resources,'repeat_timings':timings,'exact_by_projection':exact,'all_exact':all(exact.values()),'private_stream':True,'scope':'36 layers x 4 projection families x (12 recorded BF16 activation vectors + zero + spike); BF16 outputs and fused grouped INT8 values/FP32 scales.'}
    path=RESULTS/f'dp4a_scaled_validation_{args.tag}.json';path.write_text(json.dumps(result,indent=2)+'\n')
    if result['all_exact']:
        (RESULTS/f'dp4a_scaled_plan_{args.tag}.json').write_text(json.dumps({'format':'dp4a_scaled_v1','codebooks':32,'group':32,'projections':selected,'validation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selection':'All operators exact on 2016 real/edge checks. Full-model validation still required.'},indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='records'},flush=True)


if __name__=='__main__':main()
