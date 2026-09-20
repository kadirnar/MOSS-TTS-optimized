"""G128 weight / G32 or G128 activation kernel tradeoff on calibrated inputs."""
import argparse
import json

import torch
import triton

from .common import RESULTS
from .dp4a_group128 import linear
from .dp4a_packing import pack_interleaved
from .int4_dp4a import _int8_activation_grouped, int4_dp4a
from .kernels import silu_mul
from .tune_weight_reads import measure


def quantize(x, group):
    k=x.numel();q=torch.empty(k,device=x.device,dtype=torch.int8);s=torch.empty(k//group,device=x.device)
    _int8_activation_grouped[(triton.cdiv(k//group,4),)](x,q,s,k,group,4,num_warps=4)
    return q,s


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    target=RESULTS/f'group128_kernels_{args.tag}.json'
    if target.exists():raise FileExistsError('Preserve existing results')
    torch.set_num_threads(4)
    result={'method':'G128 GPTQ weights, eight-weight ring, median nine CUDA graph timings. Three layers x four real inputs plus zero/spike. Up includes following activation quantization for every candidate. Arithmetic reference is independent original-layout DP4A with repeated weight scales for G32 activations.','cases':{}}
    for ag in (128,32):
        for name in ('qkv','out','up','down'):
            paired=name=='up'
            def candidate(x,w,s,qx,c):
                y=linear(x,w,s,prequantized=qx,paired=paired,activation_group=ag,**c)
                return y if not paired or c['fused'] else (y,quantize(y,ag))
            def reference(x,saved,qx):
                scales=saved['scales'].float().repeat_interleave(128//ag,dim=1)
                y=int4_dp4a(x,saved['packed'],scales,ag,4,True,qx,True)
                if paired:
                    y=silu_mul(y);return y,quantize(y,ag)
                return y
            configs=[{'rows':r,'warps':w,'mode':2,'fused':False} for r in ((4,8,16,32) if paired else (2,4,8)) for w in (2,4,8)]
            if paired:configs += [{'rows':r,'warps':w,'mode':2,'fused':True} for r in ((128,) if ag==128 else (32,64)) for w in (4,8,16)]
            case={'candidates':[]};result['cases'][f'{name}_a{ag}']=case
            saved=torch.load(RESULTS/f'gptq_v1_g128_d10/00_{name}.pt',map_location='cuda',weights_only=True)
            w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16()
            x=torch.load(RESULTS/f'calibration_v1/00_{name}.pt',weights_only=True)[231:232].cuda();qx=quantize(x,ag)
            ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
            for c in configs:
                row={'config':c,'checks':0,'exact_checks':0,'max_relative_rms':0.,'mismatches':0}
                try:row['us']=measure(lambda ws:candidate(x,*ws,qx,c),ring)
                except Exception as e:row['error']=repr(e)
                case['candidates'].append(row);print(name,ag,row,flush=True)
                target.write_text(json.dumps(result,indent=2)+'\n')
            del ring,saved,w,s,x,qx
            for layer in (0,17,35):
                saved=torch.load(RESULTS/f'gptq_v1_g128_d10/{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
                w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16()
                xs=torch.load(RESULTS/f'calibration_v1/{layer:02d}_{name}.pt',weights_only=True)
                zero=torch.zeros(1,xs.shape[-1],device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
                for x in [xs[i:i+1].cuda() for i in (0,231,528,1186)]+[zero,spike]:
                    qx=quantize(x,ag);expected=reference(x,saved,qx)
                    for row in case['candidates']:
                        if 'error' in row:continue
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):actual=candidate(x,w,s,qx,row['config'])
                        torch.cuda.current_stream().wait_stream(stream)
                        aa=(actual[0],*actual[1]) if paired else (actual,)
                        bb=(expected[0],*expected[1]) if paired else (expected,)
                        count=sum(int((a!=b).sum()) for a,b in zip(aa,bb))
                        error=((aa[0].float()-bb[0].float()).square().mean()/bb[0].float().square().mean().clamp_min(1e-20)).sqrt().item()
                        row['checks']+=1;row['exact_checks']+=int(count==0);row['mismatches']+=count
                        row['max_relative_rms']=max(row['max_relative_rms'],error)
                del saved,w,s,xs,zero,spike,x,qx,expected,actual,aa,bb
            valid=[r for r in case['candidates'] if 'us' in r and r['checks']==18 and r['max_relative_rms']<.001]
            case['best']=min(valid,key=lambda r:r['us']) if valid else None
            exact=[r for r in valid if r['exact_checks']==18]
            case['best_exact']=min(exact,key=lambda r:r['us']) if exact else None
            print('DONE',name,ag,'BEST',case['best'],'EXACT',case['best_exact'],flush=True)
            target.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
