"""Native projection warp/register tiling with exact real-input checks."""
import argparse
import json
import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_scaled import SELECTED,linear as reference
from .dp4a_staged import linear,library
from .benchmark_gateup_quant import quantize
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    outfile=RESULTS/f'dp4a_staged_kernels_{args.tag}.json';assert not outfile.exists(),'Preserve old results'
    torch.set_num_threads(4);library()
    result={'method':'8 distinct weight/scale pairs, prequantized input, median nine graph timings; three layers x four recorded vectors plus zero/spike, private-stream comparison against selected scaled-integer Triton kernels.','cases':{}}
    for name in ('qkv','out','up','down'):
        def old(x,w,s,qx):return reference(x,w,s,**SELECTED[name],paired=name=='up',fused=name=='up',scale_mode=4 if name=='up' else 0,prequantized=qx)
        def candidate(x,w,s,qx,c):return linear(x,w,s,prequantized=qx,fused=name=='up',**c)
        saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'00_{name}.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16() if name in ('qkv','up') else saved['scales'].float()
        x=torch.load(RESULTS/'calibration_v1'/f'00_{name}.pt',weights_only=True)[231:232].cuda();qx=quantize(x)
        ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
        actual=aa=bb=None
        configs=[{'rows':r,'warps':nw} for r in ((32,64) if name=='up' else (4,8,16)) for nw in (4,8)]
        case={'reference_us':measure(lambda pair:old(x,*pair,qx),ring),'candidates':[]};result['cases'][name]=case
        expected=old(x,w,s,qx)
        for c in configs:
            row={'config':c,'checks':0,'exact_checks':0,'mismatches':0}
            try:
                actual=candidate(x,*ring[0],qx,c)
                aa=(actual[0],*actual[1]) if name=='up' else (actual,);bb=(expected[0],*expected[1]) if name=='up' else (expected,)
                row['first_mismatches']=[int((a!=b).sum()) for a,b in zip(aa,bb)]
                row['us']=measure(lambda pair:candidate(x,*pair,qx,c),ring)
            except Exception as e:row['error']=repr(e)
            case['candidates'].append(row);print(name,row,flush=True);outfile.write_text(json.dumps(result,indent=2)+'\n')
        del saved,ring,w,s,x,qx,actual,expected,aa,bb
        for layer in (0,17,35):
            saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
            w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16() if name in ('qkv','up') else saved['scales'].float()
            actual=aa=bb=None
            xs=torch.load(RESULTS/'calibration_v1'/f'{layer:02d}_{name}.pt',weights_only=True)
            zero=torch.zeros(1,xs.shape[-1],device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
            for x in [xs[i:i+1].cuda() for i in (0,231,528,1186)]+[zero,spike]:
                qx=quantize(x);expected=old(x,w,s,qx)
                for row in case['candidates']:
                    if 'error' in row:continue
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):actual=candidate(x,w,s,qx,row['config'])
                    torch.cuda.current_stream().wait_stream(stream)
                    aa=(actual[0],*actual[1]) if name=='up' else (actual,);bb=(expected[0],*expected[1]) if name=='up' else (expected,)
                    count=sum(int((a!=b).sum()) for a,b in zip(aa,bb))
                    row['checks']+=1;row['exact_checks']+=int(count==0);row['mismatches']+=count
            del saved,w,s,xs,x,qx,zero,spike,actual,expected,aa,bb
        valid=[c for c in case['candidates'] if c['exact_checks']==18]
        case['best_exact']=min(valid,key=lambda c:c['us']) if valid else None
        print('DONE',name,'REFERENCE',case['reference_us'],'BEST',case['best_exact'],flush=True);outfile.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
