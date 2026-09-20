"""Check whether sharing the gate/up activation load improves the scaled path."""
import argparse
import json
import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_scaled import linear
from .dp4a_pair import gateup,_pair_gemv
from .benchmark_gateup_quant import quantize
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v1');args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe nonempty tag required')
    outfile=RESULTS/f'dp4a_pair_kernels_{args.tag}.json'
    if outfile.exists():raise FileExistsError('Choose a fresh tag')
    torch.set_num_threads(4);configs=[{'rows':r,'warps':w} for r in (32,64) for w in (2,4,8)]
    result={'method':'8 distinct weights/scales, input prequantized, median of nine graphs; selected scaled-integer kernel is the reference. Three layers times four recorded vectors plus zero/spike.','candidates':[]}
    saved=torch.load(RESULTS/'gptq_v1_g32_d10/00_up.pt',map_location='cuda',weights_only=True)
    w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16();x=torch.load(RESULTS/'calibration_v1/00_up.pt',weights_only=True)[231:232].cuda();qx=quantize(x)
    ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
    def reference(w,s):return linear(x,w,s,rows=32,warps=4,mode=2,paired=True,fused=True,scale_mode=4,prequantized=qx)
    result['reference_us']=measure(lambda pair:reference(*pair),ring)
    for c in configs:
        row={'config':c,'mismatches':0,'exact_checks':0}
        try:row['us']=measure(lambda pair:gateup(x,*pair,prequantized=qx,**c),ring)
        except Exception as e:row['error']=repr(e)
        result['candidates'].append(row);print(row,flush=True)
    del ring,saved,w,s,x,qx
    for layer in (0,17,35):
        saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{layer:02d}_up.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16();xs=torch.load(RESULTS/'calibration_v1'/f'{layer:02d}_up.pt',weights_only=True)
        zero=torch.zeros(1,4096,device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
        for x in [xs[i:i+1].cuda() for i in (0,231,528,1186)]+[zero,spike]:
            qx=quantize(x);expected=reference(w,s)
            for row in result['candidates']:
                if 'error' in row:continue
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):actual=gateup(x,w,s,prequantized=qx,**row['config'])
                torch.cuda.current_stream().wait_stream(stream)
                count=sum(int((a!=b).sum()) for a,b in zip((actual[0],*actual[1]),(expected[0],*expected[1])))
                row['mismatches']+=count;row['exact_checks']+=int(not count)
        del saved,w,s,xs,x,qx,expected,actual,zero,spike
    valid=[c for c in result['candidates'] if c['exact_checks']==18]
    result['best_exact']=min(valid,key=lambda c:c['us']) if valid else None
    outfile.write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':main()
