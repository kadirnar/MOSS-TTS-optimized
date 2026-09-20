"""Compare scaled-integer DP4A unpacking against the selected exact operators."""
import argparse
import json
import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved,linear as packed
from .dp4a_direct import linear as direct
from .dp4a_gateup_quant import gateup_quant
from .dp4a_scaled import linear
from .benchmark_gateup_quant import quantize
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v1');args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe nonempty tag required')
    torch.set_num_threads(4)
    outfile=RESULTS/f'dp4a_scaled_kernels_{args.tag}.json'
    assert not outfile.exists(),'Use a new output name to preserve results'
    plan=json.loads((RESULTS/'dp4a_direct_exact_plan.json').read_text())
    result={'method':'8 distinct weights/scales, median of 9 graph timings, prequantized input; 3 layers x 4 captured inputs plus zero/spike exact checks on private streams. Reference is current selected path including fused gate/up quantization.','cases':{}}
    for name in ('up','qkv','out','down'):
        cfg=plan['projections'][name]
        def reference(x,w,s,qx):
            if name=='up':return gateup_quant(x,w,s,prequantized=qx)
            fn=direct if cfg.get('activation_load')=='direct' else packed
            kw={} if fn==direct else {'interleaved':True}
            return fn(x,w,s,rows=cfg['rows'],warps=cfg['warps'],prequantized=qx,**kw)
        def candidate(x,w,s,qx,c):
            return linear(x,w,s,prequantized=qx,paired=name=='up',fused=name=='up',scale_mode=4 if name=='up' else 0,**c)
        saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'00_{name}.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(saved['packed']);s=saved['scales'].to(getattr(torch,cfg['scale_dtype']))
        x=torch.load(RESULTS/'calibration_v1'/f'00_{name}.pt',weights_only=True)[231:232].cuda();qx=quantize(x)
        ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
        configs=[{'rows':r,'warps':nw,'mode':m} for r in ((32,64) if name=='up' else (2,4,8)) for nw in ((4,8) if name=='up' else (2,4,8)) for m in (1,2)]
        case={'reference_us':measure(lambda pair:reference(x,*pair,qx),ring),'candidates':[]};result['cases'][name]=case
        for c in configs:
            row={'config':c,'checks':0,'exact_checks':0,'mismatches':0}
            try:row['us']=measure(lambda pair:candidate(x,*pair,qx,c),ring)
            except Exception as e:row['error']=repr(e)
            case['candidates'].append(row);print(name,row,flush=True)
            outfile.write_text(json.dumps(result,indent=2)+'\n')
        del ring,saved,w,s,x,qx
        for layer in (0,17,35):
            saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
            w=pack_interleaved(saved['packed']);s=saved['scales'].to(getattr(torch,cfg['scale_dtype']))
            xs=torch.load(RESULTS/'calibration_v1'/f'{layer:02d}_{name}.pt',weights_only=True)
            zero=torch.zeros(1,xs.shape[-1],device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
            for x in [xs[i:i+1].cuda() for i in (0,231,528,1186)]+[zero,spike]:
                qx=quantize(x);expected=reference(x,w,s,qx)
                for row in case['candidates']:
                    if 'error' in row:continue
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):actual=candidate(x,w,s,qx,row['config'])
                    torch.cuda.current_stream().wait_stream(stream)
                    aa=(actual[0],*actual[1]) if name=='up' else (actual,)
                    bb=(expected[0],*expected[1]) if name=='up' else (expected,)
                    count=sum(int((a!=b).sum()) for a,b in zip(aa,bb))
                    row['checks']+=1;row['exact_checks']+=int(count==0);row['mismatches']+=count
            del saved,w,s,xs,zero,spike,x,qx,expected,actual,aa,bb
        valid=[r for r in case['candidates'] if r.get('exact_checks')==18]
        case['best_exact']=min(valid,key=lambda r:r['us']) if valid else None
        print('DONE',name,'REFERENCE',case['reference_us'],'BEST',case['best_exact'],flush=True)
        outfile.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
