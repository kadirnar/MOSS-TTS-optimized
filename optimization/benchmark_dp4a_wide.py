"""Larger row tiles for remaining QKV/output/down projections after fusion."""
import json
import torch
from .common import RESULTS
from .dp4a_packing import linear as packed_linear,pack_interleaved
from .dp4a_direct import linear as direct_linear
from .benchmark_gateup_quant import quantize
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    plan=json.loads((RESULTS/'dp4a_direct_exact_plan.json').read_text());cases=[]
    for name in ('qkv','out','down'):
        cfg=plan['projections'][name]
        saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'00_{name}.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(saved['packed']);s=saved['scales'].to(getattr(torch,cfg['scale_dtype']))
        x=torch.load(RESULTS/'calibration_v1'/f'00_{name}.pt',weights_only=True)[231:232].cuda();qx=quantize(x)
        def baseline(pair):
            if cfg.get('activation_load')=='direct':return direct_linear(x,*pair,rows=cfg['rows'],warps=cfg['warps'],prequantized=qx)
            return packed_linear(x,*pair,rows=cfg['rows'],warps=cfg['warps'],interleaved=True,prequantized=qx)
        def candidate(pair,c):return direct_linear(x,*pair,rows=c['rows'],warps=c['warps'],prequantized=qx)
        ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
        case={'projection':name,'reference_us':measure(baseline,ring),'candidates':[]}
        for rows in (8,16,32):
            for warps in (2,4,8):
                c={'rows':rows,'warps':warps}
                actual=candidate((w,s),c);expected=baseline((w,s))
                c.update(us=measure(lambda pair:candidate(pair,c),ring),first_mismatches=int((actual!=expected).sum()),exact_checks=0,mismatches=0)
                case['candidates'].append(c)
        del ring,saved,w,s,x,qx,actual,expected
        for layer in (0,17,35):
            saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
            w=pack_interleaved(saved['packed']);s=saved['scales'].to(getattr(torch,cfg['scale_dtype']))
            xs=torch.load(RESULTS/'calibration_v1'/f'{layer:02d}_{name}.pt',weights_only=True)
            zero=torch.zeros(1,xs.shape[-1],device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
            for x in [xs[i:i+1].cuda() for i in (0,231,528,1186)]+[zero,spike]:
                qx=quantize(x);expected=baseline((w,s))
                for c in case['candidates']:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):actual=candidate((w,s),c)
                    torch.cuda.current_stream().wait_stream(stream)
                    mismatches=int((actual!=expected).sum());c['mismatches']+=mismatches;c['exact_checks']+=int(mismatches==0)
            del saved,w,s,xs,x,qx,zero,spike,actual,expected
        eligible=[c for c in case['candidates'] if c['exact_checks']==18]
        case['best_exact']=min(eligible,key=lambda c:c['us']) if eligible else None
        cases.append(case);print(case,flush=True)
        (RESULTS/'dp4a_wide_kernels.json').write_text(json.dumps({'method':'Eight distinct weight/scale pairs, prequantized activation as supplied by the fused decoder, 9 graph timings; three layers x four captured inputs plus zero/spike, private stream. No new plan is selected automatically.','cases':cases},indent=2)+'\n')


if __name__=='__main__':main()
