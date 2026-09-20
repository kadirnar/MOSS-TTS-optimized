"""Cold-weight ring timings and real/edge numerical gate/up fusion checks."""
import json
import argparse
import torch
import triton
from .common import RESULTS
from .dp4a_packing import linear as packed_linear,pack_interleaved
from .dp4a_direct import linear as direct_linear
from .dp4a_gateup_quant import gateup_quant,_gateup_quant
from .int4_dp4a import _int8_activation_grouped
from .tune_weight_reads import measure


def quantize(x):
    n=x.numel();q=torch.empty(n,device=x.device,dtype=torch.int8);s=torch.empty(n//32,device=x.device)
    _int8_activation_grouped[(triton.cdiv(n//32,4),)](x,q,s,n,32,4,num_warps=4)
    return q,s


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scale-sweep',action='store_true');parser.add_argument('--row-sweep',action='store_true');args=parser.parse_args()
    outfile=RESULTS/('gateup_quant_row_kernels.json' if args.row_sweep else ('gateup_quant_scale_kernels.json' if args.scale_sweep else 'gateup_quant_kernels.json'))
    torch.set_num_threads(4)
    plan=json.loads((RESULTS/'dp4a_direct_exact_plan.json').read_text());cfg=plan['projections']['up']
    saved=torch.load(RESULTS/'gptq_v1_g32_d10'/'00_up.pt',map_location='cuda',weights_only=True)
    w=pack_interleaved(saved['packed']);s=saved['scales'].to(getattr(torch,cfg['scale_dtype']))
    x=torch.load(RESULTS/'calibration_v1'/'00_up.pt',weights_only=True)[231:232].cuda();qx=quantize(x)
    ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
    def reference(pair):
        out=packed_linear(x,*pair,rows=4,warps=4,interleaved=True,paired=True,prequantized=qx)
        return out,quantize(out)
    def candidate(pair,c):
        if 'row_tile' in c:
            from .dp4a_gateup_pipeline import gateup_pipeline
            return gateup_pipeline(x,*pair,tile=c['row_tile'],warps=c['warps'],prequantized=qx)
        if c['fused']:return gateup_quant(x,*pair,rows=c['rows'],warps=c['warps'],direct=c['direct'],prequantized=qx,scale_mode=c.get('scale_mode',0))
        fn=direct_linear if c['direct'] else packed_linear
        kw={} if c['direct'] else {'interleaved':True}
        out=fn(x,*pair,rows=c['rows'],warps=c['warps'],paired=True,prequantized=qx,**kw)
        return out,quantize(out)
    result={'reference_us':measure(reference,ring),'method':'8 distinct weight/scale pairs; median 9 graph replays, input prequantized as in fused model; gate/up plus output G32 quantization timed together. Three layers, four captured activations, zero and spike checks on private streams.','candidates':[]}
    configs=[{'fused':f,'direct':d,'rows':r,'warps':nw} for f in (False,True) for d in (False,True) for r in ((8,16,32,64) if not f else (32,64)) for nw in (2,4,8)]
    if args.scale_sweep:configs=[{'fused':True,'direct':True,'rows':r,'warps':w,'scale_mode':m} for r in (32,64) for w in (4,8) for m in (1,2,4,8)]
    if args.row_sweep:configs=[{'fused':True,'direct':True,'rows':32,'warps':w,'row_tile':t} for t in (8,16) for w in (2,4,8)]+[{'fused':True,'direct':True,'rows':32,'warps':4,'scale_mode':4}]
    for c in configs:
        try:
            actual=candidate((w,s),c);expected=reference((w,s))
            c['us']=measure(lambda pair:candidate(pair,c),ring)
            c['first_mismatches']=int((actual[0]!=expected[0]).sum())
            c['exact_checks']=0;c['output_mismatches']=0;c['quant_mismatches']=0;c['scale_mismatches']=0;c['max_relative_rms']=0.
        except Exception as e:c['error']=repr(e)
        result['candidates'].append(c);print(c,flush=True)
        outfile.write_text(json.dumps(result,indent=2)+'\n')
    del ring,saved,w,s,x,qx,actual,expected
    for layer in (0,17,35):
        saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{layer:02d}_up.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16()
        xs=torch.load(RESULTS/'calibration_v1'/f'{layer:02d}_up.pt',weights_only=True)
        zero=torch.zeros(1,xs.shape[-1],device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
        for x in [xs[i:i+1].cuda() for i in (0,231,528,1186)]+[zero,spike]:
            qx=quantize(x);expected=reference((w,s))
            for c in result['candidates']:
                if 'error' in c:continue
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):actual=candidate((w,s),c)
                torch.cuda.current_stream().wait_stream(stream)
                counts=[int((a!=b).sum()) for a,b in zip((actual[0],*actual[1]),(expected[0],*expected[1]))]
                for key,count in zip(('output_mismatches','quant_mismatches','scale_mismatches'),counts):c[key]+=count
                c['exact_checks']+=int(not any(counts))
                err=float((actual[0].float()-expected[0].float()).square().mean().sqrt()/expected[0].float().square().mean().sqrt().clamp_min(1e-30))
                c['max_relative_rms']=max(c['max_relative_rms'],err)
        del saved,w,s,xs,x,qx,zero,spike,actual,expected
    eligible=[c for c in result['candidates'] if c.get('exact_checks')==18]
    result['best_exact']=min(eligible,key=lambda c:c['us']) if eligible else None
    result['fastest']=min((c for c in result['candidates'] if 'us' in c),key=lambda c:c['us'])
    outfile.write_text(json.dumps(result,indent=2)+'\n')
    print('REFERENCE',result['reference_us'],'BEST EXACT',result['best_exact'],'FASTEST',result['fastest'],flush=True)


if __name__=='__main__':main()
