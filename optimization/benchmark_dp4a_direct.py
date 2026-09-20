"""Cold weight/scale ring timings and exact real-activation checks for direct loads."""
import json
import hashlib
import torch
import triton
from .common import RESULTS
from .dp4a_packing import linear as reference,pack_interleaved
from .dp4a_direct import linear,_direct_gemv
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    plan=json.loads((RESULTS/'dp4a_packing_exact_plan.json').read_text())
    results={'method':'Eight distinct weight and scale pairs, CUDA graphs, median of nine replays. Grouped activation quantization included. Private-stream exact checks: three layers, four recorded inputs, zero and spike.','cases':[]}
    for name in ('qkv','out','up','down'):
        cfg=plan['projections'][name];paired=name=='up'
        saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'00_{name}.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(saved['packed']);s=saved['scales'].to(getattr(torch,cfg['scale_dtype']))
        x=torch.load(RESULTS/'calibration_v1'/f'00_{name}.pt',weights_only=True)[231:232].cuda()
        ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
        ref=lambda pair:reference(x,*pair,rows=cfg['rows'],warps=cfg['warps'],interleaved=True,paired=paired)
        expected=ref((w,s))
        case={'projection':name,'reference_us':measure(ref,ring),'candidates':[]}
        for rows in (1,2,4):
            for warps in (1,2,4,8):
                actual=linear(x,w,s,rows=rows,warps=warps,paired=paired)
                error=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
                assert error<.001,(name,rows,warps,error)
                item={'rows':rows,'warps':warps,'us':measure(lambda pair:linear(x,*pair,rows=rows,warps=warps,paired=paired),ring),
                    'first_relative_rms':error,'exact_checks':0,'max_relative_rms':error,'mismatches':0}
                case['candidates'].append(item)
        del ring,saved,w,s,x,actual,expected
        for layer in (0,17,35):
            saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
            w=pack_interleaved(saved['packed']);s=saved['scales'].to(getattr(torch,cfg['scale_dtype']))
            xs=torch.load(RESULTS/'calibration_v1'/f'{layer:02d}_{name}.pt',weights_only=True)
            zero=torch.zeros(1,xs.shape[-1],device='cuda',dtype=torch.bfloat16)
            spike=zero.clone();spike[0,-1]=100
            for x in [xs[i:i+1].cuda() for i in (0,231,528,1186)]+[zero,spike]:
                expected=reference(x,w,s,rows=cfg['rows'],warps=cfg['warps'],interleaved=True,paired=paired)
                for c in case['candidates']:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):actual=linear(x,w,s,rows=c['rows'],warps=c['warps'],paired=paired)
                    torch.cuda.current_stream().wait_stream(stream)
                    error=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-30)).item()
                    assert error<.001,(name,layer,c,error)
                    c['max_relative_rms']=max(c['max_relative_rms'],error)
                    c['exact_checks']+=int(torch.equal(actual,expected))
                    c['mismatches']+=int((actual!=expected).sum())
        eligible=[c for c in case['candidates'] if c['exact_checks']==18]
        case['best']=min(eligible,key=lambda c:c['us']) if eligible else None
        case['fastest']=min(case['candidates'],key=lambda c:c['us'])
        if case['best']:
            c=case['best'];n,k=saved['shape'];on=n//2 if paired else n
            q=torch.zeros(k,device='cuda',dtype=torch.int8);sx=torch.ones(k//32,device='cuda');y=torch.empty(on,device='cuda',dtype=torch.bfloat16)
            kernel=_direct_gemv[(triton.cdiv(on,c['rows']),)](q,sx,w,s,y,on,k,triton.next_power_of_2(k//32),c['rows'],paired,num_warps=c['warps'])
            c['registers']=kernel.n_regs;c['spills']=kernel.n_spills;c['shared_bytes']=kernel.metadata.shared
            for ext in ('ptx','ttgir'):(RESULTS/f'direct_{name}.{ext}').write_text(kernel.asm[ext])
            del q,sx,y
        results['cases'].append(case)
        print(name,'REFERENCE',case['reference_us'],'EXACT BEST',case['best'],'FASTEST',case['fastest'],flush=True)
        (RESULTS/'dp4a_direct_kernels.json').write_text(json.dumps(results,indent=2)+'\n')
        del saved,w,s,xs,x,zero,spike,actual,expected
    # Select without consulting synthesis/quality results; preserve the existing
    # operator where the cold-ring gain is below two percent.
    for case in results['cases']:
        best=case['best']
        if best and best['us']<case['reference_us']*.98:
            plan['projections'][case['projection']].update(rows=best['rows'],warps=best['warps'],activation_load='direct')
    plan['direct_tuning_sha256']=hashlib.sha256((RESULTS/'dp4a_direct_kernels.json').read_bytes()).hexdigest()
    plan['selection']='Direct activation loads only where 18/18 exact comparisons pass and median operator gain exceeds 2%; other projections use the previous matching plan. Full-model verification recorded separately.'
    (RESULTS/'dp4a_direct_exact_plan.json').write_text(json.dumps(plan,indent=2)+'\n')


if __name__=='__main__':main()
