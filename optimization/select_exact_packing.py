"""Filter measured packing plans by exact BF16 results across real inputs."""
import json
import hashlib
import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved,linear
from .dp4a_gateup import gateup
from .int4_dp4a import int4_dp4a


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    source=RESULTS/'dp4a_packing_kernels.json'
    tuning=json.loads(source.read_text())
    records=[];plan={}
    for case in tuning['cases']:
        projection=case['projection']
        candidates=[dict(t,passed_exact_cases=0,first_mismatch=None) for t in case['trials']]
        for layer in (0,17,35):
            name=f'{layer:02d}_{projection}'
            data=torch.load(RESULTS/'gptq_v1_g32_d10'/(name+'.pt'),map_location='cuda',weights_only=True)
            original=data['packed'];interleaved=pack_interleaved(original)
            scales=data['scales'].float();short_scales=data['scales']
            samples=torch.load(RESULTS/'calibration_v1'/(name+'.pt'),weights_only=True)
            zero=torch.zeros(1,samples.shape[-1],device='cuda',dtype=torch.bfloat16)
            spike=zero.clone();spike[0,-1]=100
            inputs=[(str(i),samples[i:i+1].cuda()) for i in (0,231,528,1186)]+[('zero',zero),('spike',spike)]
            for label,x in inputs:
                expected=gateup(x,original,scales,32,1) if projection=='up' else int4_dp4a(x,original,scales,32,2 if projection=='down' else 1,True,grouped_activation=True)
                for cfg in candidates:
                    if cfg['first_mismatch'] is not None:continue
                    w=interleaved if cfg['scheme']=='interleaved' else original
                    s=short_scales if cfg['scale_dtype']=='bfloat16' else scales
                    actual=linear(x,w,s,rows=cfg['rows'],warps=cfg['warps'],interleaved=cfg['scheme']=='interleaved',paired=projection=='up')
                    if torch.equal(actual,expected):cfg['passed_exact_cases']+=1
                    else:cfg['first_mismatch']={'layer':layer,'input':label,'unequal_elements':int((actual!=expected).sum())}
            del data,original,interleaved,scales,short_scales,samples,inputs,x,actual,expected
        eligible=[c for c in candidates if c['first_mismatch'] is None]
        assert eligible,projection+' has no exact plan'
        best=min(eligible,key=lambda c:c['us'])
        plan[projection]={k:best[k] for k in ('scheme','scale_dtype','rows','warps')}
        record={'projection':projection,'eligible':len(eligible),'best':best,'candidates':candidates}
        records.append(record);print(projection,'ELIGIBLE',len(eligible),'BEST',best,flush=True)
        (RESULTS/'dp4a_packing_exact_selection.json').write_text(json.dumps(records,indent=2)+'\n')
    artifact={'format':'dp4a_packing_v1','group':32,'codebooks':32,'tuning_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'selection':'Fastest measured configuration passing all 18 exact real/edge comparisons per projection. Full-model equivalence still requires verification.',
              'projections':plan}
    (RESULTS/'dp4a_packing_exact_plan.json').write_text(json.dumps(artifact,indent=2)+'\n')


if __name__=='__main__':main()
