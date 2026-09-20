"""Exhaustive signed expansion plus selected layouts on real/edge inputs."""
import hashlib
import json
import torch
import triton
import triton.language as tl
from .common import RESULTS
from .calibrated_backend import unpack_signed
from .dp4a_packing import _expand_prmt,pack_interleaved,unpack_interleaved,linear
from .dp4a_gateup import gateup
from .int4_dp4a import int4_dp4a


@triton.jit
def _expand_test(X,Y,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(Y+i,_expand_prmt(tl.load(X+i)))


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    words=torch.arange(65536,device='cuda',dtype=torch.int32)
    expanded=torch.empty_like(words)
    _expand_test[(256,)](words,expanded,256)
    nibble=(words[:,None]>>(torch.arange(4,device='cuda')*4))&15
    signed=torch.where(nibble>=8,nibble-16,nibble)
    expected=(signed.to(torch.uint8).long()<<(torch.arange(4,device='cuda')*8)).sum(-1).int()
    assert torch.equal(expected,expanded),'PRMT signed expansion'
    other=words^0xabcd
    packed=torch.stack([words&255,words>>8,other&255,other>>8],1).byte()
    assert torch.equal(unpack_signed(packed),unpack_interleaved(pack_interleaved(packed)))
    tuning_path=RESULTS/'dp4a_packing_kernels.json'
    tuning=json.loads(tuning_path.read_text())
    plan={c['projection']:{k:c['best'][k] for k in ('scheme','scale_dtype','rows','warps')} for c in tuning['cases']}
    assert set(plan)=={'qkv','out','up','down'}
    rows=[]
    for layer in (0,17,35):
        for projection in ('qkv','out','up','down'):
            name=f'{layer:02d}_{projection}'
            data=torch.load(RESULTS/'gptq_v1_g32_d10'/(name+'.pt'),map_location='cuda',weights_only=True)
            original,scales=data['packed'],data['scales'].float()
            cfg=plan[projection]
            packed=pack_interleaved(original) if cfg['scheme']=='interleaved' else original
            selected_scales=scales.bfloat16() if cfg['scale_dtype']=='bfloat16' else scales
            assert torch.equal(selected_scales.float(),scales),'Lossy scale conversion'
            samples=torch.load(RESULTS/'calibration_v1'/(name+'.pt'),weights_only=True)
            zero=torch.zeros(1,samples.shape[-1],device='cuda',dtype=torch.bfloat16)
            spike=zero.clone();spike[0,-1]=100
            inputs=[(str(i),samples[i:i+1].cuda()) for i in (0,231,528,1186)]+[('zero',zero),('spike',spike)]
            for label,x in inputs:
                expected=gateup(x,original,scales,32,1) if projection=='up' else int4_dp4a(x,original,scales,32,2 if projection=='down' else 1,True,grouped_activation=True)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    actual=linear(x,packed,selected_scales,rows=cfg['rows'],warps=cfg['warps'],interleaved=cfg['scheme']=='interleaved',paired=projection=='up')
                torch.cuda.current_stream().wait_stream(stream)
                relative=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-12)).item()
                assert torch.isfinite(actual).all() and relative<.001,(name,label,relative)
                rows.append({'projection':name,'input':label,'relative_rms':relative,'exact':torch.equal(actual,expected)})
            del data,original,scales,packed,selected_scales,samples,inputs,x,actual,expected
    result={'exhaustive_prmt_words':65536,'exhaustive_interleaved_rows':65536,'private_stream':True,'torch':torch.__version__,'cases':rows}
    (RESULTS/'dp4a_packing_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    artifact={'format':'dp4a_packing_v1','group':32,'codebooks':32,'tuning_sha256':hashlib.sha256(tuning_path.read_bytes()).hexdigest(),'projections':plan}
    (RESULTS/'dp4a_packing_plan.json').write_text(json.dumps(artifact,indent=2)+'\n')
    print('PASSED',len(rows),'real/edge cases;',sum(r['exact'] for r in rows),'exact; max relative RMS',max(r['relative_rms'] for r in rows),flush=True)
    print(json.dumps(artifact,indent=2),flush=True)


if __name__=='__main__':main()
