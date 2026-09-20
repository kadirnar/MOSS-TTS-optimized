"""Actual calibrated projections: nibble layout, scale dtype and CTA tiling."""
import json
import torch
from .common import RESULTS
from .calibrated_backend import unpack_signed
from .int4_dp4a import int4_dp4a
from .dp4a_gateup import gateup
from .dp4a_packing import linear,pack_interleaved,unpack_interleaved
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    result={'torch':torch.__version__,'method':'CUDA graphs, eight distinct weight AND scale pairs exceeding L2. Grouped input quantization included. Real calibrated layer-zero weights and one held-out input row. Private-stream numerical checks.','cases':[]}
    for projection in ('qkv','out','up','down'):
        name='00_'+projection
        saved=torch.load(RESULTS/'gptq_v1_g32_d10'/(name+'.pt'),map_location='cuda',weights_only=True)
        packed=saved['packed'];bf16_scales=saved['scales'];scales=bf16_scales.float()
        interleaved=pack_interleaved(packed)
        assert torch.equal(unpack_signed(packed),unpack_interleaved(interleaved)),name
        x=torch.load(RESULTS/'calibration_v1'/(name+'.pt'),weights_only=True)[231:232].cuda()
        paired=projection=='up'
        def reference(pair):
            w,s=pair
            return gateup(x,w,s,32,1) if paired else int4_dp4a(x,w,s,32,2 if projection=='down' else 1,True,grouped_activation=True)
        expected=reference((packed,scales))
        ring=[(packed,scales)]+[(packed.clone(),scales.clone()) for _ in range(7)]
        case={'projection':projection,'shape':saved['shape'],'paired':paired,'reference_us':measure(reference,ring),'trials':[]}
        del ring
        for scheme,w in (('prmt',packed),('interleaved',interleaved)):
            for scale_dtype,s in (('float32',scales),('bfloat16',bf16_scales)):
                ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
                for rows in (1,2,4):
                    for warps in (1,2,4,8):
                        fn=lambda pair:linear(x,*pair,rows=rows,warps=warps,interleaved=scheme=='interleaved',paired=paired)
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):actual=fn((w,s))
                        torch.cuda.current_stream().wait_stream(stream)
                        relative=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
                        assert relative<.001,(name,scheme,scale_dtype,rows,warps,relative)
                        trial={'scheme':scheme,'scale_dtype':scale_dtype,'rows':rows,'warps':warps,'us':measure(fn,ring),
                            'relative_rms_vs_reference':relative,'exact':torch.equal(actual,expected)}
                        case['trials'].append(trial)
                del ring
        case['best']=min(case['trials'],key=lambda t:t['us'])
        result['cases'].append(case)
        print(name,'BASELINE',case['reference_us'],'BEST',case['best'],flush=True)
        (RESULTS/'dp4a_packing_kernels.json').write_text(json.dumps(result,indent=2)+'\n')
        del saved,packed,scales,bf16_scales,interleaved,x,expected,actual


if __name__=='__main__':main()
