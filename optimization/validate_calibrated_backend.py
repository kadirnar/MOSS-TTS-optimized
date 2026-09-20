"""Check real calibrated packing, native Marlin, and DP4A on a private stream."""
import argparse
import json
from pathlib import Path
import torch
from .common import RESULTS
from .calibrated_backend import unpack_signed
from .calibrated_quant import dequantize
from .marlin import MarlinLinear
from .int4_dp4a import int4_dp4a


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibration',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    manifest=json.loads((RESULTS/'calibration_v1/manifest.json').read_text())
    # First row of a held-out calibration utterance, not training data.
    offset=0
    for record in manifest['records']:
        if record['id'].endswith('_2'):break
        offset+=record['decode_tokens']
    cases=[]
    for layer in (0,17,35):
        for projection in ('qkv','out','up','down'):
            name=f'{layer:02d}_{projection}'
            saved=torch.load(args.calibration/(name+'.pt'),map_location='cuda',weights_only=True)
            packed,scales,group=saved['packed'],saved['scales'],saved['group']
            signed=unpack_signed(packed)
            unsigned=signed.to(torch.uint8)&15
            repacked=unsigned[:,::2]|(unsigned[:,1::2]<<4)
            assert torch.equal(packed,repacked),name
            w=dequantize(signed,scales,group)
            x=torch.load(RESULTS/'calibration_v1'/(name+'.pt'),weights_only=True)[offset:offset+1].cuda()
            marlin=MarlinLinear(w,4,group,signed=signed,scales=scales)
            expected=torch.nn.functional.linear(x,w)
            sx=(x.float().abs().max()/127).clamp_min(1e-8)
            q=(x.float()/sx).round().clamp(-127,127)
            # DP4A keeps weight scale products in FP32, unlike Marlin BF16.
            wf=(signed.float().view(signed.shape[0],-1,group)*scales.float()[:,:,None]).reshape_as(w)
            dp4a_expected=torch.nn.functional.linear(q*sx,wf).bfloat16()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual=marlin(x)
                dp4a_actual=int4_dp4a(x,packed,scales.float(),group)
            torch.cuda.current_stream().wait_stream(stream)
            def relative(a,b):
                return ((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt()).item()
            row={'projection':name,'shape':saved['shape'],
                 'marlin_relative_rms':relative(actual,expected),
                 'dp4a_relative_rms':relative(dp4a_actual,dp4a_expected)}
            assert row['marlin_relative_rms']<.005,row
            assert row['dp4a_relative_rms']<.001,row
            cases.append(row)
            print(row,flush=True)
            del saved,packed,scales,signed,unsigned,repacked,w,wf,x,marlin,actual,expected,dp4a_actual,dp4a_expected
    result={'calibration':str(args.calibration),'torch':torch.__version__,
            'private_stream':True,'nibble_roundtrip':'exact',
            'scope':'Kernel/packing correctness against explicitly dequantized weights, not quality acceptance.',
            'cases':cases}
    (RESULTS/'gptq_backend_validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
