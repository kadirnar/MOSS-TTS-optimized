"""Check paired DP4A against the separately validated GEMV and SiLU kernels."""
import argparse
import json
import torch
from .common import RESULTS
from .int4_dp4a import int4_dp4a
from .kernels import silu_mul
from .dp4a_gateup import gateup
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibration',default='gptq_v1_g32_d10')
    args=parser.parse_args()
    torch.set_num_threads(4)
    cases=[]
    for layer in (0,17,35):
        name=f'{layer:02d}_up'
        saved=torch.load(RESULTS/args.calibration/(name+'.pt'),map_location='cuda',weights_only=True)
        packed,scales=saved['packed'],saved['scales'].float()
        samples=torch.load(RESULTS/'calibration_v1'/(name+'.pt'),weights_only=True)
        x=samples[231:232].cuda()  # start of first held-out utterance
        matrices=[packed]+[packed.clone() for _ in range(7)]
        reference=lambda w:silu_mul(int4_dp4a(x,w,scales,32,1,True,grouped_activation=True))
        expected=reference(packed)
        trials=[]
        for warps in (1,2,4,8):
            fn=lambda w:gateup(x,w,scales,32,warps)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):actual=fn(packed)
            torch.cuda.current_stream().wait_stream(stream)
            rel=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
            assert rel<.002,(name,warps,rel)
            trials.append({'warps':warps,'us':measure(fn,matrices),'relative_rms':rel,'exact':torch.equal(actual,expected)})
        row={'layer':layer,'reference_us':measure(reference,matrices),'trials':trials}
        cases.append(row);print(row,flush=True)
        del matrices,packed,scales,samples,x,saved,expected,actual
    (RESULTS/'dp4a_gateup_kernels.json').write_text(json.dumps({'torch':torch.__version__,
        'method':'Eight distinct packed matrices exceed L2; grouped quantizer included; real GPTQ weights and held-out activations. Private CUDA stream checks.',
        'cases':cases},indent=2)+'\n')


if __name__=='__main__':main()
