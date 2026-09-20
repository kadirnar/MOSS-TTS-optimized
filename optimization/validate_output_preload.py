"""Private graphs with changing inputs and odd row tails for register preloads."""
import argparse
import json

import torch
from .common import RESULTS
from .benchmark_group128 import quantize
from .dp4a_packing import pack_interleaved
from .dp4a_layout_pdl_prefetch import linear
from .short_scales import PLAN


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'output_preload_validation_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for index in (0,17,35):
            saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{index:02d}_out.pt',map_location='cuda',weights_only=True)
            packed=pack_interleaved(saved['packed']);scales=saved['scales'].bfloat16()
            x=torch.load(RESULTS/f'calibration_v1/{index:02d}_out.pt',weights_only=True)[231:232].cuda();original=x.clone()
            for n in (4096,4093):
                w=packed[:n];s=scales[:n]
                def invoke(rows,prefetch):
                    return linear(x,w,s,prequantized=quantize(x,32),**{**PLAN['out'],'rows':rows},trigger_mode=3,prefetch=prefetch)
                for rows in (8,16):
                    for _ in range(2):invoke(rows,3)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):actual=invoke(rows,3)
                    for kind in ('real','zero','spike'):
                        x.copy_(original) if kind=='real' else x.zero_()
                        if kind=='spike':x.reshape(-1)[-1]=3
                        graph.replay();expected=invoke(4,1)
                        exact=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
                        checks.append({'layer':index,'output_rows':n,'tile_rows':rows,'input':kind,'selected_control_bits_exact':exact});assert exact,checks[-1]
                    x.copy_(original);print('CHECKED',index,n,rows,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'cases':len(checks),'all_selected_control_bits_exact':True,'checks':checks,
        'scope':'Actual G32 output weights, eight/sixteen-row register preload before original PDL wait, private graphs with changing real/zero/spike inputs and odd row tails. Operator memory validation, not whole-service sanitization.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(checks),flush=True)


if __name__=='__main__':main()
