"""Changed-input private graph checks for async copies and PDL timing probes."""
import argparse
import json

import torch

from .common import RESULTS
from .benchmark_async_weights import Chain
from .benchmark_projection_timestamps import Chain as Probe, summarize
from .benchmark_projection_pdl import load_layer
from .benchmark_norm_projection import flatten
from .benchmark_group128 import quantize
from .dp4a_packing import pack_interleaved
from .dp4a_async_weights import configured
from .dp4a_layout_pdl_prefetch import linear as reference
from .short_scales import PLAN


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--out-rows',type=int,choices=(8,16),default=16)
    p.add_argument('--output-only',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'async_weights_validation_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    variants={
        'copy_sw8_hint':Chain({'mode':1,'swizzle':8,'divisor':16}),
        'tma_sw128':Chain({'mode':2,'swizzle':128}),
        'tma_compact':Chain({'mode':3,'swizzle':0}),
        'tma_partial':Chain({'mode':4,'swizzle':0}),
        'tma_partial_r8w4':Chain({'mode':4,'swizzle':0,'tile':{'rows':8,'warps':4}}),
        'probe_up_wait':Probe(1,compact=True,families=['up']),
        'probe_qkv_full':Probe(2,compact=True,families=['qkv']),
        'probe_down_wait':Probe(1,compact=True,families=['down']),
    }
    control=Chain()
    def replay_case(fn,ref,x,res,label,probe=None):
        original=x.clone();saved=None if res is None else res.clone()
        for _ in range(2):fn()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):actual=fn()
        for kind in ('real','zero','spike'):
            x.copy_(original) if kind=='real' else x.zero_()
            if res is not None:res.copy_(saved) if kind=='real' else res.zero_()
            if kind=='spike':x.reshape(-1)[-1]=3
            if probe is not None:
                for t in probe.traces.values():t.zero_()
            graph.replay();expected=ref()
            exact=all(torch.equal(u.view(torch.uint8),v.view(torch.uint8)) for u,v in zip(flatten(actual),flatten(expected),strict=True))
            row={**label,'input':kind,'selected_control_bits_exact':exact};assert exact,row
            if probe is not None:
                # Other layers' trace buffers were cleared but not replayed.
                current={k:v for k,v in probe.traces.items() if k[0]==label['layer']}
                summarize(current,probe.mode,probe.stride)
                row['sampled_timestamps_ordered']=True
            checks.append(row)
        x.copy_(original)
        if res is not None:res.copy_(saved)
    with torch.cuda.stream(stream):
        for index in (0,17,35):
            e=load_layer(index);d=e['raw'];x=d['x'][0];res=d['residual'][0] if d['has_residual'] else None
            for name,candidate in ({} if a.output_only else variants).items():
                replay_case(lambda:candidate(e),lambda:control(e),x,res,{'layer':index,'config':name},candidate if isinstance(candidate,Probe) else None)
                print('CHECKED',index,name,flush=True)
            saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{index:02d}_out.pt',map_location='cuda',weights_only=True)
            packed=pack_interleaved(saved['packed']);scales=saved['scales'].bfloat16()
            x=torch.load(RESULTS/f'calibration_v1/{index:02d}_out.pt',weights_only=True)[231:232].cuda()
            for n in (4096,4093):
                w=packed[:n];s=scales[:n]
                def ref():return reference(x,w,s,prequantized=quantize(x,32),**PLAN['out'],trigger_mode=3,prefetch=1)
                for name,options in {'copy_cg':{'mode':1,'swizzle':1,'cache':1},'copy_sw8':{'mode':1,'swizzle':8},
                                     'tma':{'mode':2,'swizzle':0},'tma_sw128':{'mode':2,'swizzle':128}}.items():
                    candidate=configured(**options)
                    def fn():return candidate(x,w,s,prequantized=quantize(x,32),**{**PLAN['out'],'rows':a.out_rows},trigger_mode=3,prefetch=1)
                    replay_case(fn,ref,x,None,{'layer':index,'config':'out_'+name,'rows':n,'tile_rows':a.out_rows})
                    print('CHECKED',index,'out',name,n,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'cases':len(checks),'output_tile_rows':a.out_rows,'output_only':a.output_only,'all_selected_control_bits_exact':True,'checks':checks,
        'scope':'Private CUDA stream, changing real/zero/spike inputs. Five async down chains, three compact diagnostic probe schedules, and four output-copy variants including odd output-row tails. Copies preserve selected G32 arithmetic. Projection/chain memory validation, not whole-model sanitizer coverage.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(checks),flush=True)


if __name__=='__main__':main()
