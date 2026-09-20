"""Odd rows, alignment rejection and changing-input private CUDA graphs."""
import argparse
import json

import torch

from .common import RESULTS
from .benchmark_projection_pdl import load_layer
from .benchmark_norm_projection import flatten
from .dp4a_norm_pdl import linear as norm,SELECTED
from .dp4a_layout_pdl_prefetch import linear as projection
from .short_scales import PLAN
from .bulk_address import configured


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'bulk_address_validation_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    e=load_layer(17);checks=[];guards=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    probes=[{'divisor':16,'address_mode':1},{'divisor':16,'address_mode':2},{'divisor':1,'ahead':64,'address_mode':1}]
    with torch.cuda.stream(stream):
        for stage,counts in [('qkv',(1,7,9,6144)),('up',(32,64,12288)),('down',(1,3,5,4096))]:
            for count in counts:
                if stage in ('qkv','up'):
                    d=e['next'] if stage=='qkv' else e['raw'];x=d['x'][0:1].clone();res=d['residual'][0:1].clone() if d['has_residual'] else None
                    w,s=e['weights'][stage]
                    if stage=='up':
                        half=w.shape[0]//2;w=torch.cat((w[:count],w[half:half+count]));s=torch.cat((s[:count],s[half:half+count]))
                    else:w=w[:count];s=s[:count]
                    args=(x,res,d['weight'],d['eps'],w,s);options=dict(SELECTED[stage],fused=stage=='up')
                    reference=norm;kind='norm'
                else:
                    d=e['raw'];r=norm(d['x'][0:1],d['residual'][0:1] if d['has_residual'] else None,d['weight'],d['eps'],*e['weights']['up'],fused=True,**SELECTED['up'])
                    _,x,quantized=r;x=x.clone();w,s=e['weights']['down'];w=w[:count];s=s[:count]
                    args=(x,w,s);options=dict(PLAN['down'],prequantized=quantized,trigger_mode=3,prefetch=1)
                    reference=projection;kind='projection'
                for config in probes:
                    candidate=configured(kind,**config)
                    for _ in range(2):candidate(*args,**options)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):actual=candidate(*args,**options)
                    for input_index in (10,31):
                        if stage in ('qkv','up'):
                            x.copy_(d['x'][input_index:input_index+1])
                            if res is not None:res.copy_(d['residual'][input_index:input_index+1])
                        else:
                            quantized[0].copy_(torch.randint(-127,128,quantized[0].shape,device='cuda',dtype=torch.int8))
                            quantized[1].fill_(0.0125 if input_index==10 else 0.03125)
                        graph.replay();expected=reference(*args,**options);stream.synchronize()
                        exact=all(torch.equal(u,v) for u,v in zip(flatten(actual),flatten(expected),strict=True));assert exact,(stage,count,config,input_index)
                        checks.append({'stage':stage,'rows':count,'config':config,'input':input_index,'exact':True})
                # A contiguous sliced allocation can still violate PTX alignment.
                bad=torch.empty(w.numel()+1,device='cuda',dtype=w.dtype)[1:].view_as(w);bad.copy_(w)
                bad_args=list(args);bad_args[4 if kind=='norm' else 1]=bad
                try:configured(kind)(*bad_args,**options)
                except ValueError as error:assert 'aligned' in str(error)
                else:raise AssertionError('Misaligned weights accepted')
                guards.append({'stage':stage,'rows':count,'misaligned_weight_rejected':True})
                print('CHECKED',stage,count,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'all_exact':True,'checks':checks,'guards':guards,'cases':len(checks),
        'scope':'Changed-input private CUDA graphs, actual/odd rows for all three projection stages, full/prefix prefetch, cache policy and wrapped lookahead. Output and every returned intermediate compare exactly with original PDL bodies; misaligned contiguous weight buffers reject before launch.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(checks),flush=True)


if __name__=='__main__':main()
