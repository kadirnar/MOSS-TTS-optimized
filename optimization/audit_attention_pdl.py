"""Inspect SM90 instructions, register use and dependency interval overlap."""
import argparse
import json
import subprocess

import torch
from .common import RESULTS
from .benchmark_attention_pdl import load_layer,chain,configurations
from .benchmark_projection_pdl import graph_edges
from .audit_projection_pdl import save_kernel
from .qk_rope_pdl import launch as rope
from .kernels import _qk_rope_cache
from .attention_quant import _reduce_quant
from .attention_quant_pdl import reduce_quant
from .attention_pdl import library


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    folder=RESULTS/f'attention_pdl_audit_{a.tag}';folder.mkdir(exist_ok=False)
    torch.set_num_threads(4);entry=load_layer(17);d=entry['attention'];resources={}
    q=torch.empty_like(d['q']);args=(d['qkv'],d['qw'],d['kw'],d['cos'],d['sin'],q,d['k'],d['v'],d['position'])
    k=_qk_rope_cache[(40,)](*args,d['k'].shape[-2],d['eps'],enable_fp_fusion=False)
    resources['rope_control']=save_kernel(folder,'rope_control',k)
    for name,options in (('late',{'trigger':1}),('preload',{'trigger':2,'preload':True})):
        k=rope(*args,d['eps'],**options);resources['rope_'+name]=save_kernel(folder,'rope_'+name,k)
    for capacity in (128,256,512,1024):
        splits=capacity//32
        # Fresh complete chain populates the exact capacity-sized partials.
        entry['capacity']=capacity;d['position'].fill_(min(155,capacity-1))
        _,_,_,part,lse,_,_,_=chain(entry,debug=True)
        (out,(q,scale)),k=reduce_quant(part,lse,return_kernel=True)
        resources[f'reduce_pdl_{capacity}']=save_kernel(folder,f'reduce_pdl_{capacity}',k)
        k=_reduce_quant[(32,)](part,lse,out,q,scale,splits,splits,num_warps=4)
        resources[f'reduce_control_{capacity}']=save_kernel(folder,f'reduce_control_{capacity}',k)
    lib=library()
    (folder/'native_attention.sass').write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',lib._name],text=True))
    (folder/'native_attention.resources').write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',lib._name],text=True))
    timelines={};entry['capacity']=256;d['position'].fill_(155)
    for name,config in ((k,configurations()[k]) for k in ('control','q1_a2_r1','pre_q2_a1_r1')):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):chain(entry,config)
            graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph,stream=stream):result=chain(entry,config)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):graph.replay()
            torch.cuda.synchronize()
        trace=folder/(name+'_trace.json');prof.export_chrome_trace(str(trace))
        events=[e for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
        intervals=sorted((float(e['ts']),float(e['ts']+e['dur'])) for e in events)
        union=0.;end=-float('inf')
        for start,stop in intervals:
            union+=max(0.,stop-max(start,end));end=max(end,stop)
        total=sum(stop-start for start,stop in intervals)
        timelines[name]={'kernel_events':len(events),'summed_kernel_duration_us':total,
                         'union_kernel_intervals_us':union,'overlapped_interval_us':total-union,
                         'edges':graph_edges(graph),'events':[{'name':e['name'],'ts':e['ts'],'dur':e['dur']} for e in events]}
    result={'codebooks':32,'resources':resources,'timelines':timelines,
            'native_library':lib._name,
            'scope':'Static SM90 code and three profiled five-kernel graph replays per option. '
                    'Intervals include PDL waits and can overlap; summed durations double-count time and are not utilization or critical-path percentages. '
                    'Only immutable Q/K normalization weights may be loaded before the QK wait. '
                    'Profiling is excluded from request and operator timings.'}
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('SAVED',folder,flush=True)


if __name__=='__main__':main()
