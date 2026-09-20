"""SM90 instruction/resource and graph-timeline audit for projection PDL."""
import argparse
import hashlib
import json
import re
import subprocess

import torch

from .common import RESULTS
from .benchmark_projection_pdl import load_layer, chain, graph_edges
from .benchmark_group128 import quantize
from .dp4a_norm_pdl import linear as norm
from .dp4a_norm_pdl_prefetch import linear as norm_preload
from .dp4a_layout_pdl import linear as projection
from .dp4a_layout_pdl_prefetch import linear as projection_preload
from .dp4a_norm_projection import SELECTED
from .short_scales import PLAN
from .dp4a_packing import pack_interleaved


def save_kernel(folder,name,kernel):
    for ext in ('ptx','ttgir'):(folder/(name+'.'+ext)).write_text(kernel.asm[ext])
    cubin=kernel.asm['cubin'];path=folder/(name+'.cubin');path.write_bytes(cubin)
    sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(path)],text=True)
    (folder/(name+'.sass')).write_text(sass)
    instructions=[]
    for line in sass.splitlines():
        m=re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)',line)
        if m:instructions.append(m.group(1))
    # cuobjdump names the SM90 lowering ACQBULK/PREEXIT, not the PTX mnemonic.
    first_wait=next((i for i,v in enumerate(instructions) if v=='ACQBULK' or v.startswith('GRIDDEPCONTROL') and 'WAIT' in v),None)
    return {'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared,
            'cubin_sha256':hashlib.sha256(cubin).hexdigest(),'static_instructions':len(instructions),
            'static_opcodes':{v:instructions.count(v) for v in sorted(set(instructions))},
            'first_wait_index':first_wait,
            'global_loads_before_wait':None if first_wait is None else sum(v.startswith('LDG') for v in instructions[:first_wait])}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    folder=RESULTS/f'projection_pdl_audit_{a.tag}';folder.mkdir(exist_ok=False)
    torch.set_num_threads(4);entry=load_layer(17);resources={}
    for name in ('up','qkv'):
        d=entry['raw'] if name=='up' else entry['next']
        x=d['x'][:1];res=d['residual'][:1] if d['has_residual'] else None
        for variant,fn,options in (
                ('control',norm,{'pdl':False}),('pdl',norm,{'pdl':True}),
                ('all_preload',norm_preload,{'pdl':True,'prefetch':3})):
            _,kernel=fn(x,res,d['weight'],d['eps'],*entry['weights'][name],
                        fused=name=='up',**SELECTED[name],**options,return_kernel=True)
            resources[name+'_'+variant]=save_kernel(folder,name+'_'+variant,kernel)
    for name,k in (('out',4096),('down',12288)):
        w=torch.load(RESULTS/f'gptq_v1_g32_d10/17_{name}.pt',map_location='cuda',weights_only=True)
        packed=pack_interleaved(w['packed']);scale=w['scales'].bfloat16()
        x=torch.load(RESULTS/f'calibration_v1/17_{name}.pt',weights_only=True)[231:232].cuda();q=quantize(x,32)
        for variant,fn,options in (
                ('control',projection,{'pdl':False}),('pdl',projection,{'pdl':True}),
                ('scale_preload',projection_preload,{'pdl':True,'prefetch':1})):
            _,kernel=fn(x,packed,scale,prequantized=q,**PLAN[name],**options,
                        trigger_mode=3,return_kernel=True)
            resources[name+'_'+variant]=save_kernel(folder,name+'_'+variant,kernel)
    timelines={}
    for name,config in (('control',None),('pdl',{'pdl':True,'norm_trigger':1,'projection_trigger':3}),
                        ('prefetch',{'pdl':True,'norm_trigger':1,'projection_trigger':3,'norm_prefetch':0,'projection_prefetch':1})):
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
                         'edges':graph_edges(graph),
                         'events':[{'name':e['name'],'ts':e['ts'],'dur':e['dur']} for e in events]}
    result={'codebooks':32,'resources':resources,'timelines':timelines,
            'scope':'Static SM90 code plus three profiled graph replays of one actual layer chain. '
                    'Kernel intervals include dependency waits; interval overlap is not simultaneous useful work '
                    'or utilization. Profiling is excluded from latency measurements.'}
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('SAVED',folder,flush=True)


if __name__=='__main__':main()
