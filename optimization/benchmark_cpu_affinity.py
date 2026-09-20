"""Same-model NUMA placement experiment; changes only this calling thread.

No host topology, global policy, memory binding or existing service is changed.
The original thread affinity is restored even on failure.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import torch
from .common import RESULTS,load_models,stats
from .streaming import StreamingTTS
from .calibrated_backend import install_calibrated,enable_grouped_activation
from .dp4a_fusions import enable_fusions
from .dp4a_gateup import enable_gateup
from .dp4a_packing import install_packing
from .attention_quant import enable_attention_quant
from .attention_native import enable_native_attention
from .dp4a_gateup_quant import enable_gateup_quant
from .decode_buckets import DecodeContextBuckets


def cpulist(value):
    out=set()
    for part in value.strip().split(','):
        bounds=part.split('-');a=int(bounds[0]);b=int(bounds[-1])
        if not 0<=a<=b:raise ValueError('Invalid CPU range')
        out.update(range(a,b+1))
    return out


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=5);p.add_argument('--tag',required=True);p.add_argument('--pci-device',default='0000:b8:00.0');args=p.parse_args()
    if args.rounds<2 or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag and at least two rounds required')
    original=os.sched_getaffinity(0);config={'unbound':original}
    for node in sorted(Path('/sys/devices/system/node').glob('node[0-9]*')):
        allowed=cpulist((node/'cpulist').read_text())&original
        if allowed:config[node.name]=allowed
    gpu=Path('/sys/bus/pci/devices')/args.pci_device
    gpu_node=int((gpu/'numa_node').read_text());local='node'+str(gpu_node)
    if local not in config:raise ValueError('GPU-local CPU node unavailable')
    libc=ctypes.CDLL(None);libc.sched_getcpu.restype=ctypes.c_int
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True);fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10',backend='dp4a');enable_grouped_activation(fast)
    enable_fusions(fast,layout_limit=8);enable_gateup(fast);install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json')
    enable_attention_quant(fast);enable_native_attention(fast);enable_gateup_quant(fast)
    engine.warmup((128,160,256,512));manager=DecodeContextBuckets(fast);manager.warmup();manager.install()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);names=list(config);records=[]
    try:
        for round_id in range(-1,args.rounds):
            shift=round_id%len(names);order=names[shift:]+names[:shift];reference_hash=None
            for name in order:
                os.sched_setaffinity(0,config[name]);before=libc.sched_getcpu()
                assert before in config[name]
                torch.manual_seed(8000+round_id)
                chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400));pcm=torch.cat([c.pcm for c in chunks])
                metrics=engine.last_metrics.copy();assert not metrics['truncated'] and torch.isfinite(pcm).all()
                after=libc.sched_getcpu();assert after in config[name]
                digest=hashlib.sha256(pcm.numpy().tobytes()).hexdigest()
                if reference_hash is None:reference_hash=digest
                assert digest==reference_hash,(round_id,name)
                row={'round':round_id,'mode':name,'cpu_before':before,'cpu_after':after,'pcm_float32_sha256':digest,'metrics':metrics}
                if round_id>=0:records.append(row)
                print({'round':round_id,'mode':name,'cpu_before':before,'cpu_after':after,'ttfa_ms':metrics['ttfa_ms'],'pcm_exact':True},flush=True)
    finally:
        os.sched_setaffinity(0,original);engine.codec.close()
    result={'gpu_pci':args.pci_device,'gpu_numa_node':gpu_node,'affinities':{k:sorted(v) for k,v in config.items()},'records':records,'all_same_seed_pcm_exact':True,'original_affinity_restored':os.sched_getaffinity(0)==original,
        'method':'One model and identical graph/weight addresses. Only calling-thread affinity changes after setup; other existing threads are unbound. Rotate order by one each round, identical seed/text/reference within each round, one warmup round excluded. Cached voice, all 32 codebooks, native attention and fused gate/up quantizer. No memory binding or host changes.','summary':{}}
    for name in names:
        selected=[r['metrics'] for r in records if r['mode']==name]
        result['summary'][name]={'ttfa':stats([r['ttfa_ms'] for r in selected]),'prepare_median_ms':statistics.median(r['prepare_ms'] for r in selected),'prefill_median_ms':statistics.median(r['prefill_ms'] for r in selected),'initial_decode_median_ms':statistics.median(statistics.median(r['step_ms'][:32]) for r in selected),'first_codec_median_ms':statistics.median(r['codec_ms'][0] for r in selected)}
    (RESULTS/f'cpu_affinity_{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':main()
