"""Compare complete attention-PDL candidates with qualified projection PDL control."""
import argparse
import hashlib
import json
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
from .dp4a_scaled import enable_scaled
from .dp4a_norm_projection import enable_norm_projection
from .benchmark_qkv_load_paired import capture
from . import short_scales as short_module


from .projection_pdl import enable as enable_projection_pdl
from .attention_pdl import enable as enable_attention_pdl
from .codec_clock import ClockedStreamingCodec


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--pairs',type=int,default=20);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.pairs<3:raise ValueError('Safe tag and three pairs required')
    path=RESULTS/f'codec_clock_paired_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True);fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10','dp4a')
    enable_grouped_activation(fast);enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json');enable_attention_quant(fast)
    enable_native_attention(fast);enable_gateup_quant(fast);enable_scaled(fast);enable_norm_projection(fast)
    short_module.enable(fast);enable_projection_pdl(fast);enable_attention_pdl(fast)
    engine.warmup((128,160,256,512));manager=capture(fast);fast.step=manager.step
    control=engine.codec
    # Capture owns every old state tensor through this wrapper's states/graph.
    # Detach the model's streaming context so the candidate gets independent
    # KV buffers, while both wrappers replay only their already captured graphs.
    control.stack.close()
    candidate=ClockedStreamingCodec(codec);candidate.warmup()
    codecs={'control':control,'candidate':candidate}
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);rows=[]
    for pair in range(-1,a.pairs):
        order=('control','candidate') if pair%2==0 else ('candidate','control');records={}
        for name in order:
            engine.codec=codecs[name];torch.manual_seed(7000+pair)
            chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400))
            pcm=torch.cat([c.pcm for c in chunks]);assert torch.isfinite(pcm).all()
            record=engine.last_metrics.copy();assert not record['truncated']
            record['pcm_float32_sha256']=hashlib.sha256(pcm.numpy().tobytes()).hexdigest();records[name]=record
        assert records['control']['pcm_float32_sha256']==records['candidate']['pcm_float32_sha256'],pair
        row={'pair':pair,'order':order,'seed':7000+pair,'records':records,
             'gain_ms':records['control']['ttfa_ms']-records['candidate']['ttfa_ms']}
        if pair>=0:rows.append(row)
        print('PAIR',pair,{n:r['ttfa_ms'] for n,r in records.items()},'GAIN',row['gain_ms'],flush=True)
    result={'codebooks':32,'torch':torch.__version__,'rows':rows,'all_pcm_exact':True,
            'method':'One shared selected G32/attention-PDL LLM and FP32 codec weights; separate retained control/candidate graphs and KV buffers. '
                     'Candidate uses four shared stage counters, reset-only authoritative counters, and an exact first-frame T=1 shortcut; other reductions keep full capacity. '
                     'Alternating order, one excluded warmup pair; complete cached-voice requests and float32 PCM, no registration/network. '
                     'No profiler initialized before timing.',
            'ttfa':{n:stats([r['records'][n]['ttfa_ms'] for r in rows]) for n in codecs},
            'median_paired_gain_ms':statistics.median(r['gain_ms'] for r in rows),'faster_pairs':sum(r['gain_ms']>0 for r in rows)}
    path.write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',result['median_paired_gain_ms'],result['faster_pairs'],flush=True)
    engine.codec=candidate
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=34))
    prof.export_chrome_trace(str(RESULTS/f'codec_clock_profile_{a.tag}_trace.json'))
    (RESULTS/f'codec_clock_profile_{a.tag}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    candidate.close();control.close()


if __name__=='__main__':main()
