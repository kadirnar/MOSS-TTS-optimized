"""Alternate baseline and QKV-load-policy graphs, preserving all model weights."""
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
from .decode_buckets import DecodeContextBuckets
from .audio_head_buckets import DecodeAudioHeadBuckets


def capture(fast):
    contexts=DecodeContextBuckets(fast);contexts.warmup()
    heads=DecodeAudioHeadBuckets(fast,contexts);heads.warmup()
    return heads


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--pairs',type=int,default=10);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.pairs<2:raise ValueError('Safe tag and at least two pairs required')
    path=RESULTS/f'qkv_load_paired_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True);fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10','dp4a')
    enable_grouped_activation(fast);enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json');enable_attention_quant(fast)
    enable_native_attention(fast);enable_gateup_quant(fast);enable_scaled(fast);enable_norm_projection(fast)
    engine.warmup((128,160,256,512))
    managers={'control':capture(fast)}
    fast.qkv_load_policy=True
    fast.warmup();managers['candidate']=capture(fast)
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    ids=fixture['inputs']['input_ids'].cuda()
    fast.prefill(ids.repeat(1,8,1)[:,:1023])
    token=fixture['upstream_ids'][10:11][None].cuda();checks=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pos in (0,1,31,32,126,127,128,254,255,256,510,511,512,1023,144,127):
            for length in (1,9,32):
                torch.cuda.manual_seed(3927)
                expected_ids=managers['control'].step(token,pos,length,-1).clone()
                expected=(expected_ids,fast.text_logits.clone(),fast.audio_logits.clone())
                torch.cuda.manual_seed(3927)
                actual_ids=managers['candidate'].step(token,pos,length,-1)
                actual=(actual_ids,fast.text_logits,fast.audio_logits)
                assert all(torch.equal(x,y) for x,y in zip(actual,expected,strict=True)),(pos,length)
                checks.append({'position':pos,'length':length,'all_logits_and_ids_exact':True})
    torch.cuda.current_stream().wait_stream(stream)
    # Inspect the complete repeated decode graph independently of Python
    # preparation, PCM copying and request scheduling.
    fast.ids.copy_(token);fast.position.fill_(155);fast.audio_length.fill_(11);fast.delay_length.fill_(-1)
    graph_rows=[]
    for repeat in range(12):
        order=('control','candidate') if repeat%2==0 else ('candidate','control')
        timing={}
        for name in order:
            graph=managers[name].graphs[256,32][0]
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100):graph.replay()
            end.record();end.synchronize();timing[name]=begin.elapsed_time(end)/100
        graph_rows.append({'round':repeat,'order':order,'gpu_ms':timing})
    rows=[]
    for pair in range(-1,a.pairs):
        order=('control','candidate') if pair%2==0 else ('candidate','control')
        records={}
        for name in order:
            fast.qkv_load_policy=name=='candidate';fast.step=managers[name].step
            torch.manual_seed(7000+pair)
            chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400))
            pcm=torch.cat([c.pcm for c in chunks]);assert torch.isfinite(pcm).all()
            record=engine.last_metrics.copy();assert not record['truncated']
            record['pcm_float32_sha256']=hashlib.sha256(pcm.numpy().tobytes()).hexdigest()
            records[name]=record
        assert records['control']['pcm_float32_sha256']==records['candidate']['pcm_float32_sha256'],pair
        row={'pair':pair,'seed':7000+pair,'order':order,'records':records,
             'gain_ms':records['control']['ttfa_ms']-records['candidate']['ttfa_ms']}
        print({'pair':pair,'control_ms':records['control']['ttfa_ms'],'candidate_ms':records['candidate']['ttfa_ms'],'gain_ms':row['gain_ms'],'pcm_exact':True},flush=True)
        if pair>=0:rows.append(row)
    result={'codebooks':32,'torch':torch.__version__,'rows':rows,
            'method':'One shared model/process; separate complete graphs with identical head/context buckets, reversed order each pair and one excluded warmup pair. QKV load flag restored for eager steps as well as graph dispatch. Full cached-voice streaming generation; network and reference encoding excluded.',
            'all_pcm_exact':True,'private_stream_graph_checks':checks,
            'repeated_decode_graphs':graph_rows,
            'median_paired_gain_ms':statistics.median(r['gain_ms'] for r in rows),
            'improved_pairs':sum(r['gain_ms']>0 for r in rows)}
    for name in managers:result[name+'_ttfa']=stats([r['records'][name]['ttfa_ms'] for r in rows])
    path.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('rows','private_stream_graph_checks')},flush=True)
    fast.qkv_load_policy=True;fast.step=managers['candidate'].step
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=34))
    (RESULTS/f'qkv_load_profile_{a.tag}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    engine.codec.close()


if __name__=='__main__':main()
