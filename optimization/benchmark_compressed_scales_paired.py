"""Same-process streaming comparison of lossless FP32 scale allocations."""
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
from .compressed_alloc import clone,library


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=15);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<3:raise ValueError('Safe tag and at least three rounds required')
    path=RESULTS/f'compressed_scales_paired_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True);fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10','dp4a')
    enable_grouped_activation(fast);enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json');enable_attention_quant(fast)
    enable_native_attention(fast);enable_gateup_quant(fast);enable_scaled(fast);enable_norm_projection(fast)
    swaps=[];allocations=[];names=('torch','vmm_plain','compressed')
    for layer_index,layer in enumerate(model.language_model.layers):
        for module,key in ((layer.self_attn,'_quant_out_scale'),(layer.mlp,'_quant_down_scale')):
            original=getattr(module,key);assert original.dtype==torch.float32
            values={'torch':original}
            for name in names[1:]:
                values[name],meta=clone(original,compressed=name=='compressed')
                assert torch.equal(values[name],original)
                allocations.append({'layer':layer_index,'buffer':key,'variant':name,**meta})
            swaps.append((module,key,values))
    managers={}
    def select(name):
        for module,key,values in swaps:setattr(module,key,values[name])
        if name in managers:fast.step=managers[name].step
    engine.warmup((128,160,256,512));managers['torch']=capture(fast)
    for name in names[1:]:
        select(name);fast.warmup();managers[name]=capture(fast)
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    ids=fixture['inputs']['input_ids'].cuda();select('torch')
    fast.prefill(ids.repeat(1,8,1)[:,:1023]);token=fixture['upstream_ids'][10:11][None].cuda()
    checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pos in (0,1,31,32,126,127,128,254,255,256,510,511,512,1023,144,127):
            for length in (1,9,32):
                torch.cuda.manual_seed(3927)
                expected_ids=managers['torch'].step(token,pos,length,-1).clone()
                expected=(expected_ids,fast.text_logits.clone(),fast.audio_logits.clone())
                for name in names[1:]:
                    torch.cuda.manual_seed(3927)
                    actual_ids=managers[name].step(token,pos,length,-1)
                    actual=(actual_ids,fast.text_logits,fast.audio_logits)
                    assert all(torch.equal(x,y) for x,y in zip(actual,expected,strict=True)),(name,pos,length)
                    checks.append({'variant':name,'position':pos,'length':length,'logits_and_ids_exact':True})
    torch.cuda.current_stream().wait_stream(stream)
    fast.ids.copy_(token);fast.position.fill_(155);fast.audio_length.fill_(11);fast.delay_length.fill_(-1)
    graph_rows=[]
    for repeat in range(12):
        order=list(names);order=order[repeat%3:]+order[:repeat%3]
        if repeat%2:order.reverse()
        timing={}
        for name in order:
            graph=managers[name].graphs[256,32][0]
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):graph.replay()
            end.record();end.synchronize();timing[name]=start.elapsed_time(end)/100
        graph_rows.append({'round':repeat,'order':order,'gpu_ms':timing})
    rows=[]
    for repeat in range(-1,a.rounds):
        order=list(names);order=order[repeat%3:]+order[:repeat%3]
        if repeat%2:order.reverse()
        records={}
        for name in order:
            select(name);torch.manual_seed(7000+repeat)
            chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400))
            pcm=torch.cat([c.pcm for c in chunks]);assert torch.isfinite(pcm).all()
            record=engine.last_metrics.copy();assert not record['truncated']
            record['pcm_float32_sha256']=hashlib.sha256(pcm.numpy().tobytes()).hexdigest();records[name]=record
        assert all(r['pcm_float32_sha256']==records['torch']['pcm_float32_sha256'] for r in records.values())
        gains={name:records['torch']['ttfa_ms']-records[name]['ttfa_ms'] for name in names[1:]}
        row={'round':repeat,'order':order,'seed':7000+repeat,'records':records,'gains_ms':gains}
        if repeat>=0:rows.append(row)
        print('ROUND',repeat,{name:r['ttfa_ms'] for name,r in records.items()},'GAINS',gains,flush=True)
    result={'codebooks':32,'torch':torch.__version__,'rows':rows,'graph_rows':graph_rows,'checks':checks,
            'allocations':allocations,'allocator_live':library().counters(),'all_pcm_exact':True,
            'method':'Shared calibrated G32 model, BF16 prefill, FP32 codec, original head/context buckets. Only 72 FP32 out/down scale buffers reallocated; plain VMM controls mapping/alignment overhead. Separate graph sets retain every allocation owner. Eager buffers and graph dispatch both restored per variant. Rotating/reversed order and one excluded warmup triplet. Cached-voice complete streaming PCM; no reference encoding/network.',
            'ttfa':{name:stats([r['records'][name]['ttfa_ms'] for r in rows]) for name in names},
            'median_paired_gain_ms':{name:statistics.median(r['gains_ms'][name] for r in rows) for name in names[1:]},
            'faster_rounds':{name:sum(r['gains_ms'][name]>0 for r in rows) for name in names[1:]}}
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY',result['median_paired_gain_ms'],result['faster_rounds'],flush=True)
    select('compressed')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=34))
    (RESULTS/f'compressed_scales_profile_{a.tag}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    engine.codec.close()


if __name__=='__main__':main()
