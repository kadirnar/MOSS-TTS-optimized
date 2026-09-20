"""Alternate G32 and G128 residual projections without reloading BF16 weights.

Keep both sets of quantized buffers alive for their captured graphs. Restore
module buffers as well as graph dispatch, so eager pre-audio steps also use the
intended weights. This compares different quantized models: PCM need not match.
"""
import argparse
import hashlib
import io
import json
import statistics

import soundfile as sf
import torch

from .common import RESULTS, load_models, stats
from .streaming import StreamingTTS
from .calibrated_backend import install_calibrated, enable_grouped_activation
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
from .dp4a_group128_residual import install as install_residual


def keep_buffers(fast):
    return [(module,name,getattr(module,'_quant_'+name),
             getattr(module,'_quant_'+name+'_scale'),getattr(module,'_group128_plan',{}))
            for layer in fast.model.language_model.layers
            for module,name in ((layer.self_attn,'out'),(layer.mlp,'down'))]


def restore_buffers(state):
    for module,name,weight,scale,plan in state:
        setattr(module,'_quant_'+name,weight)
        setattr(module,'_quant_'+name+'_scale',scale)
        module._group128_plan=plan


def capture(fast):
    contexts=DecodeContextBuckets(fast);contexts.warmup()
    heads=DecodeAudioHeadBuckets(fast,contexts);heads.warmup()
    return heads


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tag',required=True);p.add_argument('--pairs',type=int,default=10)
    a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.pairs<2:
        raise ValueError('Safe tag and at least two pairs required')
    path=RESULTS/f'group128_residual_paired_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve previous results')
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True)
    fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10','dp4a')
    enable_grouped_activation(fast);enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json')
    enable_attention_quant(fast);enable_native_attention(fast)
    enable_gateup_quant(fast);enable_scaled(fast);enable_norm_projection(fast)
    engine.warmup((128,160,256,512))
    managers={'control':capture(fast)}
    buffers={'control':keep_buffers(fast)}
    prefill=fast.prefill_graphs
    fast.graph=None;fast.prefill_graphs={}
    config=install_residual(fast,RESULTS/'gptq_v1_g128_diag_d10',
                            RESULTS/'group128_a32_fused_plan_v2.json')
    fast.prefill_graphs=prefill
    buffers['candidate']=keep_buffers(fast)
    fast.warmup();managers['candidate']=capture(fast)
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)

    # Graph dispatch and eager candidate arithmetic must agree on a private
    # stream, including position boundaries and padded attention capacity.
    ids=fixture['inputs']['input_ids'].cuda()
    fast.prefill(ids.repeat(1,8,1)[:,:1023])
    token=fixture['upstream_ids'][10:11][None].cuda()
    checks=[];manager=managers['candidate']
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pos in (0,1,31,32,126,127,128,254,255,256,510,511,512,1023,144,127):
            for length in (1,9,32):
                cap=min(c for c in manager.contexts if pos<c)
                count=min(c for c in manager.head_counts if c>=length)
                for layer in model.language_model.layers:layer.self_attn._decode_capacity=cap
                fast._audio_head_count=count
                fast.ids.copy_(token);fast.position.fill_(pos)
                fast.audio_length.fill_(length);fast.delay_length.fill_(-1)
                torch.cuda.manual_seed(3927)
                reference=tuple(t.clone() for t in fast._decode())
                torch.cuda.manual_seed(3927)
                actual_ids=manager.step(token,pos,length,-1)
                actual=(actual_ids,fast.text_logits,fast.audio_logits)
                assert all(torch.equal(x,y) for x,y in zip(reference,actual,strict=True)),(pos,length)
                checks.append({'position':pos,'audio_length':length,'capacity':cap,'heads':count,'exact':True})
    torch.cuda.current_stream().wait_stream(stream)
    fast._audio_head_count=32
    for layer in model.language_model.layers:layer.self_attn._decode_capacity=None

    def generate(name,seed):
        restore_buffers(buffers[name]);fast.step=managers[name].step
        torch.manual_seed(seed)
        chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400))
        pcm=torch.cat([c.pcm for c in chunks])
        assert torch.isfinite(pcm).all() and not engine.last_metrics['truncated']
        metrics=engine.last_metrics.copy()
        metrics['pcm_float32_sha256']=hashlib.sha256(pcm.numpy().tobytes()).hexdigest()
        return pcm,metrics

    pcm,_=generate('control',1235)
    original=next(RESULTS.glob('all32_gptq_dp4a_norm_projection_v1_*.wav'))
    wav=io.BytesIO();sf.write(wav,pcm.numpy(),24000,format='WAV',subtype='PCM_16')
    assert wav.getvalue()==original.read_bytes(),'Control buffers/graph restoration changed the reference WAV'
    rows=[]
    for pair in range(-1,a.pairs):
        order=('control','candidate') if pair%2==0 else ('candidate','control')
        records={name:generate(name,7000+pair)[1] for name in order}
        row={'pair':pair,'seed':7000+pair,'order':order,'records':records,
             'gain_ms':records['control']['ttfa_ms']-records['candidate']['ttfa_ms']}
        print({'pair':pair,'control_ms':records['control']['ttfa_ms'],
               'candidate_ms':records['candidate']['ttfa_ms'],'gain_ms':row['gain_ms']},flush=True)
        if pair>=0:rows.append(row)
    result={'codebooks':32,'torch':torch.__version__,'config':config,
            'method':'Shared original BF16 weights, prefill and codec; separately retained quantized output/down buffers and full head/context graphs. Restore buffers for eager steps and alternate complete requests, reversing order each pair. One warmup pair excluded. Cached cloned voice, network excluded.',
            'quality_scope':'Different calibrated models; PCM equality is not expected. Independent speech diagnostics required.',
            'control_reference_wav_exact':True,'private_stream_eager_graph_checks':checks,
            'rows':rows,'median_paired_gain_ms':statistics.median(r['gain_ms'] for r in rows),
            'improved_pairs':sum(r['gain_ms']>0 for r in rows)}
    for name in managers:result[name+'_ttfa']=stats([r['records'][name]['ttfa_ms'] for r in rows])
    path.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('rows','private_stream_eager_graph_checks','config')},flush=True)
    engine.codec.close()


if __name__=='__main__':main()
