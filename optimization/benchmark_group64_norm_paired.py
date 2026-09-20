"""Complete streamed TTFA screen for selective G64 norm projections."""
import argparse
import hashlib
import json
import statistics
import types

import torch

from .common import RESULTS,stats
from .benchmark_first_audio_graph import build_engine
from .benchmark_qkv_load_paired import capture
from .first_audio_graph import enable
from .llm import FastLLM
from .group64_norm_experiment import prepare


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=12);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<3:raise ValueError('Safe tag and three rounds required')
    path=RESULTS/f'group64_norm_paired_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    engine,manager=build_engine(bulk=True,prefill_qkv=True,prefill_pointwise=True);fast=engine.llm;enable(engine)
    managers={'control':manager};bursts={'control':engine._first_audio_graphs}
    dispatches={'control':fast._bulk_norm_linear};configs={}
    for name,stages in (('up',('up',)),('up_qkv',('up','qkv'))):
        fast._bulk_norm_linear=dispatches['control']
        dispatches[name]=prepare(fast,stages);configs[name]=dispatches[name].metadata
        fast._bulk_norm_linear=dispatches[name]
        fast._audio_head_buckets=None;fast.step=types.MethodType(FastLLM.step,fast)
        fast.warmup();managers[name]=capture(fast);managers[name].install()
        engine._first_audio_graphs=None;enable(engine);bursts[name]=engine._first_audio_graphs
        print('CAPTURED',name,flush=True)
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);ids=fixture['inputs']['input_ids'].cuda()
    fast.prefill(ids.repeat(1,8,1)[:,:1023]);token=fixture['upstream_ids'][10:11][None].cuda()
    checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name in ('up','up_qkv'):
            fast._bulk_norm_linear=dispatches[name];manager=managers[name]
            for pos in (0,1,31,32,126,127,128,254,255,256,510,511,512,1023,144,127):
                for length in (1,9,32):
                    cap=min(c for c in manager.contexts if pos<c);heads=min(c for c in manager.head_counts if c>=length)
                    for layer in fast.model.language_model.layers:layer.self_attn._decode_capacity=cap
                    fast._audio_head_count=heads;fast.ids.copy_(token);fast.position.fill_(pos)
                    fast.audio_length.fill_(length);fast.delay_length.fill_(-1)
                    torch.cuda.manual_seed(3927);expected=tuple(t.clone() for t in fast._decode())
                    torch.cuda.manual_seed(3927);out=manager.step(token,pos,length,-1)
                    actual=(out,fast.text_logits,fast.audio_logits)
                    assert all(torch.equal(x,y) for x,y in zip(actual,expected,strict=True)),(name,pos,length)
                    checks.append({'mode':name,'position':pos,'length':length,'eager_graph_exact':True})
            print('GRAPH CHECKED',name,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    fast._audio_head_count=32
    for layer in fast.model.language_model.layers:layer.self_attn._decode_capacity=None
    rows=[];names=tuple(managers)
    previous=json.loads((RESULTS/'prefill_pointwise_paired_v1.json').read_text())
    prior_hashes={r['seed']:r['records']['both']['pcm_float32_sha256'] for r in previous['rows']}
    result={'codebooks':32,'configs':configs,'checks':checks,'rows':rows,'complete':False,
        'method':'One selected model; shared BF16 prefill, G32 attention-output/down and FP32 codec, with separate norm dispatch buffers, head/context graphs and first-32-audio graphs. Restore graph and eager dispatch together. Rotate/reverse complete requests, one warmup round excluded; cached voice, network excluded.',
        'quality_scope':'Different weight quantization; output may differ. No speech-quality acceptance from this timing screen.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    save()
    for repeat in range(-1,a.rounds):
        offset=repeat%len(names);order=names[offset:]+names[:offset]
        if repeat%2:order=tuple(reversed(order))
        records={}
        for name in order:
            fast._bulk_norm_linear=dispatches[name];fast.step=managers[name].step
            fast._audio_head_buckets=managers[name];engine._first_audio_graphs=bursts[name]
            torch.manual_seed(7000+repeat)
            chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400));pcm=torch.cat([c.pcm for c in chunks])
            assert torch.isfinite(pcm).all() and not engine.last_metrics['truncated'],name
            records[name]=dict(engine.last_metrics,pcm_float32_sha256=hashlib.sha256(pcm.numpy().tobytes()).hexdigest())
        seed=7000+repeat
        if seed in prior_hashes:assert records['control']['pcm_float32_sha256']==prior_hashes[seed],'Control changed'
        row={'round':repeat,'seed':seed,'order':order,'records':records,
             'gains_ms':{n:records['control']['ttfa_ms']-records[n]['ttfa_ms'] for n in names[1:]}}
        if repeat>=0:rows.append(row)
        save();print('ROUND',repeat,{n:r['ttfa_ms'] for n,r in records.items()},'GAIN',row['gains_ms'],flush=True)
    result.update(complete=True,control_prior_pcm_hashes_exact=len(rows),
        ttfa={n:stats([r['records'][n]['ttfa_ms'] for r in rows]) for n in names},
        median_paired_gain_ms={n:statistics.median(r['gains_ms'][n] for r in rows) for n in names[1:]},
        faster_rounds={n:sum(r['gains_ms'][n]>0 for r in rows) for n in names[1:]})
    save();print('SUMMARY',result['ttfa'],result['median_paired_gain_ms'],flush=True)
    engine.codec.close()


if __name__=='__main__':main()
