"""Full-stream exactness and TTFA for experimental attention-output copies."""
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
from .async_output import make_dispatch


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=12)
    p.add_argument('--configs',nargs='+',choices=('copy_r8','copy_r16','register_r8','register_r16'),default=['copy_r8','copy_r16'])
    p.add_argument('--profile',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<3:raise ValueError('Safe tag and three rounds required')
    path=RESULTS/f'async_output_paired_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    engine,manager=build_engine(bulk=True,prefill_qkv=True,prefill_pointwise=True);fast=engine.llm;enable(engine)
    managers={'control':manager};bursts={'control':engine._first_audio_graphs};dispatches={'control':None}
    def select_dispatch(name):
        for layer in fast.model.language_model.layers:layer.self_attn._async_out_linear=dispatches[name]
    for name,rows in (('copy_r8',8),('copy_r16',16),('register_r8',8),('register_r16',16)):
        if name not in a.configs:continue
        dispatches[name]=make_dispatch(rows,register_preload=name.startswith('register'));select_dispatch(name)
        fast._audio_head_buckets=None;fast.step=types.MethodType(FastLLM.step,fast)
        fast.warmup();managers[name]=capture(fast);managers[name].install()
        engine._first_audio_graphs=None;enable(engine);bursts[name]=engine._first_audio_graphs
        print('CAPTURED',name,flush=True)
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);ids=fixture['inputs']['input_ids'].cuda()
    fast.prefill(ids.repeat(1,8,1)[:,:1023]);token=fixture['upstream_ids'][10:11][None].cuda()
    checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name in tuple(managers)[1:]:
            manager=managers[name]
            for pos in (0,1,31,32,126,127,128,254,255,256,510,511,512,1023,144,127):
                for length in (1,9,32):
                    cap=min(c for c in manager.contexts if pos<c);heads=min(c for c in manager.head_counts if c>=length)
                    for layer in fast.model.language_model.layers:layer.self_attn._decode_capacity=cap
                    fast._audio_head_count=heads;fast.ids.copy_(token);fast.position.fill_(pos)
                    fast.audio_length.fill_(length);fast.delay_length.fill_(-1)
                    select_dispatch('control');torch.cuda.manual_seed(3927);expected=tuple(t.clone() for t in fast._decode())
                    select_dispatch(name);torch.cuda.manual_seed(3927);out=manager.step(token,pos,length,-1)
                    actual=(out,fast.text_logits,fast.audio_logits)
                    assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(actual,expected,strict=True)),(name,pos,length)
                    checks.append({'mode':name,'position':pos,'length':length,'selected_eager_candidate_graph_bits_exact':True})
            print('GRAPH CHECKED',name,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    fast._audio_head_count=32
    for layer in fast.model.language_model.layers:layer.self_attn._decode_capacity=None
    rows=[];names=tuple(managers)
    previous=json.loads((RESULTS/'prefill_pointwise_paired_v1.json').read_text())
    prior_hashes={r['seed']:r['records']['both']['pcm_float32_sha256'] for r in previous['rows']}
    result={'codebooks':32,'checks':checks,'rows':rows,'complete':False,
        'method':'Selected G32/FP32-codec model with output-weight staging: cp.async/swizzle8 or ordinary register preload before PDL wait, 8/16 output rows per CTA. Independent context/head/first-audio graph sets; eager dispatch restored with each set. Rotating/reversing cached-voice complete requests, one excluded warmup. Full PCM and final RNG compared; timing excludes network.',
        'configs':{name:{'rows':int(name.split('_r')[1]),'register_preload':name.startswith('register')} for name in names[1:]}}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    save()
    for repeat in range(-1,a.rounds):
        offset=repeat%len(names);order=names[offset:]+names[:offset]
        if repeat%2:order=tuple(reversed(order))
        records={};rng={}
        for name in order:
            select_dispatch(name);fast.step=managers[name].step;fast._audio_head_buckets=managers[name];engine._first_audio_graphs=bursts[name]
            torch.manual_seed(7000+repeat)
            chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400));pcm=torch.cat([c.pcm for c in chunks])
            assert torch.isfinite(pcm).all() and not engine.last_metrics['truncated'],name
            records[name]=dict(engine.last_metrics,pcm_float32_sha256=hashlib.sha256(pcm.numpy().tobytes()).hexdigest());rng[name]=torch.cuda.get_rng_state()
        assert all(records['control']['pcm_float32_sha256']==records[n]['pcm_float32_sha256'] for n in names[1:])
        assert all(torch.equal(rng['control'],rng[n]) for n in names[1:])
        seed=7000+repeat
        if seed in prior_hashes:assert records['control']['pcm_float32_sha256']==prior_hashes[seed],'Control changed'
        row={'round':repeat,'seed':seed,'order':order,'records':records,'pcm_rng_exact':True,
             'gains_ms':{n:records['control']['ttfa_ms']-records[n]['ttfa_ms'] for n in names[1:]}}
        if repeat>=0:rows.append(row)
        save();print('ROUND',repeat,{n:r['ttfa_ms'] for n,r in records.items()},'GAIN',row['gains_ms'],flush=True)
    result.update(complete=True,control_prior_pcm_hashes_exact=sum(r['seed'] in prior_hashes for r in rows),all_pcm_rng_exact=True,
        ttfa={n:stats([r['records'][n]['ttfa_ms'] for r in rows]) for n in names},
        median_paired_gain_ms={n:statistics.median(r['gains_ms'][n] for r in rows) for n in names[1:]},
        faster_rounds={n:sum(r['gains_ms'][n]>0 for r in rows) for n in names[1:]})
    save();print('SUMMARY',result['ttfa'],result['median_paired_gain_ms'],flush=True)
    if a.profile:
        name=names[1];select_dispatch(name);fast.step=managers[name].step
        fast._audio_head_buckets=managers[name];engine._first_audio_graphs=bursts[name]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=34))
        prof.export_chrome_trace(str(RESULTS/f'async_output_profile_{a.tag}_trace.json'))
        (RESULTS/f'async_output_profile_{a.tag}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    engine.codec.close()


if __name__=='__main__':main()
