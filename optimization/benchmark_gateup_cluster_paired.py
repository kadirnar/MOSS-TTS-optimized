"""Full-stream gate/up cluster checks against the selected eight-row down preset."""
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
from .qkv_cluster_binary import load_bundle
from .qkv_cluster_model import Hidden
from .gateup_cluster_binary import load_bundle as load_up_bundle
from .bulk_prefetch import configured as projection_configured
from pathlib import Path


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=12)
    p.add_argument('--binary-bundle',type=Path,default=RESULTS/'qkv_cluster_bundle_v6');p.add_argument('--configs',nargs='+',required=True);p.add_argument('--up-bundle',type=Path,required=True)
    p.add_argument('--profile',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<3:raise ValueError('Safe tag and three rounds required')
    path=RESULTS/f'gateup_cluster_paired_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    engine,old_manager=build_engine(bulk=True,prefill_qkv=True,prefill_pointwise=True,output_weight_prefetch=True);fast=engine.llm
    old_manager=None
    managers={};bursts={};dispatches={};original_hidden=fast.hidden
    selected,selected_launchers=load_bundle(a.binary_bundle)
    up_options,up_launchers=load_up_bundle(a.up_bundle)
    choices={'control':None,**{n:up_options[n] for n in a.configs}}
    original_norm=fast._bulk_norm_linear
    down_dispatches={};norm_dispatches={}
    def make_down(tile):
        fn=projection_configured('projection',divisor=16)
        options={'rows':8,'warps':4,'integer_groups':1,'integer_rows':1,'prefetch':1,'trigger_mode':3,**tile}
        def call(*args,**kwargs):return fn(*args,**{**kwargs,**options})
        return call
    def make_norm(name):
        def call(*args,**kwargs):
            if name=='control' or not kwargs.get('fused',False):return original_norm(*args,**kwargs)
            kwargs.pop('fused');kwargs.pop('rows',None)
            kwargs.pop('integer_groups',None);kwargs.pop('integer_rows',None)
            kwargs.pop('trigger_mode',None);kwargs.pop('pdl',None)
            return up_launchers[name](*args,**up_options[name],**kwargs)
        return call
    def select_dispatch(name):
        fast.hidden=dispatches[name];fast._bulk_norm_linear=norm_dispatches[name]
        for layer in fast.model.language_model.layers:layer.mlp._bulk_down_linear=down_dispatches[name]
    for name,choice in choices.items():
        fast.hidden=original_hidden
        dispatches[name]=Hidden(fast,selected_launchers['c8_t2_exact'],selected['c8_t2_exact'])
        down_dispatches[name]=make_down({});norm_dispatches[name]=make_norm(name)
        select_dispatch(name)
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
                    expected_rng=torch.cuda.get_rng_state()
                    cache=[t.clone() for layer in fast.cache.layers for t in (layer.keys,layer.values)]
                    for layer in fast.cache.layers:
                        layer.keys[:,:,pos,:].fill_(float('nan'));layer.values[:,:,pos,:].fill_(float('nan'))
                    select_dispatch(name);torch.cuda.manual_seed(3927);out=manager.step(token,pos,length,-1)
                    actual=(out,fast.text_logits,fast.audio_logits)
                    assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(actual,expected,strict=True)),(name,pos,length)
                    assert torch.equal(expected_rng,torch.cuda.get_rng_state()),'RNG mismatch'
                    assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(cache,(t for layer in fast.cache.layers for t in (layer.keys,layer.values)),strict=True)),('Cache mismatch',name,pos,length)
                    checks.append({'mode':name,'position':pos,'length':length,'selected_eager_candidate_graph_bits_exact':True,'all_72_poisoned_caches_rng_exact':True})
            print('GRAPH CHECKED',name,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    fast._audio_head_count=32
    for layer in fast.model.language_model.layers:layer.self_attn._decode_capacity=None
    rows=[];names=tuple(managers)
    previous=json.loads((RESULTS/'down_placement_paired_v2.json').read_text())
    prior_hashes={r['seed']:r['records']['down_r8']['pcm_float32_sha256'] for r in previous['rows']}
    result={'codebooks':32,'checks':checks,'rows':rows,'complete':False,
        'method':'Qualified G32/FP32-codec model with clustered QKV, register-preloaded output and eight-row down projection. Gate/up cluster variants retain original dependency waits and 32 codebooks. Independent context/head/first-audio graph sets; eager dispatch restored with each set. Rotating/reversing cached-voice complete requests, one excluded warmup. Full PCM and final RNG compared; timing excludes network.',
        'binary_bundle':str(a.binary_bundle),'up_bundle':str(a.up_bundle),
        'configs':choices}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    save()
    for repeat in range(-1,a.rounds):
        offset=(repeat//2)%len(names);order=names[offset:]+names[:offset]
        if repeat%2:order=tuple(reversed(order))
        records={};rng={}
        for name in order:
            select_dispatch(name);fast.step=managers[name].step;fast._audio_head_buckets=managers[name];engine._first_audio_graphs=bursts[name]
            torch.manual_seed(7000+repeat)
            chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400));pcm=torch.cat([c.pcm for c in chunks])
            assert torch.isfinite(pcm).all() and not engine.last_metrics['truncated'],name
            records[name]=dict(engine.last_metrics,pcm_float32_sha256=hashlib.sha256(pcm.numpy().tobytes()).hexdigest());rng[name]=torch.cuda.get_rng_state()
        pcm_exact=all(records['control']['pcm_float32_sha256']==records[n]['pcm_float32_sha256'] for n in names[1:])
        rng_exact=all(torch.equal(rng['control'],rng[n]) for n in names[1:])
        if not pcm_exact or not rng_exact:
            result['failed_request']={'round':repeat,'records':records,'pcm_exact':pcm_exact,'rng_exact':rng_exact};save()
            raise AssertionError('Complete stream differs from selected output; failure saved')
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
        prof.export_chrome_trace(str(RESULTS/f'gateup_cluster_profile_{a.tag}_trace.json'))
        (RESULTS/f'gateup_cluster_profile_{a.tag}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    engine.codec.close()


if __name__=='__main__':main()
