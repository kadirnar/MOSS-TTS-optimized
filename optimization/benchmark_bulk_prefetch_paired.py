"""Isolated full-request screen of the best bulk-prefetch chain candidate."""
import argparse
import contextlib
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
from .bulk_prefetch import configured
from . import dp4a_norm_pdl as norm_module
from . import dp4a_layout_pdl_prefetch as projection_module


@contextlib.contextmanager
def experimental_bindings():
    """Private single-thread benchmark only; always restore module functions."""
    old_norm=norm_module.linear;old_projection=projection_module.linear
    norm_candidate=configured('norm',divisor=16)
    down_candidate=configured('projection',divisor=16)
    def dispatch(x,w,s,**kwargs):
        return (down_candidate if w.shape[1]==6144 else old_projection)(x,w,s,**kwargs)
    try:
        norm_module.linear=norm_candidate;projection_module.linear=dispatch
        yield
    finally:
        norm_module.linear=old_norm;projection_module.linear=old_projection


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--pairs',type=int,default=20);p.add_argument('--profile',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.pairs<3:raise ValueError('Safe tag and at least three pairs required')
    path=RESULTS/f'bulk_prefetch_paired_{a.tag}.json';assert not path.exists(),'Preserve measurements'
    engine,manager=build_engine();fast=engine.llm;managers={'control':manager};enable(engine)
    bursts={'control':engine._first_audio_graphs}
    # Capture a distinct graph set without chaining the old manager's dispatch.
    fast._audio_head_buckets=None;fast.step=types.MethodType(FastLLM.step,fast)
    with experimental_bindings():
        fast.warmup();managers['candidate']=capture(fast);managers['candidate'].install()
        engine._first_audio_graphs=None;enable(engine);bursts['candidate']=engine._first_audio_graphs
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);ids=fixture['inputs']['input_ids'].cuda()
    fast.prefill(ids.repeat(1,8,1)[:,:1023]);token=fixture['upstream_ids'][10:11][None].cuda();checks=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pos in (0,1,31,32,126,127,128,254,255,256,510,511,512,1023,144,127):
            for length in (1,9,32):
                expected=None
                for name in ('control','candidate'):
                    torch.cuda.manual_seed(3927)
                    output=managers[name].step(token,pos,length,-1)
                    tensors=(output,fast.text_logits,fast.audio_logits)
                    if expected is None:expected=tuple(x.clone() for x in tensors)
                    else:assert all(torch.equal(x,y) for x,y in zip(tensors,expected,strict=True)),(pos,length)
                checks.append({'position':pos,'length':length,'all_logits_and_ids_exact':True})
    torch.cuda.current_stream().wait_stream(stream)
    rows=[]
    for pair in range(-1,a.pairs):
        order=('control','candidate') if pair%2==0 else ('candidate','control');records={};rng={}
        for name in order:
            fast.step=managers[name].step;fast._audio_head_buckets=managers[name];engine._first_audio_graphs=bursts[name]
            with experimental_bindings() if name=='candidate' else contextlib.nullcontext():
                torch.manual_seed(7000+pair);chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400))
            pcm=torch.cat([c.pcm for c in chunks]);assert torch.isfinite(pcm).all() and not engine.last_metrics['truncated']
            records[name]=dict(engine.last_metrics,pcm_float32_sha256=hashlib.sha256(pcm.numpy().tobytes()).hexdigest())
            rng[name]=torch.cuda.get_rng_state()
        assert records['control']['pcm_float32_sha256']==records['candidate']['pcm_float32_sha256']
        assert torch.equal(rng['control'],rng['candidate'])
        row={'pair':pair,'seed':7000+pair,'order':order,'records':records,'rng_exact':True,
            'gain_ms':records['control']['ttfa_ms']-records['candidate']['ttfa_ms']}
        if pair>=0:rows.append(row)
        print('PAIR',pair,{n:r['ttfa_ms'] for n,r in records.items()},'GAIN',row['gain_ms'],flush=True)
    result={'codebooks':32,'rows':rows,'checks':checks,'all_pcm_exact':True,'all_final_rng_exact':True,
        'ttfa':{n:stats([r['records'][n]['ttfa_ms'] for r in rows]) for n in ('control','candidate')},
        'median_paired_gain_ms':statistics.median(r['gain_ms'] for r in rows),'faster_pairs':sum(r['gain_ms']>0 for r in rows),
        'method':'One selected G32/PDL/clocked-codec/initial-audio-graph model, separate control/candidate graph sets. Candidate hints the first 1/16 of each CTA weight span in QKV, gate/up and down before dependency waits; attention output is unchanged. Alternating full cached-voice requests, one excluded warmup pair. Global function bindings are scoped to this isolated benchmark process and restored; no serving change or profiler during timing.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',result['ttfa'],result['median_paired_gain_ms'],flush=True)
    if a.profile:
        fast.step=managers['candidate'].step;fast._audio_head_buckets=managers['candidate'];engine._first_audio_graphs=bursts['candidate']
        with experimental_bindings():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=34))
        prof.export_chrome_trace(str(RESULTS/f'bulk_prefetch_profile_{a.tag}_trace.json'))
        (RESULTS/f'bulk_prefetch_profile_{a.tag}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    engine.codec.close()


if __name__=='__main__':main()
