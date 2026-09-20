"""Full prefill equivalence and complete streaming requests for fused QKV prep."""
import argparse
import hashlib
import json
import statistics

import torch

from .common import RESULTS,stats
from .benchmark_first_audio_graph import build_engine
from .first_audio_graph import enable


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--pairs',type=int,default=20);p.add_argument('--profile',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.pairs<3:raise ValueError('Safe tag and three pairs required')
    path=RESULTS/f'prefill_qkv_paired_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    engine,_=build_engine(bulk=True);fast=engine.llm;enable(engine)
    graphs={'control':fast.prefill_graphs}
    for layer in fast.model.language_model.layers:layer.self_attn._fused_prefill_qkv=True
    fast.prefill_graphs={};fast.capture_prefill((128,160,256,512));graphs['candidate']=fast.prefill_graphs
    def select(name):
        fast.prefill_graphs=graphs[name]
        for layer in fast.model.language_model.layers:layer.self_attn._fused_prefill_qkv=name=='candidate'
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);ids=fixture['inputs']['input_ids'].cuda()
    storage=[t for layer in fast.cache.layers for t in (layer.keys,layer.values)];checks=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for n in (2,3,127,128,129,145,159,160,161,255,256,257,511,512,513,1023):
            token=ids.repeat(1,8,1)[:,:n].contiguous();expected=None;reference_cache=None
            for name in ('control','candidate'):
                select(name)
                for t in storage:t.fill_(float('nan'))
                actual=fast.prefill(token)
                if expected is None:
                    expected=tuple(t.clone() for t in actual);reference_cache=[t.clone() for t in storage]
                else:
                    mismatches=[int((x.view(torch.int16)!=y.view(torch.int16)).sum()) for x,y in zip(actual,expected,strict=True)]
                    cache_exact=all(torch.equal(x.view(torch.int16),y.view(torch.int16)) for x,y in zip(storage,reference_cache,strict=True))
                    checks.append({'tokens':n,'mismatches_text_audio':mismatches,'all_72_full_kv_buffers_exact':cache_exact})
                    path.write_text(json.dumps({'codebooks':32,'checks':checks,'complete':False},indent=2)+'\n')
                    assert not any(mismatches) and cache_exact,checks[-1]
            print('PREFILL EXACT',n,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    rows=[]
    for pair in range(-1,a.pairs):
        order=('control','candidate') if pair%2==0 else ('candidate','control');records={};rng={}
        for name in order:
            select(name);torch.manual_seed(7000+pair)
            chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400));pcm=torch.cat([c.pcm for c in chunks])
            assert torch.isfinite(pcm).all() and not engine.last_metrics['truncated']
            records[name]=dict(engine.last_metrics,pcm_float32_sha256=hashlib.sha256(pcm.numpy().tobytes()).hexdigest());rng[name]=torch.cuda.get_rng_state()
        assert records['control']['pcm_float32_sha256']==records['candidate']['pcm_float32_sha256']
        assert torch.equal(rng['control'],rng['candidate'])
        row={'pair':pair,'seed':7000+pair,'order':order,'records':records,'rng_exact':True,'gain_ms':records['control']['ttfa_ms']-records['candidate']['ttfa_ms']}
        if pair>=0:rows.append(row)
        print('PAIR',pair,{n:r['ttfa_ms'] for n,r in records.items()},'GAIN',row['gain_ms'],flush=True)
    result={'codebooks':32,'complete':True,'checks':checks,'rows':rows,'all_pcm_exact':True,'all_final_rng_exact':True,
        'ttfa':{n:stats([r['records'][n]['ttfa_ms'] for r in rows]) for n in ('control','candidate')},
        'median_paired_gain_ms':statistics.median(r['gain_ms'] for r in rows),'faster_pairs':sum(r['gain_ms']>0 for r in rows),
        'method':'Same selected G32/PDL/clocked-codec/first-audio/bulk-hint engine, including QKV address specialization. Separate ordinary/fused prefill graph sets, unchanged decode graphs. Sixteen full-model prefill lengths compare text/audio logits and all 72 KV buffers bitwise including NaNs. Alternating complete cached-voice requests, one excluded warmup pair; timing before profiling.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',result['ttfa'],result['median_paired_gain_ms'],flush=True)
    if a.profile:
        select('candidate')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=34))
        prof.export_chrome_trace(str(RESULTS/f'prefill_qkv_profile_{a.tag}_trace.json'))
        (RESULTS/f'prefill_qkv_profile_{a.tag}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    engine.codec.close()


if __name__=='__main__':main()
