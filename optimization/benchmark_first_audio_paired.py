"""Compare the initial-audio graph on complete, paired streaming requests."""
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
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--pairs',type=int,default=20);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.pairs<3:raise ValueError('Safe tag and at least three pairs required')
    path=RESULTS/f'first_audio_paired_{a.tag}.json';assert not path.exists(),'Preserve measurements'
    engine,manager=build_engine();configuration=enable(engine);graphs=engine._first_audio_graphs
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);rows=[]
    for pair in range(-1,a.pairs):
        order=('control','graph') if pair%2==0 else ('graph','control');records={};rngs={}
        for name in order:
            engine._first_audio_graphs=graphs if name=='graph' else None
            torch.manual_seed(7000+pair)
            chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400))
            pcm=torch.cat([c.pcm for c in chunks]);assert torch.isfinite(pcm).all()
            record=engine.last_metrics.copy();assert not record['truncated']
            record['pcm_float32_sha256']=hashlib.sha256(pcm.numpy().tobytes()).hexdigest();records[name]=record
            rngs[name]=torch.cuda.get_rng_state()
        assert records['control']['pcm_float32_sha256']==records['graph']['pcm_float32_sha256'],pair
        assert torch.equal(rngs['control'],rngs['graph']),pair
        assert not records['control']['first_audio_graph_ms'] and len(records['graph']['first_audio_graph_ms'])==1
        row={'pair':pair,'order':order,'seed':7000+pair,'records':records,'rng_exact':True,
             'gain_ms':records['control']['ttfa_ms']-records['graph']['ttfa_ms']}
        if pair>=0:rows.append(row)
        print('PAIR',pair,{n:r['ttfa_ms'] for n,r in records.items()},'GAIN',row['gain_ms'],flush=True)
    result={'codebooks':32,'configuration':configuration,'rows':rows,'all_pcm_exact':True,'all_final_rng_states_exact':True,
            'method':'One selected calibrated G32/attention-PDL/clocked-codec engine, ordinary versus one graph for the initial 32 audio steps. Alternating complete cached-voice requests, same seed per pair, one excluded warmup pair, no concurrent GPU job or profiler before timing. Per-step times inside the graph are null; the actual group wall time is recorded separately.',
            'ttfa':{n:stats([r['records'][n]['ttfa_ms'] for r in rows]) for n in ('control','graph')},
            'median_paired_gain_ms':statistics.median(r['gain_ms'] for r in rows),'faster_pairs':sum(r['gain_ms']>0 for r in rows)}
    path.write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',result['ttfa'],result['median_paired_gain_ms'],flush=True)
    engine._first_audio_graphs=graphs
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=34))
    prof.export_chrome_trace(str(RESULTS/f'first_audio_profile_{a.tag}_trace.json'))
    (RESULTS/f'first_audio_profile_{a.tag}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    engine.codec.close()


if __name__=='__main__':main()
