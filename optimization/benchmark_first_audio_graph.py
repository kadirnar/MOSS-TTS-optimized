"""Initial 32-step graph audit, including stochastic sequence and RNG state."""
import argparse
import hashlib
import json
import statistics
import time

import torch

from . import benchmark_codec_clock_paired as preset
from .common import RESULTS
from .first_audio_graph import FirstAudioGraph


def build_engine(*,bulk=False,prefill_qkv=False,prefill_pointwise=False,output_weight_prefetch=False):
    model,codec,processor=preset.load_models()
    engine=preset.StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True,codec_clock=True)
    fast=engine.llm
    preset.install_calibrated(fast,RESULTS/'gptq_v1_g32_d10','dp4a')
    preset.enable_grouped_activation(fast);preset.enable_fusions(fast,layout_limit=8);preset.enable_gateup(fast)
    preset.install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json');preset.enable_attention_quant(fast)
    preset.enable_native_attention(fast);preset.enable_gateup_quant(fast);preset.enable_scaled(fast);preset.enable_norm_projection(fast)
    preset.short_module.enable(fast);preset.enable_projection_pdl(fast);preset.enable_attention_pdl(fast)
    if bulk:
        from .bulk_prefetch import enable
        enable(fast)
    if prefill_qkv:
        from .prefill_qkv import enable
        enable(fast)
    if prefill_pointwise:
        from .prefill_pointwise import enable
        enable(fast)
    if output_weight_prefetch:
        from .async_output import enable
        enable(fast,register_preload=True)
    engine.warmup((128,160,256,512));manager=preset.capture(fast);manager.install()
    return engine,manager


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=10);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and at least two rounds required')
    path=RESULTS/f'first_audio_graph_{a.tag}.json';assert not path.exists(),'Preserve measurements'
    engine,manager=build_engine();fast=engine.llm
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);ids=fixture['inputs']['input_ids'].cuda();prompt=ids.shape[1]
    assert prompt+32<=256
    graph=FirstAudioGraph(fast,256,save_logits=True);graph.warmup();print('CAPTURED',graph.metadata,flush=True)
    first=torch.full((1,1,33),1024,device='cuda',dtype=torch.long);first[...,0]=fast.cfg.audio_start_token_id
    checks=[];rows=[]
    for seed in (1234,7000,7001):
        for initial_delay in (-1,0):
            fast.prefill(ids);torch.cuda.manual_seed(seed);token=first;length=1;delay=initial_delay
            history=[];texts=[];audios=[]
            for step in range(32):
                token=manager.step(token,prompt+step,length,delay)
                text=int(token[0,0,0]);history.append(token.clone());texts.append(fast.text_logits.clone());audios.append(fast.audio_logits.clone())
                length+=1
                if delay<0 and text==fast.cfg.audio_assistant_delay_slot_token_id:delay=0
                if delay>=0:delay+=1
            expected=torch.cat(history).reshape(32,33);expected_text=torch.cat(texts);expected_audio=torch.stack(audios)
            rng=torch.cuda.get_rng_state();expected_status=[int(expected[-1,0]),length,delay,prompt+32]
            fast.prefill(ids);torch.cuda.manual_seed(seed);actual,status=graph.run(first,prompt,delay=initial_delay)
            counts={'ids':int((actual!=expected).sum()),'text_logits':int((graph.text_history!=expected_text).sum()),
                    'audio_logits':int((graph.audio_history!=expected_audio).sum())}
            exact=not any(counts.values()) and status.tolist()==expected_status and torch.equal(torch.cuda.get_rng_state(),rng)
            row={'seed':seed,'initial_delay':initial_delay,'mismatches':counts,'status_exact':status.tolist()==expected_status,'rng_exact':torch.equal(torch.cuda.get_rng_state(),rng),'exact':exact}
            checks.append(row);print('CHECK',row,flush=True)
    result={'codebooks':32,'checks':checks,'capture':graph.metadata,'rows':rows,'scope':'Initial 32 audio steps on one cached voice prompt; logits/IDs/status/RNG checked with fresh and immediate-delay cases. Does not measure full TTFA or qualify complete streaming.'}
    path.write_text(json.dumps(result,indent=2)+'\n')
    if not all(r['exact'] for r in checks):raise RuntimeError('First-audio graph differs; no timing claim')
    # Separate graph without audit stores for the timing comparison.
    timed=FirstAudioGraph(fast,256);timed.warmup();result['timed_capture']=timed.metadata
    for repeat in range(a.rounds):
        order=('control','graph') if repeat%2==0 else ('graph','control');times={};hashes={}
        for name in order:
            fast.prefill(ids);torch.cuda.manual_seed(8000+repeat);torch.cuda.synchronize();start=time.perf_counter()
            if name=='control':
                token=first;length=1;delay=-1;history=[]
                for step in range(32):
                    token=manager.step(token,prompt+step,length,delay);text=int(token[0,0,0]);history.append(token)
                    length+=1
                    if delay<0 and text==fast.cfg.audio_assistant_delay_slot_token_id:delay=0
                    if delay>=0:delay+=1
                history=torch.cat(history).reshape(32,33).cpu()
            else:history=timed.run(first,prompt)[0].cpu()
            times[name]=(time.perf_counter()-start)*1000;hashes[name]=hashlib.sha256(history.numpy().tobytes()).hexdigest()
        assert hashes['control']==hashes['graph']
        rows.append({'round':repeat,'order':order,'ms':times,'ids_sha256':hashes['control']});print('ROUND',repeat,times,flush=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
    result['median_ms']={n:statistics.median(r['ms'][n] for r in rows) for n in ('control','graph')}
    result['median_paired_gain_ms']=statistics.median(r['ms']['control']-r['ms']['graph'] for r in rows)
    path.write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',result['median_ms'],result['median_paired_gain_ms'],flush=True)
    engine.codec.close()


if __name__=='__main__':main()
