"""Full-model initial-interval checks across context boundaries and poisoned KV."""
import argparse
import json

import torch

from .common import RESULTS
from .benchmark_first_audio_graph import build_engine
from .first_audio_graph import FirstAudioGraph


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--short',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'first_audio_validation_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    engine,manager=build_engine();fast=engine.llm
    caps=(256,) if a.short else (128,256,512,1024)
    graphs={c:FirstAudioGraph(fast,c,save_logits=True) for c in caps}
    for graph in graphs.values():graph.warmup()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);ids=fixture['inputs']['input_ids'].cuda()
    fast.prefill(ids.repeat(1,8,1)[:,:1023]);torch.cuda.synchronize()
    storage=[t for layer in fast.cache.layers for t in (layer.keys,layer.values)]
    prefix=[t.clone() for t in storage];assert len(storage)==72
    first=torch.full((1,1,33),1024,device='cuda',dtype=torch.long);first[...,0]=fast.cfg.audio_start_token_id
    positions=(145,) if a.short else (0,1,95,96,97,127,128,223,224,225,255,256,479,480,481,511,512,991,992)
    checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pos in positions:
            graph=graphs[min(c for c in caps if pos+32<=c)]
            for initial_delay in (-1,0):
                seed=41000+pos*2+initial_delay
                for dest,source in zip(storage,prefix,strict=True):dest.copy_(source);dest[:,:,pos:].fill_(float('nan'))
                torch.cuda.manual_seed(seed);token=first;length=1;delay=initial_delay;history=[];texts=[];audios=[]
                for step in range(32):
                    token=manager.step(token,pos+step,length,delay)
                    text=int(token[0,0,0]);history.append(token.clone());texts.append(fast.text_logits.clone());audios.append(fast.audio_logits.clone())
                    length+=1
                    if delay<0 and text==fast.cfg.audio_assistant_delay_slot_token_id:delay=0
                    if delay>=0:delay+=1
                expected=torch.cat(history).reshape(32,33);text_expected=torch.cat(texts);audio_expected=torch.stack(audios)
                expected_cache=[t.clone() for t in storage];rng=torch.cuda.get_rng_state()
                for dest,source in zip(storage,prefix,strict=True):dest.copy_(source);dest[:,:,pos:].fill_(float('nan'))
                torch.cuda.manual_seed(seed);actual,status=graph.run(first,pos,delay=initial_delay);stream.synchronize()
                counts={'ids':int((actual!=expected).sum()),'text_logits':int((graph.text_history!=text_expected).sum()),'audio_logits':int((graph.audio_history!=audio_expected).sum())}
                cache_exact=all(torch.equal(x.view(torch.int16),y.view(torch.int16)) for x,y in zip(storage,expected_cache,strict=True))
                status_exact=status.tolist()==[int(expected[-1,0]),length,delay,pos+32]
                rng_exact=torch.equal(torch.cuda.get_rng_state(),rng)
                exact=not any(counts.values()) and cache_exact and status_exact and rng_exact
                checks.append({'position':pos,'capacity':graph.capacity,'initial_delay':initial_delay,'mismatches':counts,'all_72_kv_buffers_exact':cache_exact,'status_exact':status_exact,'rng_exact':rng_exact,'exact':exact})
                path.write_text(json.dumps({'codebooks':32,'checks':checks,'complete':False},indent=2)+'\n')
                assert exact,checks[-1]
            print('CHECKED',pos,flush=True)
        guard=[]
        for cap,graph in graphs.items():
            before=fast.position.clone()
            try:graph.run(first,cap-31)
            except ValueError:pass
            else:raise AssertionError('Interval outside context capacity accepted')
            assert torch.equal(before,fast.position);guard.append(cap)
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'complete':True,'all_exact':True,'checks':checks,'intervals':len(checks),'decode_steps_per_path':len(checks)*32,'guarded_capacities':guard,
            'scope':'Private-stream full-model intervals, original per-step context/head graph dispatch versus one initial graph. Every text/audio logit and sampled ID, final RNG/status and all 72 entire KV buffers compared bitwise, including poisoned future slots. Immediate-delay and ordinary fresh-audio cases. Short sanitizer mode covers only position 145; separate full run covers all capacity boundaries and final valid position.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(checks),flush=True);engine.codec.close()


if __name__=='__main__':main()
