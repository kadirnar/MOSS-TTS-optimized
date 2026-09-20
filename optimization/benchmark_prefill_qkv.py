"""Capture real prefill activations and compare fused QKV preparation."""
import argparse
import json
import statistics

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .common import RESULTS
from .benchmark_first_audio_graph import build_engine
from .prefill_qkv import project
from .tune_weight_reads import measure


def reference(entry):
    a=entry['attention'];x=entry['qkv'];n=x.shape[1]
    q,k,v=x.split((4096,1024,1024),-1)
    q=a.q_norm(q.reshape(1,n,32,128)).transpose(1,2)
    k=a.k_norm(k.reshape(1,n,8,128)).transpose(1,2)
    v=v.reshape(1,n,8,128).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,entry['cos'],entry['sin'])
    entry['keys'].index_copy_(2,entry['positions'],k)
    entry['values'].index_copy_(2,entry['positions'],v)
    return q


def candidate(entry):
    a=entry['attention']
    return project(entry['qkv'],a.q_norm.weight,a.k_norm.weight,entry['cos'],entry['sin'],
        entry['keys'],entry['values'],entry['positions'],a.q_norm.variance_epsilon)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=8);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'prefill_qkv_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    engine,_=build_engine();fast=engine.llm;fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    states={};hooks=[]
    for index,layer in enumerate(fast.model.language_model.layers):
        def hook(module,args,kwargs,index=index):
            states[index]=(kwargs['hidden_states'].clone(),tuple(t.clone() for t in kwargs['position_embeddings']))
        hooks.append(layer.self_attn.register_forward_pre_hook(hook,with_kwargs=True))
    ids=torch.full((1,160,33),1024,device='cuda',dtype=torch.long);ids[...,0]=fast.cfg.pad_token_id
    source=fixture['inputs']['input_ids'].cuda();ids[:,:source.shape[1]].copy_(source)
    positions=torch.arange(160,device='cuda');last=torch.tensor([source.shape[1]-1],device='cuda')
    try:fast._prefill_forward(ids,positions,last)
    finally:
        for hook in hooks:hook.remove()
    assert len(states)==36
    ring=[]
    for i,layer in enumerate(fast.model.language_model.layers):
        x,(cos,sin)=states[i];cache=fast.cache.layers[i]
        ring.append({'layer':i,'attention':layer.self_attn,'qkv':F.linear(x,layer.self_attn._qkv),
            'cos':cos.contiguous(),'sin':sin.contiguous(),'positions':positions,
            'keys':cache.keys,'values':cache.values})
    del states
    checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for entry in ring:
            entry['keys'].fill_(float('nan'));entry['values'].fill_(float('nan'))
            q=reference(entry);expected=(q.clone(),entry['keys'].clone(),entry['values'].clone())
            entry['keys'].fill_(float('nan'));entry['values'].fill_(float('nan'))
            actual=(candidate(entry),entry['keys'],entry['values'])
            mismatches=[int((u.view(torch.int16)!=v.view(torch.int16)).sum()) for u,v in zip(actual,expected,strict=True)]
            assert actual[0].stride()==q.stride(),(actual[0].stride(),q.stride())
            row={'layer':entry['layer'],'mismatches_q_k_v':mismatches,'strides_exact':True};checks.append(row)
            print('CHECK',row,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'tokens':160,'checks':checks,'rows':[],
        'method':'Real 160-token padded prefill states from all 36 layers. Original q/k norms, rotary and index_copy versus one fused Triton preparation kernel. Exact Q strides and bitwise full KV buffers including poisoned unused slots. Rotating all layers, timings are per-layer preparation only, not TTFA.'}
    path.write_text(json.dumps(result,indent=2)+'\n')
    if any(any(c['mismatches_q_k_v']) for c in checks):raise RuntimeError('Numerical change: do not claim an exact prefill replacement')
    for repeat in range(a.rounds):
        order=('control','candidate') if repeat%2==0 else ('candidate','control')
        timing={n:measure(reference if n=='control' else candidate,ring) for n in order}
        result['rows'].append({'round':repeat,'order':order,'us':timing});print('ROUND',repeat,timing,flush=True)
    result['median_us']={n:statistics.median(r['us'][n] for r in result['rows']) for n in ('control','candidate')}
    result['median_paired_gain_us']=statistics.median(r['us']['control']-r['us']['candidate'] for r in result['rows'])
    path.write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',result['median_us'],flush=True);engine.codec.close()


if __name__=='__main__':main()
