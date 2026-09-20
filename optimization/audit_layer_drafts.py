"""Measure layer-skipping draft cost and joint speculative acceptance.

This is an audit, not a speculative serving implementation. Drafts have their
own full-prefix KV state, then consume target-generated histories. Probability
comparisons include every active audio codebook and the unforced text slot.
"""
import argparse
import json
import math
import time

import numpy as np
import soundfile as sf
import torch
from transformers.cache_utils import StaticCache

from .common import RESULTS,load_models,stats
from .streaming import StreamingTTS
from .llm import sample_topk
from .reference_encoder import ReferenceEncoder
from .calibrated_backend import install_calibrated,enable_grouped_activation
from .dp4a_fusions import enable_fusions
from .dp4a_gateup import enable_gateup
from .dp4a_packing import install_packing
from .attention_quant import enable_attention_quant
from .attention_native import enable_native_attention
from .dp4a_gateup_quant import enable_gateup_quant
from .dp4a_scaled import enable_scaled
from .dp4a_norm_projection import enable_norm_projection


def distribution(logits,temperature,top_k,top_p):
    values,index=logits.float().div(temperature).topk(min(top_k,logits.shape[-1]),dim=-1)
    probability=values.softmax(-1)
    probability=probability.masked_fill(probability.cumsum(-1)-probability>top_p,0)
    probability=probability/probability.sum(-1,keepdim=True)
    return torch.zeros_like(logits,dtype=torch.float32).scatter_(-1,index,probability)


def joint_acceptance(p,q,samples,seed):
    generator=torch.Generator(device=q.device).manual_seed(seed)
    drawn=torch.multinomial(q,samples,replacement=True,generator=generator)
    a=p.gather(-1,drawn);b=q.gather(-1,drawn)
    accepted=(a.log()-b.log()).sum(0).exp().clamp_max(1)
    overlap=torch.minimum(p,q).sum(-1)
    # Joint acceptance cannot exceed either proposal's probability of lying
    # in the other's support. The product of marginal overlaps is a lower
    # bound, not an upper bound, and is labelled accordingly.
    upper=torch.minimum((q*(p>0)).sum(-1).prod(),(p*(q>0)).sum(-1).prod())
    return {'joint_acceptance_mc':accepted.mean().item(),
            'mc_standard_error':(accepted.std()/math.sqrt(samples)).item(),
            'support_upper_bound':upper.item(),'product_overlap_lower_bound':overlap.prod().item(),
            'mean_channel_overlap':overlap.mean().item(),'channel_overlaps':overlap.tolist()}


def probabilities(text,audio,length,delay):
    active=(torch.arange(32,device=audio.device)<length)
    if delay>=0:active &= torch.arange(32,device=audio.device)>=delay
    rows=distribution(audio[active],1.7,25,.8)
    if delay<0:
        t=torch.zeros((1,1024),device=audio.device)
        t[:,:2]=distribution(text,1.5,2,1.)
        rows=torch.cat((t,rows))
    return rows


def new_cache(fast):
    cfg=fast.model.config.language_config
    cache=StaticCache(cfg,max_cache_len=fast.max_length)
    cache.early_initialization(1,cfg.num_key_value_heads,cfg.head_dim,torch.bfloat16,torch.device('cuda'))
    return cache


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--samples',type=int,default=8192)
    a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.samples<1024:
        raise ValueError('Safe tag and at least 1024 Monte Carlo draws required')
    destination=RESULTS/f'layer_draft_audit_{a.tag}.json'
    if destination.exists():raise FileExistsError('Preserve earlier results')
    identity=torch.tensor([[.25,.75],[.5,.5]],device='cuda')
    assert joint_acceptance(identity,identity,1024,1)['joint_acceptance_mc']==1
    disjoint=torch.tensor([[1.,0.]],device='cuda')
    assert joint_acceptance(disjoint,1-disjoint,1024,1)['joint_acceptance_mc']==0
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True)
    fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10','dp4a')
    enable_grouped_activation(fast);enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json')
    enable_attention_quant(fast);enable_native_attention(fast)
    enable_gateup_quant(fast);enable_scaled(fast);enable_norm_projection(fast)
    original_layers=model.language_model.layers
    for layer in original_layers:layer.self_attn._decode_capacity=256
    fast.warmup();fast.capture_prefill((160,256))
    encoder=ReferenceEncoder(processor,buckets=(40,64));encoder.warmup()
    manifest=json.loads((RESULTS/'quality_suite/expanded_g32_v1/manifest.json').read_text())
    references={};cases=[]
    for record in manifest['records']:
        if not record['id'].endswith(('_0','_6')):continue
        voice=record['voice']
        if voice not in references:
            wav,sr=sf.read(record['reference'],dtype='float32')
            references[voice]=encoder.encode(torch.from_numpy(wav).reshape(1,-1),sr)
        inputs=processor([[processor.build_user_message(text=record['text'],reference=[references[voice]],language=record['language'])]],mode='generation').to('cuda')
        ids=inputs.input_ids;prompt=ids.shape[1]
        assert prompt+32<=256
        torch.manual_seed(record['seed'])
        logits,_=fast.prefill(ids)
        banned=[model.config.pad_token_id,model.config.audio_assistant_gen_slot_token_id,
                model.config.audio_assistant_delay_slot_token_id,model.config.audio_end_token_id,model.config.im_end_token_id]
        logits=logits.clone();logits[:,banned]=-torch.inf
        first=sample_topk(logits,1.5,50,1.)
        assert first.item()==model.config.audio_start_token_id,record['id']
        token=torch.full((1,1,33),1024,device='cuda',dtype=torch.long);token[...,0]=first
        steps=[];length=1;delay=-1
        for step in range(32):
            current=token.clone()
            token=fast.step(current,prompt+step,length,delay)
            steps.append({'input':current.cpu(),'length':length,'delay':delay,
                          'probabilities':probabilities(fast.text_logits,fast.audio_logits,length,delay).cpu()})
            text=int(token[0,0,0])
            if text in (model.config.audio_assistant_gen_slot_token_id,model.config.audio_assistant_delay_slot_token_id):length+=1
            if delay<0 and text==model.config.audio_assistant_delay_slot_token_id:delay=0
            if delay>=0:delay+=1
        cases.append({'id':record['id'],'text':record['text'],'voice':voice,'seed':record['seed'],
                      'prompt_ids':ids.cpu(),'steps':steps})
        print('TARGET',record['id'],prompt,flush=True)
    torch.save(cases,RESULTS/f'layer_draft_histories_{a.tag}.pt')
    target_cache=fast.cache;target_graph=fast.graph;target_prefill=fast.prefill_graphs
    configurations={'full_control':list(range(36))}
    for count in (34,30,24,18,12):
        configurations[f'uniform_{count}']=sorted(set(np.rint(np.linspace(0,35,count)).astype(int).tolist()))
    configurations['middle_gap_24']=list(range(12))+list(range(24,36))
    result={'codebooks':32,'torch':torch.__version__,'monte_carlo_draws':a.samples,
            'scope':'Eight target-generated first-frame histories, four cloned voices, two texts per voice. Drafts have independent caches and prefill with their own retained layers; decode is teacher-forced on target history. Joint acceptance is an on-history diagnostic, not measured speculative throughput or TTFA. No block verifier or trained draft is implemented.',
            'configurations':{},'history_file':f'layer_draft_histories_{a.tag}.pt'}
    for name,keep in configurations.items():
        if name=='full_control':fast.cache=target_cache;fast.graph=target_graph;fast.prefill_graphs=target_prefill
        else:
            model.language_model.layers=torch.nn.ModuleList([original_layers[i] for i in keep])
            fast.cache=new_cache(fast);fast.graph=None;fast.prefill_graphs={}
            fast.warmup();fast.capture_prefill((160,256))
        rows=[];prefill_times=[];decode_times=[]
        for case_index,case in enumerate(cases):
            ids=case['prompt_ids'].cuda();prompt=ids.shape[1]
            torch.cuda.synchronize();start=time.perf_counter();fast.prefill(ids);torch.cuda.synchronize()
            prefill_times.append((time.perf_counter()-start)*1000)
            for step,entry in enumerate(case['steps']):
                token=entry['input'].cuda()
                start_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start_event.record();fast.step(token,prompt+step,entry['length'],entry['delay']);end_event.record();end_event.synchronize()
                decode_times.append(start_event.elapsed_time(end_event))
                q=probabilities(fast.text_logits,fast.audio_logits,entry['length'],entry['delay'])
                target=entry['probabilities'].cuda()
                if name=='full_control':assert torch.equal(q,target),(case['id'],step)
                row={'case':case['id'],'step':step,'active_audio_channels':q.shape[0]-(entry['delay']<0),
                     **joint_acceptance(target,q,a.samples,28000+case_index*32+step)}
                rows.append(row)
        # Continuous graph replay avoids per-step audit copies/statistics in
        # the standalone draft-cost estimate. This includes its sampler.
        graph_times=[]
        for repeat in range(9):
            start_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(100):fast.graph.replay()
            end_event.record();end_event.synchronize()
            graph_times.append(start_event.elapsed_time(end_event)/100)
        entry={'layers':keep,'retained_layers':len(keep),'rows':rows,'prefill':stats(prefill_times),
               'audited_decode_gpu':stats(decode_times),'repeated_graph_gpu':stats(graph_times),
               'mean_joint_acceptance':float(np.mean([r['joint_acceptance_mc'] for r in rows])),
               'mean_support_upper_bound':float(np.mean([r['support_upper_bound'] for r in rows]))}
        result['configurations'][name]=entry
        print(name,{k:v for k,v in entry.items() if k not in ('rows','layers','audited_decode_gpu')},flush=True)
        destination.write_text(json.dumps(result,indent=2)+'\n')
        model.language_model.layers=original_layers
    fast.cache=target_cache;fast.graph=target_graph;fast.prefill_graphs=target_prefill
    for layer in original_layers:layer.self_attn._decode_capacity=None
    engine.codec.close()


if __name__=='__main__':main()
