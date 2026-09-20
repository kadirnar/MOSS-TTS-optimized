"""Actual first-audio graph probes, with separate uninstrumented control.

Captures one trace slot per projection and audio step. Probes are diagnostic;
their timing and register effects are measured, never subtracted as speedups.
"""
import argparse
import contextlib
import hashlib
import json
import statistics

import torch

from .common import RESULTS,stats
from .benchmark_first_audio_graph import build_engine
from .first_audio_graph import FirstAudioGraph
from .projection_timestamps_compact import configured
from .dp4a_norm_pdl import SELECTED
from .short_scales import PLAN
from .benchmark_projection_timestamps import summarize


class Probe:
    def __init__(self,llm,family,mode,stride=16):
        self.llm=llm;self.family=family;self.mode=mode;self.stride=stride
        self.traces={};self.functions={};self.counts={};self.old_norm=llm._bulk_norm_linear
        self.old_down=[l.mlp._bulk_down_linear for l in llm.model.language_model.layers]
        for i,layer in enumerate(llm.model.language_model.layers):
            module=layer.self_attn if family=='qkv' else layer.mlp
            w=getattr(module,'_quant_'+family);n=w.shape[0]//(2 if family=='up' else 1)
            rows=(PLAN if family=='down' else SELECTED)[family]['rows'];key=w.data_ptr()
            trace=torch.zeros((32,n//rows,4),device=w.device,dtype=torch.int64)
            self.traces[(i,family)]=trace;self.counts[key]=0
            self.functions[key]=[configured('projection' if family=='down' else 'norm',trace[j],mode=mode,stride=stride) for j in range(32)]

    def norm(self,x,res,nw,eps,w,s,**kwargs):
        key=w.data_ptr()
        if key not in self.functions:return self.old_norm(x,res,nw,eps,w,s,**kwargs)
        slot=self.counts[key]%32;self.counts[key]+=1
        return self.functions[key][slot](x,res,nw,eps,w,s,**kwargs)

    def down(self,x,w,s,**kwargs):
        key=w.data_ptr();slot=self.counts[key]%32;self.counts[key]+=1
        return self.functions[key][slot](x,w,s,**kwargs)

    @contextlib.contextmanager
    def active(self):
        try:
            if self.family=='down':
                for layer in self.llm.model.language_model.layers:layer.mlp._bulk_down_linear=self.down
            else:self.llm._bulk_norm_linear=self.norm
            yield
        finally:
            self.llm._bulk_norm_linear=self.old_norm
            for layer,old in zip(self.llm.model.language_model.layers,self.old_down,strict=True):layer.mlp._bulk_down_linear=old


def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=8);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<3:raise ValueError('Safe tag and three rounds required')
    path=RESULTS/f'projection_waits_full_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    engine,_=build_engine(bulk=True,prefill_qkv=True,prefill_pointwise=True);fast=engine.llm
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);ids=fixture['inputs']['input_ids'].cuda();prompt=ids.shape[1]
    assert prompt+32<=256
    first=torch.full_like(fast.ids,1024);first[...,0]=fast.cfg.audio_start_token_id
    control=FirstAudioGraph(fast,256);control.warmup();graphs={'control':control};probes={}
    for name,family,mode in (('up_wait','up',1),('qkv_full','qkv',2),('down_wait','down',1)):
        probe=Probe(fast,family,mode);probes[name]=probe
        with probe.active():
            graph=FirstAudioGraph(fast,256);graph.warmup();graphs[name]=graph
        assert all(n==64 for n in probe.counts.values()),probe.counts
        print('CAPTURED',name,flush=True)
    rows=[];checks=[];names=tuple(graphs)
    # Compare full cache contents once, then IDs/final logits/status/RNG in
    # every timing round. Reads occur after the timed CUDA interval.
    baseline=None
    for name,graph in graphs.items():
        fast.prefill(ids);torch.cuda.manual_seed(831);history,status=graph.run(first,prompt)
        state=[digest(t) for t in (history,status,fast.text_logits,fast.audio_logits)]
        state.extend(digest(t) for layer in fast.cache.layers for t in (layer.keys,layer.values))
        state.append(digest(torch.cuda.get_rng_state()))
        if baseline is None:baseline=state
        else:assert state==baseline,name
        checks.append({'mode':name,'all_72_kv_buffers_ids_logits_status_rng_exact':True})
    result={'codebooks':32,'scope':'Actual 32-step first-audio graph on the frozen cloned-voice prompt. Probes retain original G32 arithmetic. Timings are graph GPU intervals, not end-to-end TTFA. Counts/timestamps are sampled lane observations and include probe perturbation.',
        'checks':checks,'rows':rows,'complete':False}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    save()
    for repeat in range(-1,a.rounds):
        order=names[repeat%len(names):]+names[:repeat%len(names)]
        if repeat%2:order=tuple(reversed(order))
        ms={};states={}
        for name in order:
            fast.prefill(ids);torch.cuda.manual_seed(8100+repeat);torch.cuda.synchronize()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();history,status=graphs[name].run(first,prompt);end.record();end.synchronize()
            ms[name]=start.elapsed_time(end)
            states[name]=[digest(t) for t in (history,status,fast.text_logits,fast.audio_logits,torch.cuda.get_rng_state())]
        assert all(state==states['control'] for state in states.values())
        if repeat>=0:rows.append({'round':repeat,'order':order,'ms':ms,'ids_logits_status_rng_exact':True})
        save();print('ROUND',repeat,ms,flush=True)
    result['timing']={name:stats([r['ms'][name] for r in rows]) for name in names}
    result['median_probe_overhead_ms']={name:statistics.median(r['ms'][name]-r['ms']['control'] for r in rows) for name in probes}
    raw={name:{f'{i:02d}_{stage}':t.cpu() for (i,stage),t in probe.traces.items()} for name,probe in probes.items()}
    result['lane_intervals']={}
    result['per_step']={}
    for name,probe in probes.items():
        flat={(i*32+j,stage):raw[name][f'{i:02d}_{stage}'][j] for (i,stage) in probe.traces for j in range(32)}
        result['lane_intervals'][name]=summarize(flat,probe.mode,probe.stride)
        result['per_step'][name]=[summarize({(i,stage):raw[name][f'{i:02d}_{stage}'][j] for (i,stage) in probe.traces},probe.mode,probe.stride) for j in range(32)]
    torch.save(raw,RESULTS/f'projection_waits_full_{a.tag}_raw.pt')
    result['complete']=True;save();print('SUMMARY',result['median_probe_overhead_ms'],result['lane_intervals'],flush=True)
    engine.codec.close()


if __name__=='__main__':main()
