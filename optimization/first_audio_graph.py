"""Experimental GPU-only execution of the 32 steps after audio_start.

The delay schedule cannot emit audio_end before these 32 steps finish. Every
step still runs the complete LLM and full-shape sampler, and every first PCM
frame retains all 32 codebooks. This object shares the owner's sequential KV
state; it must never overlap another request or graph replay.
"""
import time
import torch
import triton
import triton.language as tl


@triton.jit
def _advance(NEXT,IDS,POS,LENGTH,DELAY,HISTORY,STATUS,STEP:tl.constexpr,DELAY_ID:tl.constexpr):
    i=tl.arange(0,64)
    values=tl.load(NEXT+i,i<33,0)
    tl.store(IDS+i,values,i<33)
    tl.store(HISTORY+STEP*33+i,values,i<33)
    text=tl.load(NEXT)
    position=tl.load(POS)+1
    length=tl.load(LENGTH)+1
    delay=tl.load(DELAY)
    delay=tl.where((delay<0)&(text==DELAY_ID),0,delay)
    delay=tl.where(delay>=0,delay+1,delay)
    tl.store(POS,position);tl.store(LENGTH,length);tl.store(DELAY,delay)
    tl.store(STATUS,text);tl.store(STATUS+1,length);tl.store(STATUS+2,delay);tl.store(STATUS+3,position)


class FirstAudioGraph:
    def __init__(self,llm,capacity,*,head_prefixes=True,save_logits=False):
        if llm.graph is None or capacity%32 or not 32<=capacity<=llm.max_length:
            raise ValueError('A warmed LLM and valid context capacity are required')
        self.llm=llm;self.capacity=capacity;self.head_prefixes=head_prefixes;self.save_logits=save_logits
        device=llm.ids.device
        self.history=torch.empty((32,33),device=device,dtype=torch.long)
        self.status=torch.empty(4,device=device,dtype=torch.long)
        self.text_history=torch.empty((32,2),device=device,dtype=torch.bfloat16) if save_logits else None
        self.audio_history=torch.empty((32,32,1024),device=device,dtype=torch.bfloat16) if save_logits else None
        self.graph=None;self.metadata={}

    def _body(self):
        llm=self.llm
        for step in range(32):
            llm._audio_head_count=((step//8)+1)*8 if self.head_prefixes else 32
            ids,text,audio=llm._decode()
            if self.save_logits:
                self.text_history[step].copy_(text.flatten());self.audio_history[step].copy_(audio)
            _advance[(1,)](ids,llm.ids,llm.position,llm.audio_length,llm.delay_length,
                           self.history,self.status,step,llm.cfg.audio_assistant_delay_slot_token_id,num_warps=4)

    def _initialize(self,ids,position,delay=-1):
        llm=self.llm;llm.ids.copy_(ids);llm.position.fill_(position)
        llm.audio_length.fill_(1);llm.delay_length.fill_(delay)

    @torch.inference_mode()
    def warmup(self):
        if self.graph is not None:raise RuntimeError('First-audio graph already captured')
        llm=self.llm;layers=llm.model.language_model.layers
        previous=[getattr(l.self_attn,'_decode_capacity',None) for l in layers]
        previous_head=getattr(llm,'_audio_head_count',32)
        saved=[v.clone() for v in (llm.ids,llm.position,llm.audio_length,llm.delay_length)]
        rng=torch.cuda.get_rng_state(llm.ids.device)
        first=torch.full_like(llm.ids,1024);first[...,0]=llm.cfg.audio_start_token_id
        torch.cuda.synchronize();before=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved();start=time.perf_counter()
        try:
            for layer in layers:layer.self_attn._decode_capacity=self.capacity
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._initialize(first,0);self._body()
            torch.cuda.current_stream().wait_stream(stream)
            self._initialize(first,0)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self._body()
            torch.cuda.synchronize()
            self.metadata={'capacity':self.capacity,'steps':32,'codebooks':32,'head_prefixes':self.head_prefixes,
                           'save_logits':self.save_logits,'capture_seconds':time.perf_counter()-start,
                           'additional_live_bytes':torch.cuda.memory_allocated()-before,
                           'additional_reserved_bytes':torch.cuda.memory_reserved()-reserved}
        finally:
            for layer,cap in zip(layers,previous,strict=True):layer.self_attn._decode_capacity=cap
            llm._audio_head_count=previous_head
            for destination,source in zip((llm.ids,llm.position,llm.audio_length,llm.delay_length),saved,strict=True):destination.copy_(source)
            torch.cuda.set_rng_state(rng,llm.ids.device)

    def run(self,ids,position,*,delay=-1):
        if self.graph is None:raise RuntimeError('Warm the first-audio graph before use')
        if not 0<=position or position+32>self.capacity:raise ValueError('Initial interval exceeds captured context capacity')
        if ids.shape!=(1,1,33) or ids.dtype!=torch.long or ids.device!=self.llm.ids.device:
            raise ValueError('Expected a CUDA int64 audio-start row with all 32 channels')
        if delay not in (-1,0):raise ValueError('Only fresh audio or an immediate-delay validation probe is supported')
        self._initialize(ids,position,delay);self.graph.replay()
        return self.history,self.status


@torch.inference_mode()
def enable(engine,capacities=(128,256,512,1024),*,head_prefixes=None):
    if engine._busy or getattr(engine,'_first_audio_graphs',None):
        raise RuntimeError('Install initial-audio graphs once, before accepting requests')
    llm=engine.llm
    if llm.graph is None or any(not l.self_attn._triton_decode or l.self_attn._decode_backend is not None for l in llm.model.language_model.layers):
        raise ValueError('Initial-audio graphs require warmed custom decode attention')
    if head_prefixes is None:head_prefixes=bool(getattr(llm,'_audio_head_buckets',None))
    graphs={}
    for cap in sorted(set(capacities)):
        if cap>llm.max_length:continue
        graph=FirstAudioGraph(llm,cap,head_prefixes=head_prefixes);graph.warmup();graphs[cap]=graph
    if not graphs:raise ValueError('No supported initial-audio capacity')
    engine._first_audio_graphs=graphs
    return {'codebooks':32,'steps':32,'graphs':[g.metadata for g in graphs.values()]}
