"""Opt-in decode graphs that omit attention blocks beyond a safe position cap."""
from collections import Counter
import torch


class DecodeContextBuckets:
    def __init__(self,llm,buckets=(128,256,512)):
        if llm.graph is None:raise ValueError('Capture the full-capacity decode graph first')
        if getattr(llm,'_decode_context_buckets',None):raise ValueError('Decode buckets are already installed')
        layers=llm.model.language_model.layers
        if any(not a.self_attn._triton_decode or a.self_attn._decode_backend is not None for a in layers):
            raise ValueError('Context buckets require the custom Triton attention backend')
        self.llm=llm
        self.buckets=tuple(sorted(set(buckets)))
        block=layers[0].self_attn._decode_block
        if any(b<=0 or b>=llm.max_length or b%block for b in self.buckets):raise ValueError('Invalid context bucket')
        self.graphs={llm.max_length:(llm.graph,(llm.next_ids,llm.text_logits,llm.audio_logits))}
        self.usage=Counter()
        self.original_step=llm.step

    @torch.inference_mode()
    def warmup(self):
        llm=self.llm
        saved_position=llm.position.clone()
        llm.position.zero_()
        try:
            for cap in self.buckets:
                for layer in llm.model.language_model.layers:layer.self_attn._decode_capacity=cap
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):llm._decode()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs=llm._decode()
                self.graphs[cap]=(graph,outputs)
        finally:
            for layer in llm.model.language_model.layers:layer.self_attn._decode_capacity=None
            llm.position.copy_(saved_position)

    def step(self,ids,position,audio_length,delay_length):
        if not 0<=position<self.llm.max_length:raise ValueError('KV capacity exceeded')
        cap=min(b for b in self.graphs if position<b)
        graph,outputs=self.graphs[cap]
        self.llm.graph=graph
        self.llm.next_ids,self.llm.text_logits,self.llm.audio_logits=outputs
        self.usage[cap]+=1
        return self.original_step(ids,position,audio_length,delay_length)

    def install(self):
        if any(b not in self.graphs for b in self.buckets):raise RuntimeError('Warm every bucket before installing')
        self.llm._decode_context_buckets=self
        self.llm.step=self.step

    def stats(self):
        return {'capacities':sorted(self.graphs),'uses':dict(self.usage)}
