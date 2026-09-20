"""Opt-in voice-prefix KV cache, with captured BF16 suffix prefills.

Only fixed conditioning before the text marker can be cached. The key covers
all 33 input channels. Generated audio and user text are never stored here.
Use one request at a time, like the owning FastLLM.
"""
from collections import OrderedDict
import torch


class PrefixPrefillCache:
    def __init__(self,llm,tokenizer,prefix_length=96,max_entries=16):
        if prefix_length<=0 or max_entries<=0:raise ValueError('Positive cache sizes required')
        self.llm=llm
        self.original_prefill=llm.prefill
        self.prefix_length=prefix_length
        self.max_entries=max_entries
        self.marker=tokenizer.encode('- Text:\n',add_special_tokens=False)
        self.entries=OrderedDict()
        self.graphs={}
        layers=llm.cache.layers
        self.prefix=torch.zeros((len(layers),2,1,8,prefix_length,128),device='cuda',dtype=torch.bfloat16)
        self.hits=0
        self.misses=0
        self.bypasses=0

    def _restore(self):
        for i,layer in enumerate(self.llm.cache.layers):
            layer.keys[:,:,:self.prefix_length].copy_(self.prefix[i,0])
            layer.values[:,:,:self.prefix_length].copy_(self.prefix[i,1])

    @torch.inference_mode()
    def warmup(self,buckets=(16,32,64,128,256)):
        attentions=[layer.self_attn for layer in self.llm.model.language_model.layers]
        try:
            for a in attentions:a._prefill_offset=self.prefix_length
            for n in buckets:
                if n+self.prefix_length>=self.llm.max_length:continue
                ids=torch.full((1,n,33),1024,device='cuda',dtype=torch.long)
                ids[...,0]=self.llm.cfg.pad_token_id
                position=torch.arange(self.prefix_length,self.prefix_length+n,device='cuda')
                last=torch.tensor([n-1],device='cuda')
                def forward():
                    self._restore()
                    return self.llm._prefill_forward(ids,position,last)
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=forward()
                self.graphs[n]=(graph,ids,position,last,out)
        finally:
            for a in attentions:a._prefill_offset=0

    @torch.inference_mode()
    def prefill(self,ids):
        if ids.shape[0]!=1 or ids.shape[1]>=self.llm.max_length:
            return self.original_prefill(ids)
        cpu_ids=ids.detach().cpu()
        text_ids=cpu_ids[0,:,0].tolist()
        markers=[i for i in range(len(text_ids)-len(self.marker)+1)
                 if text_ids[i:i+len(self.marker)]==self.marker]
        # If the marker cannot be verified, preserve normal prefill semantics.
        if not markers or markers[0]+len(self.marker)<self.prefix_length:
            self.bypasses+=1
            return self.original_prefill(ids)
        n=ids.shape[1]-self.prefix_length
        buckets=[b for b in self.graphs if b>=n]
        if n<=0 or not buckets:
            self.bypasses+=1
            return self.original_prefill(ids)
        key=cpu_ids[:,:self.prefix_length].contiguous().numpy().tobytes()
        cached=self.entries.get(key)
        if cached is None:
            out=self.original_prefill(ids)
            cached=torch.stack([torch.stack([layer.keys[:,:,:self.prefix_length],
                layer.values[:,:,:self.prefix_length]]) for layer in self.llm.cache.layers])
            self.entries[key]=cached
            if len(self.entries)>self.max_entries:self.entries.popitem(last=False)
            self.misses+=1
            return out
        self.entries.move_to_end(key)
        self.prefix.copy_(cached)
        graph,static_ids,position,last,out=self.graphs[min(buckets)]
        static_ids[:,:n].copy_(ids[:,self.prefix_length:])
        last.fill_(n-1)
        graph.replay()
        self.hits+=1
        return out

    def install(self):
        if not self.graphs:raise RuntimeError('Warm suffix graphs before installing')
        self.llm.prefill=self.prefill

    def stats(self):
        return {'hits':self.hits,'misses':self.misses,'bypasses':self.bypasses,'entries':len(self.entries),
            'prefix_tokens':self.prefix_length,'generated_audio_cached':False,'text_cached':False}
