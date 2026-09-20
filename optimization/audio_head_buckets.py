"""Experimental graph prefixes for heads masked by the existing delay schedule.

All 32 codebooks remain required for each codec frame. Only unavailable future
heads are skipped; logits and sampling keep the full 32-row shape. Supported
prefixes passed the initial cuBLAS arithmetic audit; smaller prefixes did not.
"""
from collections import Counter

import torch


class DecodeAudioHeadBuckets:
    def __init__(self, llm, contexts):
        if contexts.llm is not llm or llm.graph is None:
            raise ValueError('Captured context graphs for this LLM are required')
        if getattr(llm, '_audio_head_buckets', None):
            raise ValueError('Audio head buckets are already installed')
        self.llm = llm
        self.contexts = tuple(sorted(contexts.graphs))
        self.head_counts = (8, 16, 24, 32)
        self.graphs = {(cap, 32): pair for cap, pair in contexts.graphs.items()}
        self.original_step = contexts.original_step
        self.usage = Counter()

    @torch.inference_mode()
    def warmup(self):
        llm = self.llm
        saved_position = llm.position.clone()
        llm.position.zero_()
        try:
            for cap in self.contexts:
                for layer in llm.model.language_model.layers:
                    layer.self_attn._decode_capacity = cap
                for count in self.head_counts[:-1]:
                    llm._audio_head_count = count
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            llm._decode()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        outputs = llm._decode()
                    self.graphs[cap, count] = (graph, outputs)
        finally:
            llm._audio_head_count = 32
            for layer in llm.model.language_model.layers:
                layer.self_attn._decode_capacity = None
            llm.position.copy_(saved_position)

    def step(self, ids, position, audio_length, delay_length):
        if not 0 <= position < self.llm.max_length:
            raise ValueError('KV capacity exceeded')
        cap = min(c for c in self.contexts if position < c)
        count = min(c for c in self.head_counts if c >= min(audio_length, 32))
        graph, outputs = self.graphs[cap, count]
        self.llm.graph = graph
        self.llm.next_ids, self.llm.text_logits, self.llm.audio_logits = outputs
        self.usage[cap, count] += 1
        return self.original_step(ids, position, audio_length, delay_length)

    def install(self):
        if len(self.graphs) != len(self.contexts) * len(self.head_counts):
            raise RuntimeError('Warm all head/context combinations first')
        self.llm._audio_head_buckets = self
        self.llm.step = self.step

    def stats(self):
        return {'contexts': self.contexts, 'head_prefixes': self.head_counts,
                'uses': {f'{cap}/{count}': n for (cap, count), n in self.usage.items()}}
