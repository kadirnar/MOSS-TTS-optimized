"""Cached LFQ projection and persistent, CUDA-graphed causal codec state."""
from contextlib import ExitStack, nullcontext
import types
import math

import torch
from torch.nn.utils import parametrize

from moss_audio_tokenizer.modeling_moss_audio_tokenizer import StreamingModule
from .kernels import codebook_sum, codec_attention


def _fast_attention(self,query,key,value):
    if self._streaming_state is None:
        return self._original_forward(query,key,value)
    if query.shape[0]!=1 or self.weights_per_step:
        raise ValueError("Triton codec attention requires batch 1 with shared projection weights")
    projected=self.in_projs[0](query)
    x=codec_attention(projected,self._cos_sin,self._streaming_state,self.num_heads)
    return self.out_projs[0](x)


@torch.inference_mode()
def fuse_codec_attention(codec):
    ds=torch.arange(32,device=codec.device,dtype=torch.float32)
    freq=torch.exp(ds*(-math.log(10000)*2/64))
    pos=torch.arange(32768,device=codec.device,dtype=torch.float32)
    angle=pos[:,None]*freq[None,:]
    cs=torch.cat([angle.cos(),angle.sin()],-1)
    for module in codec.decoder.modules():
        if module.__class__.__name__=='MossAudioTokenizerMultiheadAttention':
            if module.embed_dim//module.num_heads!=64 or module.rope.max_period!=10000:
                raise ValueError('Triton codec kernels require 64-dimensional heads and the original rotary period')
            if not hasattr(module,'_original_forward'):
                module._original_forward=module.forward
                module.register_buffer('_cos_sin',cs)
                module.forward=types.MethodType(_fast_attention,module)


@torch.inference_mode()
def cache_codebooks(codec):
    quantizer = codec.quantizer
    if getattr(quantizer, "_projected_codebooks", None) is not None:
        return
    # Freeze weight normalization in inference; retain the encoder's weights.
    for module in quantizer.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    indexes = torch.arange(quantizer.codebook_size, device=codec.device).view(1,-1)
    with torch.backends.cudnn.flags(allow_tf32=False):
        tables = [q.decode_code(indexes).squeeze(0).T.contiguous() for q in quantizer.quantizers]
    quantizer.register_buffer("_projected_codebooks", torch.stack(tables))
    def decode_codes(self, codes):
        with torch.autocast("cuda", enabled=False):
            emb = codebook_sum(codes, self._projected_codebooks)
            return self.output_proj(emb)
    quantizer.decode_codes = types.MethodType(decode_codes, quantizer)


class StreamingCodec:
    """One sequential stream at a time. Reset before each new utterance.

    Graph inputs/outputs and KV state are owned by this instance. Returned audio
    is cloned because the next replay overwrites the static graph output.
    """
    def __init__(self, codec, *, graph=True, bf16=False, cached_codebooks=True, triton_attention=True):
        self.codec = codec
        self.graph_enabled = graph
        self.bf16 = bf16
        if cached_codebooks:
            cache_codebooks(codec)
        if triton_attention:
            fuse_codec_attention(codec)
        if bf16:
            # Materialize weight conversion once; otherwise autocast records
            # conversion of every FP32 projection on every graph replay.
            for module in codec.decoder.modules():
                if isinstance(module,torch.nn.Linear):module.to(torch.bfloat16)
        self.stack = ExitStack()
        for module in codec.decoder:
            if isinstance(module, StreamingModule):
                self.stack.enter_context(module.streaming(1))
        self.states = [m._streaming_state for m in codec.decoder.modules() if isinstance(m, StreamingModule) and m._streaming_state is not None]
        if bf16:
            for state in self.states:
                if getattr(state, "kv_cache", None) is not None:
                    state.kv_cache.cache = state.kv_cache.cache.to(torch.bfloat16)
        self.reset_mask = torch.ones(1, dtype=torch.bool, device=codec.device)
        self.codes = torch.zeros((32,1,1), dtype=torch.long, device=codec.device)
        self.lengths = torch.ones(1, dtype=torch.long, device=codec.device)
        self.graph = None
        self.frames=0

    def context(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16)

    @torch.inference_mode()
    def warmup(self):
        with self.context():
            for _ in range(3):
                self.codec._decode_frame(self.codes, self.lengths)
        torch.cuda.synchronize()
        self.reset()
        if self.graph_enabled:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), self.context():
                for _ in range(3):
                    self.codec._decode_frame(self.codes, self.lengths)
            torch.cuda.current_stream().wait_stream(stream)
            self.reset()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), self.context():
                self.audio = self.codec._decode_frame(self.codes, self.lengths).audio
        self.reset()

    @torch.inference_mode()
    def reset(self):
        self.frames=0
        offsets=[]
        for state in self.states:
            if hasattr(state,'offset'):offsets.append(state.offset)
            if hasattr(state,'offsets'):offsets.append(state.offsets)
            if getattr(state,'kv_cache',None) is not None:
                offsets.append(state.kv_cache.end_offset)
            if hasattr(state,'offset_cpu'):state.offset_cpu=0
        # This class never changes exec_mask. Reset all device counters in one
        # multi-tensor launch instead of hundreds of tiny torch.where kernels.
        torch._foreach_zero_(offsets)

    @torch.inference_mode()
    def decode(self, codes):
        if codes.shape != self.codes.shape:
            raise ValueError(f"Expected {tuple(self.codes.shape)}, got {tuple(codes.shape)}")
        if self.frames>=4096:
            raise ValueError("Codec rotary table capacity exceeded (327.68 seconds)")
        self.frames+=1
        if self.graph is not None:
            self.codes.copy_(codes)
            self.graph.replay()
            return self.audio.clone()
        with self.context():
            return self.codec._decode_frame(codes, self.lengths).audio

    def close(self):
        self.stack.close()
