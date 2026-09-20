"""Batch-one codec graphs with a shared frame clock and a first-frame specialization.

Only this wrapper owns streaming decoder state. Encoder and quantizer are left
alone. Per-layer device offsets are unused here; the immutable frame counter
determines each stage's offset and advances once after a complete decoder frame.
"""
import types
import torch
import torch.nn.functional as F

from .codec import StreamingCodec
from .codec_clock_kernels import attention


def _attention(self,query,key,value):
    if self._streaming_state is None:return self._clock_previous(query,key,value)
    if query.shape[0]!=1 or self.weights_per_step:raise ValueError('Shared clock requires batch one and fixed projection weights')
    owner=self._clock_owner
    kv_only=owner.capture_first and owner.kv_only and query.shape[1]==1
    if kv_only:projected=F.linear(query,self.in_projs[0].weight[self.embed_dim:])
    else:projected=self.in_projs[0](query)
    x=attention(projected,self._cos_sin,self._streaming_state,self.num_heads,owner.clock,
                first=owner.capture_first,kv_only=kv_only)
    return self.out_projs[0](x)


def _transformer(self,x,*args,**kwargs):
    if self._streaming_state is None:return self._clock_previous(x,*args,**kwargs)
    # This preset uses RoPE in attention, never additive positional embeddings.
    for layer in self.layers:x=layer(x,*args,**kwargs)
    return x


class ClockedStreamingCodec(StreamingCodec):
    def __init__(self,codec,*,first_frame=True,kv_only=False,graph=True):
        if kv_only and not first_frame:raise ValueError('KV-only mode requires first-frame specialization')
        modules=list(codec.decoder.modules())
        for module in modules:
            name=module.__class__.__name__
            if hasattr(module,'_clock_owner'):raise ValueError('Codec already has a clock owner')
            if name=='MossAudioTokenizerTransformer' and module.positional_embedding!='rope':
                raise ValueError('Shared codec clock requires rotary-only transformers')
            if name=='MossAudioTokenizerMultiheadAttention' and (module.weights_per_step or module.embed_dim//module.num_heads!=64):
                raise ValueError('Shared codec clock requires fixed weights and 64-dimensional heads')
        if any(p.dtype!=torch.float32 for p in codec.decoder.parameters()):raise ValueError('This codec preset retains FP32 decoder weights')
        super().__init__(codec,graph=graph)
        self.first_frame=bool(first_frame);self.kv_only=bool(kv_only);self.capture_first=False
        self.clock=torch.zeros(1,device=codec.device,dtype=torch.int64)
        self.first_graph=None;self.first_audio=None;self.modified=[]
        for module in modules:
            name=module.__class__.__name__
            replacement=_attention if name=='MossAudioTokenizerMultiheadAttention' else _transformer if name=='MossAudioTokenizerTransformer' else None
            if replacement is not None:
                module._clock_previous=module.forward;module._clock_owner=self
                module.forward=types.MethodType(replacement,module);self.modified.append(module)

    def _frame(self):
        output=self.codec._decode_frame(self.codes,self.lengths).audio
        self.clock.add_(1)
        return output

    @torch.inference_mode()
    def warmup(self):
        if self.graph is not None or self.first_graph is not None:raise RuntimeError('Codec graphs already captured')
        self.capture_first=False
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self._frame()
        torch.cuda.current_stream().wait_stream(stream);self.reset()
        if self.graph_enabled:
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.audio=self._frame()
        self.reset()
        if self.first_frame:
            self.capture_first=True
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.clock.zero_();self._frame()
            torch.cuda.current_stream().wait_stream(stream);self.reset()
            if self.graph_enabled:
                self.first_graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.first_graph):self.first_audio=self._frame()
        self.capture_first=False;self.reset()

    @torch.inference_mode()
    def reset(self):
        super().reset()
        self.clock.zero_()

    @torch.inference_mode()
    def decode(self,codes):
        if codes.shape!=self.codes.shape or codes.dtype!=torch.long or codes.device!=self.codes.device:
            raise ValueError('Expected one complete CUDA int64 frame with shape [32,1,1]')
        if self.frames>=4096:raise ValueError('Codec rotary table capacity exceeded (327.68 seconds)')
        first=self.first_frame and self.frames==0;self.codes.copy_(codes)
        if self.graph is not None:
            graph=self.first_graph if first else self.graph;graph.replay()
            output=self.first_audio if first else self.audio
        else:
            self.capture_first=first
            try:output=self._frame()
            finally:self.capture_first=False
        self.frames+=1
        return output.clone()

    def close(self):
        torch.cuda.current_stream(self.clock.device).synchronize()
        for module in self.modified:
            module.forward=module._clock_previous
            del module._clock_previous,module._clock_owner
        self.modified.clear();super().close()
