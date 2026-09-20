"""Opt-in vLLM-Omni v1 codec adapter using our validated FP32 causal decoder.

The upstream 0.28 streaming session requires a pool API only its v2 codec has.
This adapter is deliberately limited to one active stream and all 32 codebooks.
"""
import torch
from moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerModel
from .codec import StreamingCodec


class SingleStreamSession:
    def __init__(self, codec):
        self.engine = StreamingCodec(codec)
        self.engine.warmup()
        self.leased = False

    def acquire(self):
        if self.leased:
            return None
        self.engine.reset()
        self.leased = True
        return 0

    def release(self, slot, *, state_already_reset=False):
        if slot != 0:
            raise ValueError('Only one codec stream is supported')
        self.engine.reset()
        self.leased = False

    def reset_slots(self, slots):
        if any(slot != 0 for slot in slots):
            raise ValueError('Only one codec stream is supported')
        self.engine.reset()

    @torch.inference_mode()
    def step(self, slot_codes, *, terminal_slots=None):
        if not slot_codes:
            return {}
        if set(slot_codes) != {0} or not self.leased:
            raise RuntimeError('Invalid or unleased codec stream')
        codes = slot_codes[0]
        if codes.shape[0] != 32 or torch.any((codes<0)|(codes>=1024)):
            raise ValueError('Expected all 32 valid codebooks')
        chunks = [self.engine.decode(codes[:,i:i+1].reshape(32,1,1)).reshape(-1)
                  for i in range(codes.shape[1])]
        audio = torch.cat(chunks).float().cpu()
        if terminal_slots:
            self.engine.reset()
        return {0:audio}

    def close(self):
        self.engine.close()


def install(cls):
    @torch.inference_mode()
    def load_weights(self, weights):
        if self._n_vq != 32 or self._stream_state_capacity != 1:
            raise ValueError('Optimized v1 codec requires 32 codebooks and max_num_seqs=1')
        for _ in weights:
            pass
        device = self.vllm_config.device_config.device
        self._codec = MossAudioTokenizerModel.from_pretrained(self._codec_path,
            dtype=torch.float32, device_map=str(device)).eval()
        self._n_channels = 1
        self._sr_tensor = torch.tensor(24000,dtype=torch.int32)
        self._stream_session = SingleStreamSession(self._codec)
        return {f'_codec.{name}' for name,_ in self._codec.named_parameters()}
    cls.load_weights = load_weights
