"""Continuous, band-limited conversion from the native codec to 48 kHz."""

import torch
from torchaudio.transforms import Resample


def output_resampler():
    """Build a reusable CPU kernel; construction belongs to model startup."""
    return Resample(
        24000,
        48000,
        lowpass_filter_width=32,
        rolloff=0.95,
        resampling_method="sinc_interp_kaiser",
        beta=8.6,
    )


class StreamingResampler:
    """Keep both filter history and lookahead across arbitrary input chunks.

    The first output withholds the filter's right context. flush() releases
    that tail with zero padding, matching resampling the whole utterance.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.width = kernel.width
        self.pending = torch.zeros(self.width, dtype=torch.float32)
        self.closed = False

    def push(self, pcm):
        if self.closed:
            raise RuntimeError("Resampler is already flushed")
        if pcm.ndim != 1 or pcm.device.type != "cpu" or pcm.dtype != torch.float32:
            raise ValueError("Expected one-dimensional CPU float32 PCM")
        self.pending = torch.cat((self.pending, pcm))
        count = max(0, self.pending.numel() - 2 * self.width)
        if not count:
            return self.pending.new_empty(0)
        converted = self.kernel(self.pending)
        result = converted[2 * self.width : 2 * (self.width + count)].clone()
        self.pending = self.pending[count:].clone()
        return result

    def flush(self):
        if self.closed:
            return self.pending.new_empty(0)
        result = self.push(torch.zeros(self.width, dtype=torch.float32))
        self.closed = True
        return result
