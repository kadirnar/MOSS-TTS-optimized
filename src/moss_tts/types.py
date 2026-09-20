"""Public types; importing this module does not initialize CUDA."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Voice:
    """Reusable reference codes returned by MossTTS.clone_voice()."""

    codes: "torch.Tensor" = field(repr=False)
    codebooks: ClassVar[int] = 32


@dataclass(frozen=True)
class AudioChunk:
    """Independent CPU float32 mono PCM containing 80 ms of audio."""

    pcm: "torch.Tensor" = field(repr=False)
    frame: int
    elapsed_ms: float
    sample_rate: ClassVar[int] = 24000

    def pcm16(self) -> bytes:
        """Return signed 16-bit little-endian PCM without a container header."""
        import torch

        return (
            (self.pcm.clamp(-1, 1) * 32767)
            .to(torch.int16)
            .numpy()
            .astype("<i2", copy=False)
            .tobytes()
        )
