"""Public, serialized streaming API for the 8B model."""

import threading
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import BinaryIO, Literal

from .types import AudioChunk, Voice


class MossTTS:
    """One CUDA device, one active request, all 32 acoustic codebooks.

    Load with ``from_pretrained`` and reuse the instance and cloned voices.
    Close abandoned generators before starting another request. Use the model
    as a context manager to release its codec state when finished.
    """

    sample_rate = 48000
    codebooks = 32

    def __init__(self, engine, encoder, *, device, preset: str):
        self._engine = engine
        self._encoder = encoder
        self.device = device
        self.preset = preset
        self._lock = threading.Lock()
        self._closed = False
        self._last_metrics = {}
        from .resampling import output_resampler

        self._resample_kernel = output_resampler()

    @classmethod
    def from_pretrained(
        cls,
        *,
        preset: Literal["bf16", "gptq"] = "bf16",
        calibration_path: str | Path | None = None,
        device: str = "cuda:0",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        max_length: int = 1024,
        cpu_threads: int = 4,
    ) -> "MossTTS":
        """Download pinned checkpoints, load the 8B model, and capture CUDA graphs.

        ``bf16`` keeps the original weight precision. ``gptq`` selects calibrated
        G32 decode and SM90 kernels; supply the export explicitly. Both preserve
        BF16 prefill, FP32 codec, and 32 codebooks. Initialization is cold work.
        """
        if preset not in ("bf16", "gptq"):
            raise ValueError("preset must be 'bf16' or 'gptq'")
        if (preset == "gptq") != (calibration_path is not None):
            raise ValueError("Provide calibration_path exactly when preset='gptq'")
        if max_length < 128 or max_length % 128:
            raise ValueError("max_length must be a positive multiple of 128")
        if preset == "gptq" and max_length != 1024:
            raise ValueError("Bundled gptq kernels require max_length=1024")
        if not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        import torch

        target = torch.device(device)
        if target.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("MOSS-TTS 8B streaming requires an NVIDIA CUDA device")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        if preset == "gptq":
            import json

            import triton

            from ._runtime.paths import nvcc
            from .models import TTS_REVISION

            if torch.cuda.get_device_capability(target) != (9, 0):
                raise ValueError("The gptq preset is qualified for SM90 (Hopper) only")
            if triton.__version__ != "3.7.1":
                raise ValueError("The gptq preset requires Triton 3.7.1; install the hopper extra")
            folder = Path(calibration_path)
            config = json.loads((folder / "config.json").read_text())
            if config.get("complete") is False:
                raise ValueError("The calibrated export is incomplete")
            if config.get("group") != 32 or config.get("tts_revision") != TTS_REVISION:
                raise ValueError("A G32 export of the pinned 8B revision is required")
            missing = [
                folder / f"{layer:02d}_{projection}.pt"
                for layer in range(36)
                for projection in ("qkv", "out", "up", "down")
                if not (folder / f"{layer:02d}_{projection}.pt").is_file()
            ]
            if missing:
                raise ValueError(
                    f"Incomplete calibrated export: {len(missing)} projections missing"
                )
            nvcc()

        from ._runtime.presets import install_gptq, warmup
        from ._runtime.reference_encoder import ReferenceEncoder
        from ._runtime.streaming import StreamingTTS
        from .models import load_models

        torch.set_num_threads(cpu_threads)
        with torch.cuda.device(target), torch.inference_mode():
            model, codec, processor = load_models(
                cache_dir=cache_dir, local_files_only=local_files_only
            )
            engine = StreamingTTS(
                model,
                codec,
                processor,
                max_length=max_length,
                attention_block=32 if preset == "gptq" else 128,
                fused_residual=True,
                fused_gateup=preset == "bf16",
                codec_clock=preset == "gptq",
            )
            try:
                if preset == "gptq":
                    install_gptq(engine, calibration_path)
                warmup(engine, preset)
                encoder = ReferenceEncoder(processor)
                encoder.warmup()
            except BaseException:
                engine.codec.close()
                raise
        return cls(engine, encoder, device=target, preset=preset)

    @contextmanager
    def _request(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("This model is busy; finish or close the active stream first")
        try:
            if self._closed:
                raise RuntimeError("This model is closed")
            import torch

            with torch.cuda.device(self.device), torch.inference_mode():
                yield
        finally:
            self._lock.release()

    def clone_voice(self, audio_path: str | Path | BinaryIO) -> Voice:
        """Encode 0.2–15 seconds of reference audio once and reuse its codes."""
        import soundfile as sf
        import torch

        with self._request():
            with sf.SoundFile(audio_path) as source:
                if not 0.2 <= len(source) / source.samplerate <= 15:
                    raise ValueError("Reference audio must be between 0.2 and 15 seconds")
                wave = source.read(dtype="float32", always_2d=True)
                sample_rate = source.samplerate
            waveform = torch.from_numpy(wave.T.copy())
            if not torch.isfinite(waveform).all():
                raise ValueError("Reference audio must contain only finite samples")
            return Voice(codes=self._encoder.encode(waveform, sample_rate))

    def stream(
        self,
        text: str,
        *,
        voice: Voice | None = None,
        language: str = "English",
        max_new_tokens: int = 400,
        seed: int = 1234,
    ) -> Iterator[AudioChunk]:
        """Yield 48-kHz PCM; edge chunks are shorter to preserve filter continuity."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a nonempty string")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 33:
            raise ValueError("At least 33 generated steps are required for all 32 codebooks")
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if voice is not None and not isinstance(voice, Voice):
            raise TypeError("voice must be returned by clone_voice()")

        def generate():
            import torch

            from .resampling import StreamingResampler

            started = time.perf_counter()
            with self._request(), torch.random.fork_rng(devices=[self.device.index]):
                torch.cuda.manual_seed(seed)
                reference = None if voice is None else voice.codes
                resampler = StreamingResampler(self._resample_kernel)
                emitted = 0
                samples = 0
                first_audio_ms = None
                resample_ms = 0.0
                with closing(
                    self._engine.stream(text, reference, language, max_new_tokens)
                ) as chunks:
                    for chunk in chunks:
                        if chunk.sample_rate != 24000:
                            raise RuntimeError("Expected the pinned 24-kHz codec output")
                        tick = time.perf_counter()
                        pcm = resampler.push(chunk.pcm)
                        resample_ms += (time.perf_counter() - tick) * 1000
                        if pcm.numel():
                            elapsed = (time.perf_counter() - started) * 1000
                            if first_audio_ms is None:
                                first_audio_ms = elapsed
                            samples += pcm.numel()
                            yield AudioChunk(pcm, emitted, elapsed)
                            emitted += 1
                tick = time.perf_counter()
                tail = resampler.flush()
                resample_ms += (time.perf_counter() - tick) * 1000
                if tail.numel():
                    elapsed = (time.perf_counter() - started) * 1000
                    if first_audio_ms is None:
                        first_audio_ms = elapsed
                    samples += tail.numel()
                    yield AudioChunk(tail, emitted, elapsed)
                    emitted += 1
                self._last_metrics = {
                    **self._engine.last_metrics,
                    "ttfa_ms": first_audio_ms,
                    "total_ms": (time.perf_counter() - started) * 1000,
                    "resample_ms": resample_ms,
                    "output_chunks": emitted,
                    "output_samples": samples,
                    "sample_rate": self.sample_rate,
                }

        return generate()

    @property
    def metrics(self) -> dict:
        """Timing and frame counts for the last completed request."""
        from copy import deepcopy

        if self._closed:
            raise RuntimeError("This model is closed")
        return deepcopy(self._last_metrics)

    def close(self) -> None:
        """Release codec state and model references after active work ends."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Close the active stream before closing this model")
        try:
            if not self._closed:
                self._engine.codec.close()
                self._closed = True
                self._encoder = None
                self._engine = None
        finally:
            self._lock.release()

    def __enter__(self) -> "MossTTS":
        if self._closed:
            raise RuntimeError("This model is closed")
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
