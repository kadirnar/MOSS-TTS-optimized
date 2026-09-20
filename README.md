# MOSS-TTS Optimized

Fast inference engine for [MOSS-TTS v1.5 8B](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)
with voice cloning, streaming **48 kHz audio**, and all **32 codebooks**.
Triton/CUDA kernels and CUDA graphs accelerate the LLM and codec.

## Benchmark

NVIDIA H200 NVL | MOSS-TTS v1.5 8B | Batch size 1 | 48 kHz output

| Method | Cached voice TTFA | Speedup | Fresh voice TTFA | Speedup |
|---|---:|---:|---:|---:|
| PyTorch BF16 (base) | 920.17 ms | 1.00x | 974.17 ms | 1.00x |
| Triton/CUDA BF16 | 166.59 ms | 5.52x | 188.38 ms | 5.17x |
| Triton/CUDA + GPTQ INT4/G32 | 73.04 ms | 12.60x | 96.00 ms | 10.15x |

TTFA is time to first playable audio, measured over 12 runs with a warm model.
Fresh voice includes reading and encoding a 3.112-second reference; cached voice
reuses its codes. Loading and network transit are excluded. The native 24 kHz
codec output is resampled to 48 kHz in every row, including the baseline.
G32 uses quantized weights. [Measurements and reproduction](docs/usage.md#reproduce-the-48-khz-benchmark).

## Quick Start

```bash
pip install git+https://github.com/kadirnar/MOSS-TTS-optimized.git
```

```python
from contextlib import closing

import soundfile as sf
from moss_tts import MossTTS

with MossTTS.from_pretrained() as tts:
    voice = tts.clone_voice("reference.wav")
    with closing(tts.stream("Hello, this is my voice.", voice=voice)) as chunks:
        with sf.SoundFile("speech.wav", "w", samplerate=tts.sample_rate,
                          channels=1, subtype="PCM_16") as output:
            for chunk in chunks:
                output.write(chunk.pcm.numpy())
```

The default preset is BF16. For G32, pass `preset="gptq"` and
`calibration_path="checkpoints/gptq-g32"` to `from_pretrained()`.
See [usage](docs/usage.md) for calibration, CLI and HTTP streaming.

## Requirements

- Linux, Python 3.12+, CUDA-enabled PyTorch 2.9+ and compatible torchaudio.
- NVIDIA GPU with enough memory for the 8B model, codec and CUDA graphs.
- G32: Hopper SM90, Triton 3.7.1, `nvcc` and calibrated weights.

## License

[Apache 2.0](LICENSE). See [NOTICE](NOTICE) for upstream attribution.
