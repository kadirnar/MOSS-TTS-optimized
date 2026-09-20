# MOSS-TTS Optimized

Fast inference library for [MOSS-TTS v1.5 8B](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5),
a text-to-speech model with voice cloning and streaming audio. Custom Triton/CUDA
kernels and CUDA graphs accelerate both the LLM and audio codec while retaining
**all 32 codebooks** in every output frame.

## Benchmark

NVIDIA H200 NVL | MOSS-TTS v1.5 8B | Batch size 1 | 24 kHz mono | 32 codebooks

### Streaming Voice Cloning

Historical measurements of the selected **GPTQ INT4/G32 decoder**, with BF16
prefill and an FP32 codec. Time to first audio (TTFA) ends at the first complete
PCM chunk: **1,920 samples / 80 ms of audio**.

| Workload | Median TTFA | P95 TTFA |
|---|---:|---:|
| Warm engine, previously encoded voice | **71.86 ms** | 73.06 ms |
| Warm loopback HTTP, previously encoded voice | **75.02 ms** | 76.55 ms |
| Fresh voice registration + HTTP synthesis | **110.67 ms** | 113.46 ms |

Warm measurements include text preparation, LLM generation and codec decoding;
fresh-voice measurements also include reference registration and encoding, with
the WAV/base64 payload prepared beforehand. Model loading, graph warmup and
external network transit are excluded.

**The 50 ms target has not been reached.** These results predate the library
wrapper; its migration was checked for correctness, not rebenchmarked. G32
changes weight precision and does not imply equivalence to upstream BF16.
See the [improvement table](docs/performance.md), [timing samples](docs/benchmark.json)
and [library validation](docs/verification.json).

## Quick Start

Install a CUDA-enabled PyTorch/torchaudio stack suitable for your GPU, then:

```bash
pip install git+https://github.com/kadirnar/MOSS-TTS-optimized.git
```

### Streaming with Voice Cloning

The default `bf16` preset keeps the original weight precision. Provide a
0.2–15 second reference recording and stream the generated audio to a WAV file:

```python
import wave
from contextlib import closing

from moss_tts import MossTTS

with MossTTS.from_pretrained() as tts:
    voice = tts.clone_voice("reference.wav")

    with wave.open("speech.wav", "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)

        with closing(tts.stream("Hello, this is my voice.", voice=voice)) as chunks:
            for chunk in chunks:
                output.writeframes(chunk.pcm16())
```

Each chunk contains 80 ms of audio. Send `chunk.pcm16()` to your audio sink for
playback as chunks arrive. The first load downloads pinned model/codec weights
and warms the runtime; keep the model loaded for subsequent requests.

### Calibrated G32 Inference

Select the optimized Hopper path with an existing calibrated export:

```python
from contextlib import closing

from moss_tts import MossTTS

with MossTTS.from_pretrained(
    preset="gptq",
    calibration_path="checkpoints/gptq-g32",
) as tts:
    voice = tts.clone_voice("reference.wav")
    with closing(tts.stream("Hello, this is my voice.", voice=voice)) as chunks:
        for chunk in chunks:
            pcm = chunk.pcm16()  # Send to your audio sink.
```

Weights and reference recordings are not bundled. See [calibration and usage](docs/usage.md)
to create an export, configure caches or run offline. G32 retains BF16 weights
for prefill; it does not have a standalone 4-bit model's memory footprint.

### Command Line and HTTP

```bash
moss-tts synthesize --text "Hello world." --reference reference.wav --output speech.wav

pip install "moss-tts-optimized[server] @ git+https://github.com/kadirnar/MOSS-TTS-optimized.git"
moss-tts serve --host 127.0.0.1 --port 8000
```

The HTTP server supports voice registration and streaming raw PCM. See the
[HTTP example](docs/usage.md#http-streaming). Both APIs accept complete text and
stream audio output; each model handles one active request at a time.

## Requirements

- Linux and Python 3.12+.
- PyTorch 2.9+ with a compatible CUDA-enabled torchaudio installation.
- An NVIDIA GPU with enough memory for the 8B model, codec and CUDA graphs.
- G32: Hopper SM90, PyTorch 2.13.0, Triton 3.7.1, and a compatible CUDA toolkit with `nvcc`.

Other dependencies and supported version ranges are declared in [pyproject.toml](pyproject.toml).
See [development and tests](docs/usage.md#tests-and-distribution) for local installation.

## License

[Apache 2.0](LICENSE). Maintained by **kadirnar**, based on
[OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) and
[MOSS-Audio-Tokenizer](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer).
See [NOTICE](NOTICE) for third-party attribution.
