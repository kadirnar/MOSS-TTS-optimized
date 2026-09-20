# MOSS-TTS Optimized

A Python library for **MOSS-TTS v1.5 8B** with voice cloning, streaming audio,
and Triton/CUDA acceleration. Every audio frame retains **all 32 codebooks**.

```python
from contextlib import closing
from moss_tts import MossTTS

with MossTTS.from_pretrained() as tts:
    voice = tts.clone_voice("reference.wav")
    with closing(tts.stream("Hello, this is my voice.", voice=voice)) as chunks:
        for chunk in chunks:
            pcm = chunk.pcm16()  # 80 ms of 24 kHz mono PCM, ready for your audio sink
```

## Install

Requires Linux, Python 3.12+, and an NVIDIA GPU with sufficient memory for the
8B model, codec and CUDA graphs. Install a PyTorch/torchaudio CUDA stack suitable
for your GPU first, then install this checkout:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e '.[server]'         # HTTP streaming
pip install -e '.[hopper,server]'  # Qualified G32 host versions and HTTP
pip install -e '.[dev,server]'     # Packaging and CPU tests
```

The first load downloads pinned model and codec snapshots to the Hugging Face
cache, then warms the kernels and captures graphs. Pass `local_files_only=True`
after downloading for offline startup. Weights and reference recordings are not
bundled in the repository. Importing `moss_tts` does not load Torch or initialize CUDA.

## Choose a preset

| Preset | Weights / runtime | Intended use |
|---|---|---|
| `bf16` (default) | BF16 LLM, FP32 codec, Triton and CUDA graphs | Original weight precision |
| `gptq` | Calibrated INT4/G32 decode, BF16 prefill, FP32 codec | Selected Hopper optimization |

```python
tts = MossTTS.from_pretrained(
    preset="gptq",
    calibration_path="checkpoints/gptq-g32",
)
```

G32 requires an explicit calibrated export, Hopper SM90, Triton 3.7.1 and `nvcc`.
The package includes the selected SM90 cubins and native CUDA sources. Compiled
libraries are cached outside the installed package. See [usage](docs/usage.md)
for calibration, device selection, concurrency, and cache configuration.

The historical selected path measured **71.86 ms engine TTFA**, **75.02 ms
loopback HTTP TTFA**, and **110.67 ms including fresh voice registration** on an
H200 NVL. **The 50 ms target remains unmet.** These are prior qualified results;
latency depends on the workload and environment. See the
[complete improvement table and evidence](docs/performance.md).

## Command line

```bash
moss-tts synthesize --text "Hello world." --reference reference.wav --output speech.wav
moss-tts serve --host 127.0.0.1 --port 8000
```

The optional HTTP server supports voice registration and incremental raw PCM:
`POST /v1/voices`, `DELETE /v1/voices/{voice_id}`, `POST /v1/audio/speech`, and
`GET /health`. It owns one GPU worker and returns HTTP 429 for overlapping work.
The API accepts complete text; audio output is streamed as it is generated.

## Repository

```text
src/moss_tts/  Public API, CLI, HTTP transport, 8B model, codec and private kernels
examples/     Minimal library usage
tests/       API, packaging, HTTP, text normalization and opt-in GPU checks
docs/        Usage, improvement table and compact validation evidence
```

Run `pytest -m 'not gpu'` for CPU checks. The optional GPU test is described in
[usage](docs/usage.md). Build a wheel and source distribution with `python -m build`.

Maintained by **kadirnar** as an independent repository. Based on
[OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) and
[OpenMOSS/MOSS-Audio-Tokenizer](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer).
See [NOTICE](NOTICE) and [LICENSE](LICENSE) for source attribution and licensing.
Documentation, comments and interfaces use English; multilingual test inputs
remain in their original language.
