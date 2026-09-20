# Usage

## Python API

Public output is 48 kHz mono. The pinned codec produces 24 kHz internally; a
continuous Kaiser-windowed sinc filter converts it to 48 kHz without changing
pitch or duration. Resampling cannot add bandwidth absent from the native signal.
The filter retains context across chunks and flushes its tail at the end.
Interior chunks have 3,840 samples (80 ms); the first has 3,772 samples and the
final tail has 68. Consume all chunks, including the tail, for exact duration.
`AudioChunk.elapsed_ms` and `metrics["ttfa_ms"]` include output resampling.


```python
from contextlib import closing
from moss_tts import MossTTS

with MossTTS.from_pretrained(device="cuda:0", local_files_only=True) as tts:
    voice = tts.clone_voice("reference.wav")
    with closing(tts.stream(
        "Welcome to the demonstration.", voice=voice, language="English", seed=1234,
    )) as stream:
        for chunk in stream:
            # chunk.pcm: independent CPU float32 tensor at 48 kHz.
            # chunk.pcm16(): signed 16-bit little-endian PCM bytes.
            consume(chunk.pcm16())  # Replace with your application's audio sink.
    print(tts.metrics)
```

Omit `voice` for unconditioned synthesis. A reference can be a local filename
or a binary file-like object such as `io.BytesIO`. It must contain 0.2–15 seconds
of audio. Stereo is mixed to mono, resampled to 24 kHz, and loudness-normalized
before encoding. Reusing `Voice` avoids repeating reference encoding.

One model owns mutable KV caches and codec state. Generation and voice encoding
are serialized; overlapping calls fail with a busy error. Iteration owns the
request until completion or `stream.close()`. Use `contextlib.closing` when the
consumer may stop early. Audio tensors remain valid after later graph replays.

`max_new_tokens` defaults to 400 and must be at least 33. Total prompt plus
generation positions must fit `max_length`, normally 1024. A low budget can
produce a truncated utterance; inspect `tts.metrics["truncated"]` after completion.
The G32 binaries require `max_length=1024`; the BF16 path accepts multiples of 128.

Initialization downloads weights, compiles kernels and captures CUDA graphs.
Keep the model loaded for warm streaming requests. `cpu_threads=4` sets PyTorch's
process-wide CPU thread count; choose another value explicitly when needed.
The library does not provide incremental text input or concurrent GPU batching.

## G32 calibration

The `gptq` preset requires all 144 calibrated projection files and `config.json`.
An existing selected export can be supplied directly:

```python
tts = MossTTS.from_pretrained(
    preset="gptq", calibration_path="checkpoints/gptq-g32", local_files_only=True,
)
```

To create a new export, provide a JSONL dataset. Reference paths are relative to
the dataset file. Use varied training text, voices and languages, with separate
held-out validation examples. A minimal format example is:

```jsonl
{"text":"The morning train is arriving.","reference":"reference.wav","language":"English","split":"train"}
{"text":"Please remember to bring your report.","reference":"reference.wav","language":"English","split":"validation"}
```

```bash
moss-tts calibrate --dataset calibration.jsonl --output checkpoints/gptq-g32
```

Calibration first generates BF16 sequences and collects decoder activations,
then exports symmetric INT4 groups of 32 with static BF16 scales and damping 0.1.
It reports held-out projection error. A new directory is required; existing
exports and original Hugging Face weights are preserved. This is offline GPU
work, not a serving startup step. New data creates different weights and needs
its own speech-quality evaluation. Two example rows document the file format;
they do not establish a sufficient calibration corpus.

G32 retains the BF16 source weights for prefill, so its memory footprint is
larger than a standalone 4-bit checkpoint. The measured environment was H200
NVL with Python 3.12, Torch 2.13.0, Triton 3.7.1 and Transformers 5.14.1. Native
attention was built with CUDA 12.8. Isolated Triton 3.8 cubins are bundled;
the host remains on Triton 3.7.1.

## HTTP streaming

```bash
pip install -e '.[server]'
moss-tts serve --host 127.0.0.1 --port 8000
# Add --preset gptq --calibration-path checkpoints/gptq-g32 for the G32 path.
```

The process warms one model on startup. The default loopback binding is intended
for local use or an authenticated reverse proxy. Run persistent deployments with
a process manager. The server does not implement authentication itself.

Register a voice and stream audio with Python's standard HTTP client:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

base = "http://127.0.0.1:8000"

def post(path, payload):
    request = Request(base + path, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    return urlopen(request, timeout=120)

with post("/v1/voices", {
    "wav_base64": base64.b64encode(Path("reference.wav").read_bytes()).decode(),
}) as response:
    voice = json.load(response)["voice"]

with post("/v1/audio/speech", {
    "input": "Hello, this is streamed audio.", "voice": voice, "language": "English",
}) as response, open("speech.pcm", "wb") as output:
    while chunk := response.read(7680):
        output.write(chunk)
```

Responses contain raw 48 kHz mono `s16le` PCM without a WAV header. HTTP transport
boundaries can differ from audio frames; buffer complete two-byte samples.
Omit `voice` for unconditioned synthesis. The voice cache is process-local and
holds 64 entries. Delete unused entries with `DELETE /v1/voices/{voice_id}`.
Overlapping encoding/synthesis requests receive HTTP 429. Disconnecting cancels
generation and releases the model for the next request.

## Caches and installation

- `HF_HOME` or `cache_dir=` controls Hugging Face checkpoint storage.
- `MOSS_TTS_CACHE` controls native kernel builds; the default is
  `$XDG_CACHE_HOME/moss-tts` or `~/.cache/moss-tts`.
- `MOSS_TTS_NVCC` overrides the compiler path; otherwise `PATH`, then
  `$CUDA_HOME/bin/nvcc`, then `/usr/local/cuda/bin/nvcc` is used.
- Package files are read-only resources. Runtime builds do not write into `site-packages`.
- The `hopper` extra pins qualified host versions; it does not install the CUDA toolkit.

## Tests and distribution

```bash
pip install -e '.[dev,server]'
pytest -m 'not gpu'
python -m build
```

The CPU suite checks request ownership/cancellation, HTTP errors and voice
lifecycle, package resources and binary hashes, CLI entry points, and the existing
38 multilingual text-normalization cases including idempotence.

Run the complete-model integration check explicitly:

```bash
MOSS_TTS_TEST_REFERENCE=/path/to/reference.wav pytest -m gpu
MOSS_TTS_TEST_REFERENCE=/path/to/reference.wav \
MOSS_TTS_TEST_CALIBRATION=/path/to/gptq-g32 pytest -m gpu
```

Download checkpoints before these offline GPU tests. Run GPU jobs sequentially.
The tests validate streaming and recovery; they are not statistically qualified
performance benchmarks. See [performance](performance.md) for prior measurements.

## Reproduce the 48 kHz benchmark

```bash
python examples/benchmark.py --reference reference-48k.wav \
    --calibration checkpoints/gptq-g32 --output docs/benchmark_48khz.json
```

Each backend runs in a separate process with the same text, seed, reference and
32 codebooks. Three warmup requests per workload precede 12 measured requests.
TTFA ends when the first 3,772-sample 48 kHz PCM chunk is available as `s16le`
bytes, including text processing, generation, decoding and resampling. Fresh
voice measurements also include reading and encoding the reference WAV. Model
loading, warmup and network transit are excluded. A forward observer exposes the
first complete frame from the upstream generation loop; its overhead is included.
The quantized G32 row changes weight precision. These timings do not measure full
utterance throughput or establish equal speech quality between backends.
