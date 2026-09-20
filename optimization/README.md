# MOSS-TTS v1.5 streaming optimization

See the [complete improvement table](IMPROVEMENTS.md) for measured gains, experimental paths and the latest 71.86-ms result. Cleanup removed rejected derived checkpoints and Python bytecode; `results/cleanup_20260920.json` records every removal and the regeneration commands.

This implementation optimizes the requested **MOSS-TTS-v1.5 8B Delay checkpoint**, including voice-reference encoding, the LLM, and the causal audio decoder. It produces actual incremental 24 kHz mono PCM with **all 32 codebooks**. It does not substitute MOSS-TTS-Realtime, reduce the number of codebooks, or replay prerecorded output.

**Retain all 32 codebooks, including the first PCM chunk.** Every implementation and experiment here uses all 32; there is no reduced-codebook serving mode.

The latest [historical-KV preload pass](REPORT_ATTENTION_HISTORY.md) adds optional `--attention-history` to the qualified gate/up compiler preset. A twenty-pair repeat measures **73.64 → 71.86 ms warm cached-voice TTFA**, with **1.69 ms median paired gain** and 19/20 faster. Paired loopback HTTP is **76.44 → 75.02 ms**, with **1.92 ms paired gain**; fresh registration plus synthesis is **110.67 ms median**, partly affected by unchanged-encoder variation. All 48 cloning WAVs are byte-identical; all 76 in-process and 60 HTTP streams preserve PCM/RNG. Only historical cache loads precede the retained dependency wait. All 32 codebooks remain enabled, the 50-ms target remains unmet, and existing services are unchanged.

The preceding [CUDA register/resource sweep](REPORT_GATEUP_RESOURCES.md) tests 26 variants and keeps the preceding selected kernel. Its best candidate has only **0.068 ms median paired gain** in the final comparison, with slightly worse p95, and remains benchmark-only. All 136 complete streams and 240 full-model graph cases match; memory, race and synchronization checks each pass 252 cases. Common runtime shifts affect absolute TTFA, so its approximately 71-ms observations are not claimed as a 2-ms kernel improvement. All 32 codebooks remain required and the 50-ms goal remains open.

The latest [exact gate/up compiler pass](REPORT_GATEUP_COMPILER.md) adds opt-in `--gateup-compiler` to the qualified `--down-tile8` preset. Final paired warm cached-voice TTFA is **73.88 → 73.50 ms**, with **0.24 ms median paired gain**. Paired HTTP is **77.40 → 77.10 ms**; fresh-reference registration plus synthesis is **110.75 ms median**, with a worse fresh-workload p95. All 48 cloning WAVs remain byte-identical; memory, race and synchronization checks each pass 252 cases. The selected kernel uses one CTA; multi-CTA gate/up variants lose despite fewer registers. All 32 codebooks and streaming voice cloning remain enabled. The **50-ms target is still unmet**, and existing supervisor services are unchanged.

The preceding [MLP schedule review and paired HTTP test](REPORT_MLP_SCHEDULE.md) rejects 58 screened configurations as additional improvements and confirms the existing down tile remains selected. A single temporary process with alternating graph variants measures **77.79 → 77.31 ms median HTTP TTFA**, a **0.33 ms paired gain** with 15/20 pairs faster; p95 regresses and all outliers are retained. Ten fresh-reference pairs measure **110.93 / 109.97 ms** median, partly influenced by unchanged-encoder variation. All sixty measured streams preserve PCM/RNG. These paired results supplement the earlier separate-process HTTP regression; no further kernel change or 50-ms success is claimed.

**The 50 ms TTFA target has not been reached.** The preceding [down-projection tile pass](REPORT_DOWN_TILE.md) measures **73.77 ms median warm cached-voice TTFA** versus 74.23 ms control, with **0.45 ms median paired gain** and 17/20 pairs faster. Optional `--down-tile8` uses eight rows/four warps with the qualified clustered-QKV preset; all 32 codebooks remain enabled. All 48 bilingual cloning WAVs remain byte-identical, and memcheck/racecheck/synccheck each pass 72 cases. In-process p95 slightly regresses. The separate HTTP comparison also regresses, **75.75 → 77.42 ms**, so no HTTP latency improvement is claimed. Initial LLM generation remains about 60.56 ms; one fresh-reference HTTP observation is 115.10 ms. Cluster placement and full down-weight preloads remain unselected. Streaming voice cloning and the original supervisor services are retained.

The preceding [clustered QKV preparation](REPORT_QKV_CLUSTER.md) measures **74.14 ms median warm cached-voice TTFA** versus 74.23 ms control: a small **0.11 ms median paired gain**, with 16/20 pairs faster. Optional `--qkv-cluster` fuses G32 QKV projection, head normalization, rotary handling and KV writes using eight-CTA SM90 clusters. It uses isolated Triton 3.8 cubins in the selected Triton 3.7.1 runtime, with explicit reduction order and zero semantics. All 48 bilingual cloning WAVs remain byte-identical; memcheck, racecheck and synccheck each pass 72 cases. HTTP median is 77.72 ms, but its p95 slightly regresses; one fresh-reference observation is 112.97 ms. Initial LLM generation still takes about 61.08 ms. All 32 codebooks, streaming voice cloning and the original supervisor services are retained.

The preceding [projection wait audit and output-weight staging](REPORT_ASYNC_WEIGHTS.md) measures **74.25 ms median warm cached-voice TTFA**, with **0.61 ms median paired gain** and 19/20 pairs faster. Optional `--output-weight-prefetch` preloads immutable G32 attention-output weights before the CUDA dependency wait, using sixteen rows per block. It outperforms the tested shared-memory copy schedule; TMA down-projection variants are slower. All 164 measured streams across four comparisons retain exact PCM/RNG states. Both 48-utterance cloning suites (shared-copy and register-preload candidates) match previous WAVs byte for byte. The final HTTP comparison measures 79.35 ms median and one fresh-reference observation 119.96 ms; the report records runtime variation and earlier faster HTTP observations. The initial 32-step LLM interval is still **61.19 ms**. All 32 codebooks, G32 decode, BF16 prefill, the FP32 codec and streaming voice cloning remain enabled; original supervisor services are unchanged.

The preceding [prefill pointwise fusions](REPORT_PREFILL_POINTWISE.md) measure **74.73 ms median warm cached-voice TTFA**, with **0.64 ms median paired gain** over the fused-QKV preset. All eighty streams in a four-mode ablation and all 48 bilingual cloning WAVs match their controls. HTTP median is **78.44 ms**; fresh-reference registration plus synthesis takes **117.89 ms** in one observation. Optional `--prefill-pointwise` fuses prefill SiLU/product and residual/normalization, removing another 108 launches. Split-normalization decode schedules and register caps are slower and remain unselected. The selected preset retains all 32 codebooks, calibrated INT4/G32 decode, BF16 prefill, the FP32 clocked codec and initial-audio graph. Neither original supervisor service has been replaced.

The subsequent [G64 weight-group experiments](REPORT_GROUP64.md) retain all 32 acoustic codebooks and screen 53 configurations across three all-layer comparisons. Selective gate/up + QKV reaches **74.06 ms**, with **0.73 ms median paired gain**; gate/up alone gains **0.44 ms**. Both regress normalized Chinese transcription diagnostics in the original and expanded cloning suites, so neither is selected. All 96 new candidate WAVs finish and English WER remains zero. Wider activation groups and most compact-scale layouts are slower. The selected G32 preset above remains the qualified result.

The preceding [BF16 prefill QKV fusion and address specialization](REPORT_PREFILL_QKV.md) measures **75.30 ms median**, with **1.79 ms median paired gain** from the corrected QKV preparation kernel. `--prefill-qkv` remains selected; `--bulk-prefetch` includes the separately measured QKV address cleanup.

The preceding [Hopper bulk-prefetch option](REPORT_BULK_PREFETCH.md) measures **77.26 / 77.27 ms median** in two twenty-pair runs, with **0.45 / 0.48 ms median paired gain**. It hints only a small prefix of projection weights before dependency waits and remains part of the selected preset.

The preceding [initial audio CUDA graph](REPORT_FIRST_AUDIO_GRAPH.md) measures **77.91 ms median warm cached-voice TTFA**, with **1.62 ms median paired gain** and all twenty pairs faster. Every paired full PCM stream and all 48 bilingual cloning WAVs remain identical. Its optional `--first-audio-graph` removes per-step CPU round trips for the initial 32 audio steps and remains part of the selected preset.

The preceding [shared codec clocks and first-frame specialization](REPORT_CODEC_CLOCK.md) measure **79.53 ms median warm cached-voice TTFA**, with **0.30 ms median paired gain** over the preceding attention-PDL path. All 40 measured paired PCM streams and 48 bilingual cloning WAVs are identical. Its optional `--codec-clock` remains part of the selected preset.

The preceding [CUDA attention dependency overlap](REPORT_ATTENTION_PDL.md) measures **79.71 ms median warm cached-voice TTFA**, with **4.00 ms median paired gain** over projection PDL. A later runtime phase measures **77.50 ms**, with a smaller **1.86 ms paired gain**; unchanged stages also shift, and all samples are retained. These separate-process medians must not be subtracted to estimate a codec gain. The optional `--attention-pdl` extends `--projection-pdl` through native attention and requires exact short scales. The latest codec comparison alternates both implementations in one process.

A subsequent [CUDA resource and parallel-warp audit](REPORT_PROJECTION_RESOURCES.md) tests 44 alternatives: CUDA 13 shared-memory register spilling, register caps, and parallel gate/up partitions. All 5,014 chain/graph comparisons match and both sanitizers pass 54 representative cases, but every alternative has a slower median. No resource or warp-partition alternative is selected. That audit motivated the initial-audio scheduling change now measured and qualified above.

The preceding [CUDA projection dependency overlap](REPORT_PROJECTION_PDL.md) measured **83.46 / 83.47 ms median** in two fifteen-round comparisons, with **0.76 / 0.64 ms median paired gain** over exact short-scale storage. Extending dependencies through attention passes 2,592 all-layer comparisons, 288 complete-graph checks, 135 measured full-stream comparisons, and both CUDA sanitizers. Projections remain the largest profiled resident-interval category; overlap durations include waits and must not be summed as critical-path percentages.

The preceding [exact BF16 scale storage](REPORT_SHORT_SCALES.md) measures **84.19 / 84.13 ms median**, with **0.52 / 0.44 ms median paired gain** over ordinary FP32 scale storage. It halves 72 unchanged-value scale buffers from 288 to 144 MiB. PDL adds explicit CUDA graph dependencies and preloads only immutable output/down scales before waiting for producer data. Every tested complete stream matches; whole-matrix preloads are slower and rejected. Profiling durations overlap under PDL and must not be added as if kernels were serialized.

The earlier [normalization/consumer fusion](REPORT_NORM_PROJECTION.md) reduces warm cached-voice TTFA from **87.25 to 85.48 ms median** in ten alternating same-process pairs, with **1.77 ms median paired gain**. Every pair improves, all complete PCM streams match, and all 16 bilingual cloning WAVs remain byte-identical to the preceding calibrated path.

Its sequential HTTP servers measure **92.44 → 88.89 ms median TTFA**, candidate p95 **90.83 ms**, over 20 complete cached-voice requests. All PCM hashes/frame counts match, and cancellation/recovery passes. Runtime variation contributes to the HTTP difference; the paired in-process experiment better isolates the kernel gain. One fresh-registration-plus-synthesis observation takes **127.75 ms**. HTTP and in-process tests use identical text and 145 prompt tokens. See the normalization report for stage timings, quality limits, sanitizer checks and the opt-in `--norm-projection` serving command. Temporary test endpoints are stopped.

A subsequent optional [audio-head graph optimization](REPORT_AUDIO_HEADS.md) skips only future heads already masked by the delay schedule. It keeps all 32 codebooks in every PCM frame and preserves every tested stream/WAV. Ten paired requests measure **85.27 → 84.84 ms median**, with **0.51 ms median paired gain** (eight improvements, two regressions; p95 slightly worsens). Its HTTP median is **88.72 ms**, with a large outlier and timing shifts in unchanged stages. `--audio-head-buckets` is an additional opt-in option; the narrower paired gain is more informative than the larger sequential HTTP difference.

The latest [G128 projection experiments](REPORT_GROUP128.md) retain all 32 codebooks; G128 refers to weight quantization groups. New packed kernels, producer fusions and activation-weighted scale search reach **83.39 ms** in a separate-process compiler experiment, but regress on 32 additional cloning utterances. A narrower output/down-only candidate gives **1.01 ms median paired gain** (84.84 → 83.76 ms, all ten pairs faster), yet also worsens Chinese transcription diagnostics. Both remain experimental; selected G32 is unchanged. The expanded G32 suite finishes all 32 samples with 1.18% normalized Chinese CER, 0% English WER and intended-reference top-one matches throughout. This small diagnostic set is not broad or human quality acceptance.

A subsequent [draft and CUDA load-policy audit](REPORT_DRAFT_AND_LOAD.md) measures six untrained layer-skipping drafts and 152 load/scheduling configurations. Joint acoustic acceptance is too low for the tested drafts to justify a verifier. A QKV cache-policy microbenchmark initially suggests a 9% gain, but rotating all 36 layers reduces it to about 0.4%; full requests improve only a noisy 0.13 ms. It remains a benchmark/quality option, with all 48 cloning WAVs byte-identical. Serving and the selected G32 preset are unchanged. The report records the benchmark correction, failed variants and remaining projection bottleneck.

The [lossless CUDA allocation experiment](REPORT_COMPRESSION.md) adds optional `--compressed-scales` for 72 FP32 scale buffers. Two fifteen-round comparisons show **0.21 / 0.15 ms median paired gain**, with candidate medians **84.43 / 84.33 ms** and every PCM stream exact. HTTP median is **88.20 → 87.97 ms**; all twenty streams match and cancellation recovers. All 48 cloning WAVs remain byte-identical. Broader weight, codec and audio-head compression is rejected after measurement. This is a small optional gain, defaults off, and leaves existing services unchanged; the 50 ms target remains unmet with all 32 codebooks retained.

The subsequent [short-scale pass](REPORT_SHORT_SCALES.md) halves those buffers from 288 to 144 MiB with unchanged values and fixed Gluon reduction layouts. It improves another **0.32 / 0.20 ms median paired** versus compressed FP32; the flags are mutually exclusive. All 1,296 all-layer comparisons, 192 full-graph checks, 48 cloning WAVs and twenty HTTP PCM streams match, and memcheck/racecheck pass. A separate [INT4 PTX experiment](REPORT_INT4_PTX.md) passes extensive integer/operator checks but is rejected: on this H200 its matrix instructions lower into INT8 IMMA/conversion sequences and run substantially slower than selected DP4A. Projections remain about 59% of profiled GPU time.

Earlier iterations are preserved in [scaled INT4 unpacking](REPORT_SCALED_DP4A.md), [gate/up quantization](REPORT_GATEUP_QUANT.md), [native CUDA attention](REPORT_NATIVE_ATTENTION.md), and [calibration](REPORT_CALIBRATED.md). A separate [native projection and integer tensor-core sweep](REPORT_NATIVE_PROJECTIONS.md) tested 448 configurations but found no reliable full-request improvement; its native output candidate remains experimental. [GOAL_REVIEW.md](GOAL_REVIEW.md) records the broader library/kernel experiments and [REPORT_REVIEW.md](REPORT_REVIEW.md) covers running services. All 32 codebooks remain enabled.

## Running service on this instance

The supervisor service `moss_tts_optimized` listens on **127.0.0.1:18080**. It is local-only. Logs are at `/var/log/portal/moss_tts_optimized.log`. It enables fused residual/RMSNorm and gate/up/SiLU kernels, retaining the original attention tile and precision. All 16 bilingual cloning samples were byte-identical to the previous custom BF16 implementation. The current 20-request loopback HTTP median is 175.45 ms; the separate five-request in-process median is 170.96 ms.

```bash
supervisorctl status moss_tts_optimized
curl http://127.0.0.1:18080/health
curl -N http://127.0.0.1:18080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, this is a speech synthesis test.","voice":"demo","language":"English"}' \
  --output output.pcm
ffplay -f s16le -ar 24000 -ac 1 output.pcm
```

Each generated codec frame is 1,920 samples (80 ms), transmitted as 3,840 signed 16-bit little-endian PCM bytes. HTTP packet boundaries may split or join these chunks. A client should buffer complete two-byte samples. `demo` uses the repository's supplied Chinese reference recording.

For private access from your own computer:

```bash
ssh -p 50716 -L 18080:127.0.0.1:18080 root@93.91.156.84
```

Then call `http://127.0.0.1:18080` on your computer.

Register a different reference voice, then reuse its returned ID:

```python
import base64
import requests

base = "http://127.0.0.1:18080"
with open("reference.wav", "rb") as f:
    payload = {"wav_base64": base64.b64encode(f.read()).decode()}
r = requests.post(base + "/v1/voices", json=payload)
r.raise_for_status()
voice = r.json()["voice"]

with requests.post(base + "/v1/audio/speech", json={
    "input": "Hello, this is a streaming voice cloning test.",
    "language": "English", "voice": voice, "seed": 1234,
}, stream=True) as r:
    r.raise_for_status()
    with open("output.pcm", "wb") as f:
        for chunk in r.iter_content(chunk_size=3840):
            f.write(chunk)
```

Voice registration accepts 0.2–15 seconds of WAV audio, converts stereo to mono, resamples, applies the original loudness normalization, and caches codec tokens. The cache holds 64 voices and is process-local. There is one GPU worker; overlapping synthesis/registration requests receive HTTP 429. Disconnecting cancels generation and releases state.

## Python streaming interface

```python
import torch
from optimization.common import load_models
from optimization.streaming import StreamingTTS
from optimization.reference_encoder import ReferenceEncoder

model, codec, processor = load_models()
engine = StreamingTTS(model, codec, processor)
engine.warmup()                       # compile/capture before accepting traffic
encoder = ReferenceEncoder(processor)
encoder.warmup()

# reference_wave: float32 tensor [channels, samples]
reference_codes = encoder.encode(reference_wave, sample_rate=24000)
for chunk in engine.stream("Hello world.", reference_codes, language="English"):
    play(chunk.pcm, sample_rate=24000)  # independent CPU float32 tensor
```

The instance owns mutable KV state and must be used by one thread/request at a time. A generator that is abandoned early must be closed. The HTTP implementation handles both requirements with a dedicated worker and bounded queues. CUDA kernels use the current stream; numerical validation also exercises a non-default stream.

## Reproduce measurements

Use Python 3.12 and the versions in [requirements-runtime.txt](requirements-runtime.txt). Install the CUDA 12.8 PyTorch wheels from `https://download.pytorch.org/whl/cu128`. The two model revisions are pinned in `common.py`; download those full snapshots with `huggingface_hub.snapshot_download` before loading. Downloaded checkpoints remain unchanged; experimental calibrated weights are exported separately.

From the repository root:

```bash
/venv/main/bin/python -m optimization.baseline
/venv/main/bin/python -m optimization.benchmark_upstream_ttfa
/venv/main/bin/python -m optimization.benchmark_codec
/venv/main/bin/python -m optimization.benchmark_encoder
/venv/main/bin/python -m optimization.benchmark_llm
/venv/main/bin/python -m optimization.benchmark_attention_backends
/venv/main/bin/python -m optimization.benchmark_libraries
/venv/main/bin/python -m optimization.benchmark_streaming --runs 5 --max-new-tokens 160
/venv/main/bin/python -m optimization.benchmark_workloads
/venv/main/bin/python -m optimization.validate
/venv/main/bin/python -m optimization.validate_encoder
/venv/main/bin/python -m optimization.profile_final
/venv/main/bin/python -m optimization.benchmark_http
```

Run GPU benchmarks sequentially without concurrent inference. `baseline` produces the reusable fixture used by the other scripts. Warmup/capture is excluded from warm latency measurements. Raw samples, environment versions, traces, and generated audio are saved under `results/`.

Optional FP8 MLP experiment:

```bash
/venv/main/bin/python -m optimization.benchmark_streaming \
  --fp8-mlp --runs 5 --max-new-tokens 160
```

Full-codebook follow-up experiments (run one GPU command at a time):

```bash
/venv/main/bin/python -m optimization.benchmark_quantization --mode none
/venv/main/bin/python -m optimization.benchmark_quantization --mode fp8_all
/venv/main/bin/python -m optimization.benchmark_quantization --mode int8_all
/venv/main/bin/python -m optimization.benchmark_quantization --mode int4_all
/venv/main/bin/python -m optimization.tune_weight_reads
/venv/main/bin/python -m optimization.validate_weight_kernels
/venv/main/bin/python -m optimization.benchmark_flashinfer
/venv/main/bin/python -m optimization.validate_quantized_audio
/venv/main/bin/python -m optimization.build_all32_report
```

Add `--tiled` to the FP8/INT8 runs to test multiple rows per CTA, or to the INT4 run to test paired nibble unpacking. The isolated tile sweep improved some microbenchmarks but the integrated FP8/INT8 variants regressed TTFA. These modes retain all 32 codebooks, BF16 prefill and activations, and the FP32 codec. They retain the original BF16 weights in memory. `weight_quantization='fp8_all'`, `'int8_all'` or `'int4_all'` can be passed explicitly to `StreamingTTS` for experiments; the default is `'none'`. FlashInfer is optional, selectable with `attention_backend='flashinfer'` or `'flashinfer_tc'`, and requires the pinned optional FlashInfer packages.

`benchmark_quantization` saves five complete measured requests, first-chunk latency, component timing, 36 teacher-forced comparisons, generated audio and a subsequent GPU profile. A short ASR check is insufficient to approve quantization for voice-cloning quality. The server CLI accepts `--weight-quantization fp8_all` explicitly; it is disabled by default.

The current experimental FP8 configuration uses `/venv/moss-vllm` (Torch 2.13.0+cu130, Triton 3.7.1) with both fusions and attention block 32:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_quantization \
  --mode fp8_all --attention-block 32 --fused-residual --fused-gateup --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_workloads \
  --mode fp8_all --attention-block 32 --fused-residual --fused-gateup --name workloads_reproduce
```

Its measured in-process median is 118.14 ms; its 20-request HTTP median is 122.17 ms. Including fresh reference encoding measured 138.79 ms in process. On the diagnostic four-voice/16-utterance suite, normalized Chinese ASR error increased from 4.46% to 7.59%; mean speaker-vector cosine changed from 0.9104 to 0.9090. This is why it is experimental. The separate local supervisor service `moss_tts_fp8_experimental` is running on port 18083 with autostart disabled. It has an independent voice cache and the same API; change `base` to `http://127.0.0.1:18083` to select it explicitly.

`benchmark_libraries` additionally requires the optional library versions listed in `requirements-runtime.txt`. It compiles a small CUDA/C shared library using nvcc. SGLang/vLLM CUDA operators are used directly; their complete serving dependency stacks are not prerequisites for the streaming implementation. The SGLang attention kernel copy preserves its Apache-2.0 notice and replaces only a platform-detection import; see `vendor/README.md`.

## Scope and limits

- Batch one, full text input, streaming audio output. Incremental text input is not implemented.
- One assistant audio segment per call; no multi-speaker/multi-segment serving interface.
- Default static KV capacity is 1,024 total prompt plus generation positions. Oversized requests are rejected. The Python constructor accepts larger multiples of 128, with different latency/memory costs.
- The kernels specialize the exact Qwen3-8B and original MOSS codec dimensions. Shape guards reject incompatible architectures.
- The default service retains BF16 LLM weights and FP32 codec computation. Floating point operation ordering changes some logits relative to upstream; the new fusions preserve the prior custom BF16 output on the 16-sample suite. Numerical, cache-wrap, reset, reference-token, bilingual ASR and speaker-vector checks are recorded. Corpus-level quality and human listening acceptance remain outstanding.
- Full SGLang-Omni 0.1.6 and vLLM-Omni 0.28 scheduler/HTTP trials are recorded in `GOAL_REVIEW.md`. vLLM required an explicitly gated v1 codec and incremental delay adapter, so its result is a hybrid implementation, not an unmodified-library benchmark.
- No Tensor Parallel implementation, speculative decoder, distilled draft model, INT4 deployment, or quality-reduced codebook mode is claimed.
