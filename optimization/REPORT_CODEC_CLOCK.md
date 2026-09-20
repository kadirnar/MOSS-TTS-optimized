# Shared codec clocks and first-frame specialization

This pass keeps **all 32 codebooks, FP32 codec weights/arithmetic, and streaming voice cloning**. It removes redundant decoder bookkeeping and specializes only the first codec frame. The LLM remains the qualified G32/attention-PDL implementation. **The 50 ms target remains open.**

The selected `ClockedStreamingCodec` uses four shared stage offsets for 1/2/4/8 tokens per audio frame. All attention kernels in a stage read the same immutable offset. A single `torch._foreach_add_` advances the four offsets **after the entire decoder frame**. This avoids writing an offset while another CTA still reads it. The original 68 per-attention device increments and four unused rotary-transformer offset updates are removed. Per-layer device offset fields are unused by this wrapper; the four shared offsets are authoritative. Reset clears them before a new utterance, including after cancellation.

The 32 initial decoder layers each process one token in the first frame. With only that valid position, their attention output is exactly V. A small Triton kernel still computes/stores the rotated K and V needed by subsequent frames, and directly writes V to the attention output. It retains the original QKV projection. Later stages and every steady-state frame retain the established full-capacity attention reductions. A separate first-frame CUDA graph is dispatched only immediately after reset.

The wrapper owns its graphs, output clones, clock buffers, and streaming cache state. It supports one sequential batch-one stream, the original four decoder stages, fixed projection weights, FP32 weights, and rotary-only transformers. Encoder and quantizer behavior are unchanged. The 4096-frame/327.68-second rotary-table guard remains enforced before launching an excessive frame. Optional `--codec-clock` wiring is provided in benchmark, quality-generation and server entry points; existing services remain unchanged.

## Codec screening

The clean standalone test times all variants before initializing a profiler. It uses one unchanged codec, 140 real-code frames spanning every attention cache wrap, three reset/replay streams, and four first-frame code probes. Cache capacities are 125/250/500/1000 for the 1/2/4/8-token stages.

| Variant | Median frame ms | Median first frame ms | Full FP32 waveform exact |
|---|---:|---:|---|
| Existing codec | 4.872 | 4.776 | Yes |
| Shared stage counters | 4.750 | 4.636 | Yes |
| Selected counters plus first-frame shortcut | 4.750 | 4.540 | Yes |
| Also shorten first-frame reductions | 4.755 | 4.347 | No |
| Also omit unused first-stage Q projection | 4.756 | 4.498 | No |

The shorter reduction changes 252,549 waveform elements in the 140-frame stream, despite an approximately 122.65 dB signal-to-difference ratio. Omitting Q changes the matrix shape/implementation and also changes floating results. Neither is selected. The selected path matches all reset streams and all four first-frame probes exactly. First-frame timings here have only three samples; complete-request comparisons are separate.

An initial shared-frame-counter variant and an early stage-counter run initialized profiling after the control but before candidate timing. Their numerical checks remain useful, but the sequential timing comparison is not reliable: profiler initialization may leave driver hooks that affect subsequent graph execution. Both initial results retain an explicit timing caveat and their source snapshots. The corrected run, `codec_clock_stage_v2.json`, collects **all** timing data before any profiler initialization. No speedup claim uses the initial comparisons.

The initial-stage shortcut removes 32 attention launches from the first graph. Shared counters remove 79 kernel launches from every frame. The two-frame profile changes from **2,036 to 1,846 kernels**: 907 for the first frame and 939 for the steady frame, versus 1,018 each previously. Profiles are used to inspect work, not to supply request latency measurements.

## Boundary and state validation

The dedicated validator compares **147 output frames per path**: 140 frames covering every cache wrap, six frames across all-zero/all-1023/random reset probes, and the final allowed rotary-table frame. It poisons unused cache slots with NaNs, compares every stored element of all **68 KV buffers** as integer bit patterns, and replays graphs on a private CUDA stream. Both waveforms and caches match exactly. The next frame is rejected before launch, and the candidate's shared offsets reach exactly 4096 times their respective stage widths.

The shorter sanitizer workload compares eleven frames per path with the same reset, poisoned-cache and rotary-limit cases. It does not claim wraparound coverage; that belongs to the separate 140-frame run. **Memcheck reports zero errors. Filtered racecheck reports zero hazards, errors, or warnings.** The race filter includes the changed `_rope` kernel and original rotary/cache kernels; numerical/cache comparisons still execute the complete codec. An initial unfiltered racecheck spent more than three minutes instrumenting unchanged library work before completing its first case and was deliberately terminated. Its log and aborted-run record remain; it is not reported as a completed check of cuBLAS or the whole codec.

## Complete streaming requests and voice cloning

Twenty alternating same-process pairs compare the qualified attention-PDL LLM with the existing codec and with the selected codec. Both wrappers retain independent captured graphs and cache buffers; weights and LLM configuration are unchanged. One warmup pair is excluded, all measured requests run to completion, and profiling starts only after timing finishes.

| Measurement | Existing codec | Selected codec |
|---|---:|---:|
| Median warm cached-voice TTFA | 79.803 ms | 79.535 ms |
| p95 TTFA | 80.256 ms | 79.948 ms |
| Median preparation | 1.199 ms | 1.201 ms |
| Median prefill | 9.208 ms | 9.210 ms |
| Median of initial 32 decode-step means | 2.000 ms | 2.000 ms |
| Median first codec frame | 4.769 ms | 4.517 ms |

The median paired gain is **0.301 ms**, with 17 of 20 pairs faster. All **40 measured complete FP32 PCM streams match exactly**, as do the two warmup streams. These timings include text preparation, the full delay schedule and codec decoding, with cached cloned-voice conditioning. They exclude reference registration and network transit. Reset now clears only the authoritative shared clocks; the standalone screen above preceded that host/reset cleanup.

All **48 generated WAVs**, their generation metadata, and **eight reference WAVs** from the original and additional bilingual cloning suites match the preceding attention-PDL files byte for byte. ASR is not rerun on identical audio. Existing normalized Chinese CER of 5.37% / 1.18%, English WER of 0% / 0%, and speaker cosine of 0.9143 / 0.9149 apply to those identical files. This is equivalence to the selected calibrated G32 path, not the original BF16 model; reused four-voice machine-scored diagnostics do not establish broad or human quality acceptance.

Sequential temporary HTTP servers preserve all twenty complete PCM streams and frame counts. Median loopback TTFA is **84.971 → 83.062 ms**, with p95 **86.081 → 84.191 ms**. Both validate malformed requests and cancellation/recovery (429 while occupied, then 200). The client measures through a complete 3,840-byte/80-ms PCM chunk. Both temporary servers stop afterward.

The internal HTTP engine median is 81.054 → 79.349 ms, but unchanged preparation, initial sampling and LLM steps also vary. First codec work changes 4.774 → 4.511 ms, consistent with the paired result. **The full 1.91 ms HTTP difference is not attributed to the codec optimization**; the alternating experiment isolates the smaller 0.30 ms gain more reliably. One fresh-reference registration-plus-synthesis observation is 124.010 / 122.293 ms; registration alone is 36.636 / 35.596 ms. These single fresh-reference observations are not a latency distribution. They include server waveform parsing, reference encoding and two loopback requests; client WAV/base64 preparation precedes the timer.

An independent five-request benchmark through the ordinary CLI, without audio-head buckets, measures **80.223 ms median / 80.315 ms p95**. Its entire 36-step diagnostic dictionary and saved WAV are identical to the preceding attention-PDL benchmark. It confirms optional-flag integration; it is not an additional paired speedup estimate.

## Revisited LLM projection partitioning

Programmatic dependent launch changes the cost of a separate normalization producer, so this pass revisits the previous fusion decision. New standalone residual/RMSNorm/G32 quantization and scaled-DP4A consumers use explicit waits/hints. Twenty configurations vary which projection is split, row tiles, warp counts, and hint placement. The actual 36-layer MLP → down → following QKV ring tests three saved inputs per layer.

Of **2,052** chain/intermediate comparisons, **1,968 are exact**. Two R8 QKV configurations account for all 84 failed comparisons (116 mismatched tensor elements in aggregate) and are rejected. Wider gate/up configurations are substantially slower. Best exact split gate/up chain is **37.400 µs**, versus **36.104 µs** for the fused control.

One R4 QKV split measures 36.048 µs, but its six paired differences range from −0.051 to +0.123 µs, with only four positive rounds. This is no convincing replacement for the current fusion, and no full-request speedup is claimed. The selected LLM implementation stays unchanged. All variants, including failed and slow ones, remain in `split_projection_pdl_ring_v1.json`; the one-layer pilot is separate.

## Remaining bottleneck

The selected full-request profile covers 33 LLM decode steps and two codec chunks in a deliberately truncated 34-step diagnostic request. It contains 14,060 kernel events. A sweep of overlapping resident intervals finds 39.881 ms with projections alone, 9.516 ms with both projections and LLM attention, and 6.760 ms with attention alone. Other kernel intervals total 24.337 ms and intervals with no profiled kernel total 23.722 ms. PDL intervals include dependency waits; these are neither utilization measurements nor pure arithmetic/critical-path percentages.

Projections remain the principal target for a larger improvement. In ordinary complete requests, the initial 32 LLM decode steps average about 2.00 ms each, whereas the first codec frame is now about 4.52 ms. This codec change cannot close the remaining approximately 29.5 ms gap to the warm cached-voice target. Further projection arithmetic, memory traffic and dependency scheduling work must retain all 32 codebooks and independently check complete-stream quality. None of these measurements proves that 50 ms is impossible.

## Research context and reproduction

The current [Triton persistence tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html) discusses architecture-dependent scheduling and resource tradeoffs; its large FP16 matrix examples do not establish performance for these batch-one packed projections. The recent [Faster IndexTTS-2 paper](https://arxiv.org/abs/2607.21042) evaluates whole-pipeline acceleration and streaming on another TTS architecture, and shows that lower precision alone need not improve PyTorch performance. Its latency/quality results are not MOSS measurements. This pass therefore measures actual full-codebook codec work and retains FP32 when exact alternatives suffice.

Use `/venv/moss-vllm` and sequential GPU execution with fresh tags:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_codec_clock --tag reproduce --stage-clocks
/venv/moss-vllm/bin/python -m optimization.validate_codec_clock --tag reproduce_wrap --frames 140
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_codec_clock --tag reproduce_memcheck --frames 4
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --kernel-name kns=_rope --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_codec_clock --tag reproduce_racecheck --frames 4
/venv/moss-vllm/bin/python -m optimization.benchmark_codec_clock_paired --tag reproduce --pairs 20
/venv/moss-vllm/bin/python -m optimization.benchmark_codec_clock_http --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_split_projection_pdl --tag reproduce --layers 36 --rounds 6
```

Append `--codec-clock` to the selected full-quality/benchmark/server commands in the preceding attention-PDL report. Persistent services should use supervisor; temporary loopback validation servers stop after their checks.

## Evidence files

- Clean codec screen: [codec_clock_stage_v2.json](results/codec_clock_stage_v2.json). Initial `v1` results retain their profiler-timing caveats and source snapshots.
- Complete-request timing: [codec_clock_paired_v1.json](results/codec_clock_paired_v1.json), [stage summary](results/codec_clock_timing_summary_v1.json).
- Boundaries and sanitizers: `codec_clock_validation_wrap_v1`, `codec_clock_validation_memcheck_v1` and `codec_clock_validation_racecheck_filtered_v1` JSON/log pairs under `results/`. The unfiltered racecheck abort record remains separate.
- Exact bilingual files: [codec_clock_audio_equivalence.json](results/quality_suite/codec_clock_audio_equivalence.json).
- HTTP: [comparison](results/http_codec_clock_comparison_v1.json) and [stage summary](results/http_codec_clock_stage_summary_v1.json), with individual request, health and server logs retained.
- Independent CLI: [codec_clock_regression_summary_v1.json](results/codec_clock_regression_summary_v1.json).
- Rejected projection partition sweep: [split_projection_pdl_ring_v1.json](results/split_projection_pdl_ring_v1.json).
- Profile counts/resources and interval sweep: [codec_clock_profile_summary_v1.json](results/codec_clock_profile_summary_v1.json). Complete traces remain alongside it.
- Static/integration checks: [codec_clock_static_checks_v1.json](results/codec_clock_static_checks_v1.json). All 203 Python source files parse and eight relevant CLI help commands succeed. The original services remain ready with all 32 codebooks; temporary port 18084 is free and only the two original GPU processes remain.
- Source snapshot: [codec_clock_sources.tar.gz](results/codec_clock_sources.tar.gz), with per-file hashes and archive verification in [codec_clock_source_hashes.json](results/codec_clock_source_hashes.json). It contains 257 source/documentation/configuration files, including the licenses and prior calibration/packing plans. Measurement logs, traces, audio and model weights remain separate. [codec_clock_pass_summary_v1.json](results/codec_clock_pass_summary_v1.json) indexes the results and limitations.
