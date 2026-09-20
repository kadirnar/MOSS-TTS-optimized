# Hopper bulk L2 prefetch for the selected projection path

Prefix-only asynchronous L2 hints give a small repeated improvement while retaining **all 32 codebooks** and streaming voice cloning. Two independent twenty-pair full-request comparisons measure **77.257 / 77.274 ms median warm cached-voice TTFA**, with **0.446 / 0.483 ms median paired gain** over the qualified initial-audio-graph path. **The 50 ms target remains unmet.** All measured full PCM streams and final RNG states match their controls exactly.

The H200 NVL, original two supervisor services, GPU clocks and installed runtime are unchanged. The experiment uses the selected `/venv/moss-vllm`: PyTorch 2.13 + CUDA 13.0, Triton 3.7.1, Transformers 5.14.1 and cuDNN 9.20. GPU tests run sequentially, and timing precedes profiling.

## Change and hardware constraints

`bulk_prefetch.py` uses Hopper's `cp.async.bulk.prefetch.L2.global` instruction before a projection's existing programmatic dependency wait. Only one thread in each CTA issues each hint. Immutable packed weights can be requested before producer activations become available; the original `griddepcontrol.wait` remains in the arithmetic body before activation reads. Prefetch hints supply no producer synchronization and are not assumed to complete at any particular time.

The selected setting requests only the **first 1/16 of each CTA's contiguous weight span** in QKV, gate/up and down projections. That is 1,024 bytes for QKV, 4,096 bytes for each gate and up branch, and 1,536 bytes for down. It uses no lookahead, no scale hint and no eviction-policy descriptor. Attention-output projection keeps its existing scale preload. No new weight packing, quantization or model arithmetic is introduced.

The original Gluon arithmetic bodies, host shape checks and allocations are reused. A private copy of each host callable replaces only its kernel dispatcher, so multiple variants coexist without mutating the source module. Additional checks reject misaligned weight/scale addresses before launch. Source addresses and hint sizes obey the PTX 16-byte alignment/multiple requirements. The opt-in `--bulk-prefetch` entry points install per-model/per-module callables before graph capture; ordinary serving does not replace global functions.

The isolated paired benchmark uses temporary function bindings to capture both graph sets on one model and always restores them. This is confined to its single-thread test process. Serving, quality generation and the independent CLI use the per-model integration. Existing CUDA graph ownership and one-request-at-a-time state rules still apply.

## Screened configurations

The pilot tests eight configurations on one layer. It validates compilation and exact outputs but is not used to select the fastest full-model path. The complete screen rotates all **36 actual layers**, with three recorded inputs per layer and a private-stream graph comparison for each configuration. Each timing round rotates/reverses configuration order and uses the existing nine-replay, three-ring CUDA-event method.

There are **53 configurations**: original and wrapper controls, 36 combinations of stage, prefix fraction, scale hints and cache policy, twelve CTA-lookahead choices, and three combined-stage settings. All **5,777 output/intermediate comparisons** are exact, with no failed configuration. The two controls distinguish any wrapper effect from the prefetch instruction itself.

| Chain configuration | Median µs |
|---|---:|
| Original control | 36.025 |
| Wrapper with hints disabled | 36.031 |
| All three stages, first 1/16 | **35.552** |
| Gate/up only, first 1/16, policy hint | 35.738 |
| Gate/up only, first 1/16 | 35.792 |
| All three stages, complete spans | 42.868 |

The selected chain has **0.487 µs median paired gain**, with all six timing rounds faster. Full-span prefetch is substantially slower. These are stage-chain measurements, not TTFA. Register/shared-memory use stays at 162 registers / 1,024 bytes for gate/up, 128 / 1,024 for down, and 56 / 2,048 for QKV, with no spills.

The compiler audit retains PTX, intermediate IR, cubins and SASS for original, prefix and full-span variants. Actual SM90 code contains **`UBLKPF.L2` before `ACQBULK`**, and every PTX hint precedes the dependency wait. The initial audit's opcode filter recognized neither the actual `UBLKPF` spelling nor its lines; its original summary is retained, and the corrected summary parses the already saved binaries without changing measurements.

## Complete-request comparisons

The two runs alternate control and candidate with seeds 7000–7019 and one excluded warmup pair. Both retain calibrated INT4/G32 projections, BF16 prefill, exact short scales, projection/attention PDL, the FP32 clocked codec, context/audio-head buckets and the initial 32-step graph. Each request completes before the next begins.

| Measurement | Run 1 control | Run 1 hint | Run 2 control | Run 2 hint |
|---|---:|---:|---:|---:|
| Median TTFA | 77.658 ms | 77.257 ms | 77.775 ms | 77.274 ms |
| p95 TTFA | 77.904 ms | 77.734 ms | 78.768 ms | 78.013 ms |
| Median initial 32 audio steps | 62.319 ms | 61.866 ms | 62.384 ms | 61.875 ms |
| Median prefill | 9.213 ms | 9.214 ms | 9.215 ms | 9.216 ms |
| Median first codec frame | 4.514 ms | 4.521 ms | 4.524 ms | 4.520 ms |

Median paired gain is **0.446 / 0.483 ms**, with **18/20 and 19/20 pairs faster**. All **80 measured complete PCM streams** and final CUDA RNG states match within their pairs; warmup streams match too. The first run's twenty controls match the preceding initial-audio pass's PCM hashes. Separate-process historical medians are not subtracted to estimate this gain.

TTFA includes complete text processing, the full initial delay schedule, first codec decoding and the CPU copy of a playable 80-ms PCM chunk. It excludes cached voice registration and network transit. The 32-step graph reports one observed aggregate time; its per-step timing entries remain null.

## Correctness and memory safety

Each full-model paired process also checks **48 context/head cases**, comparing every text/audio logit and sampled ID at positions including context boundaries and the final valid KV slot. These add 96 exact graph comparisons across both processes.

The dedicated validator covers **66 changing-input private-stream graph cases**: actual and odd output row counts for all three projection stages, full-span/prefix hints, cache policy and wrapped lookahead. Every output and returned intermediate agrees exactly with the original PDL bodies. Eleven contiguous but deliberately misaligned weight buffers reject before launch. Compute Sanitizer memcheck reports **zero errors**, and unfiltered racecheck reports **zero hazards, errors or warnings** for that validator. This is projection validation, not a sanitizer run over the whole HTTP service.

## Cloning quality

The integrated per-model option generates **48 bilingual WAVs** identical to the preceding initial-audio-graph suites. All **eight reference WAVs** and generation metadata agree too, with complete, finite output throughout. Existing normalized Chinese CER **5.37% / 1.18%**, English WER **0% / 0%**, and speaker cosine **0.9143 / 0.9149** apply to the identical original/expanded suite files. ASR is not rerun on unchanged audio.

These are reused four-voice machine-scored diagnostics. Exact equivalence is to the selected calibrated G32 implementation; this is neither original-BF16 equivalence nor broad human quality acceptance.

The ordinary benchmark CLI, using the integrated option without audio-head buckets, completes five requests at **77.988 ms median / 78.192 ms p95**. Its full 36-step diagnostic dictionary and saved WAV match the preceding initial-audio CLI run exactly. This independently checks entry-point integration; it is not another paired speedup estimate.

## HTTP streaming

Twenty complete cached-voice requests through sequential temporary loopback servers measure **81.542 → 80.435 ms median TTFA**, with p95 **82.778 → 81.314 ms**. All twenty full signed-16-bit PCM streams and frame counts match. Both servers pass malformed-input and cancellation/recovery checks, with 429 while occupied followed by 200. Temporary servers stop afterward; the original services remain unchanged.

The internal engine median changes **77.834 → 77.273 ms**. Initial audio graph work changes 62.398 → 61.893 ms, while first codec work stays at 4.524 ms. The larger 1.11-ms HTTP difference includes host/runtime variation and is not attributed entirely to the prefetch hint; the repeated paired experiment better isolates the 0.45–0.48-ms gain.

One fresh-reference registration plus synthesis observation is **121.908 / 118.380 ms**; registration alone is 35.840 / 33.643 ms. These are single observations, not distributions. Server waveform parsing, reference encoding and two loopback requests are included, while client WAV/base64 preparation precedes timing. All TTFA observations end at a complete 3,840-byte / 80-ms PCM chunk.

## Remaining bottleneck

The post-timing profile still contains 13,937 kernel events in a truncated 34-step request with 33 LLM steps and two codec chunks. Projection/attention counts remain 4,752 / 3,564. Projection-only resident intervals total 39.933 ms, projection/attention overlap 9.227 ms and attention-only 6.743 ms. These intervals include PDL waits and are neither utilization nor pure arithmetic percentages; separate traces are not paired performance evidence.

The initial LLM interval remains about **61.87 ms**. This hint improves memory scheduling slightly but leaves approximately **27.3 ms** between observed warm TTFA and the target. Projection computation/traffic remains the principal optimization target; BF16 prefill at about 9.22 ms is another measured opportunity. No result establishes that 50 ms is impossible, and codebook count stays at 32.

The SASS audit also identifies a concrete follow-up: the generic lookahead expression retains a runtime `rem.u32` even when lookahead is zero, lowering to reciprocal/integer correction instructions before the hint. Removing that unused modulo needs a new numerical/timing comparison; no gain from its removal is claimed in this pass.

## Research context and reproduction

The current [PTX bulk-prefetch specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch) defines this instruction as a nonblocking cache hint available on SM90, with 16-byte address/size constraints. The newly documented direct eviction-priority suffix requires a newer architecture and is not used; the screen's optional policy uses the older cache-policy descriptor form supported by this GPU.

The recent [Leech-lattice serving-layout preprint](https://arxiv.org/abs/2609.02652) emphasizes that actual device layout, launch geometry and decode work determine batch-one performance. The [lossless weight-compression paper](https://arxiv.org/abs/2606.15789) also studies serving gains from memory savings, particularly feasible batch size. Those results concern other formats/workloads and are not predictions for this MOSS checkpoint. This pass preserves the current weights and measures a hardware hint on the real layer chain and complete streams.

Run sequentially with fresh tags:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_bulk_prefetch --tag reproduce --layers 36 --rounds 6
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_bulk_prefetch --tag reproduce_memcheck
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_bulk_prefetch --tag reproduce_racecheck
/venv/moss-vllm/bin/python -m optimization.benchmark_bulk_prefetch_paired --tag reproduce --pairs 20 --profile
/venv/moss-vllm/bin/python -m optimization.audit_bulk_prefetch --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_bulk_prefetch_http --tag reproduce
```

Append `--bulk-prefetch` to the complete selected server/quality/benchmark commands in [the initial audio graph report](REPORT_FIRST_AUDIO_GRAPH.md). Persistent services use supervisor; temporary local validation servers stop after testing.

## Evidence

- [Full 53-configuration screen](results/bulk_prefetch_ring_v1.json); the eight-configuration pilot is retained separately.
- [First paired run](results/bulk_prefetch_paired_v1.json), [repeat](results/bulk_prefetch_paired_v2.json), and [timing/stage summary](results/bulk_prefetch_timing_summary_v1.json). The pre-profile-option source snapshot for the first run remains alongside its results.
- `bulk_prefetch_validation_memcheck_v1` and `bulk_prefetch_validation_racecheck_v1` JSON/log pairs under `results/`.
- [Profile summary](results/bulk_prefetch_profile_summary_v1.json), with full trace and table retained.
- [Cloning WAV/reference/metadata equivalence](results/quality_suite/bulk_prefetch_audio_equivalence.json).
- [HTTP comparison](results/http_bulk_prefetch_comparison_v1.json) and [HTTP stage summary](results/http_bulk_prefetch_stage_summary_v1.json), with request, health and server logs retained.
- [Independent CLI regression](results/bulk_prefetch_regression_summary_v1.json).
- [PTX/cubin/SASS audit](results/bulk_prefetch_audit_v1/summary.json), with the initial opcode-filter summary preserved.
- [Static, CLI and service checks](results/bulk_prefetch_static_checks_v1.json).
- [Source archive](results/bulk_prefetch_sources.tar.gz), [per-file hashes and archive verification](results/bulk_prefetch_source_hashes.json), and [pass index](results/bulk_prefetch_pass_summary_v1.json). Measurement traces, logs, WAVs and model weights remain outside the source archive.
