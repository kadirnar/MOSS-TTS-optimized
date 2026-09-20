# BF16 prefill QKV fusion and QKV prefetch address specialization

The corrected Triton prefill fusion measures **75.305 ms median warm cached-voice TTFA**, with **1.787 ms median paired gain** and all twenty pairs faster. The separate QKV prefetch address cleanup contributes **0.188 ms median paired gain**. Streaming voice cloning retains **all 32 codebooks**, the selected calibrated INT4/G32 decode path, BF16 prefill, FP32 codec, and complete sampling schedule. **The 50 ms target remains unmet.**

Measurements use the unchanged H200 NVL and `/venv/moss-vllm`: PyTorch 2.13 + CUDA 13.0, Triton 3.7.1, Transformers 5.14.1 and cuDNN 9.20. GPU experiments run sequentially; profiling follows timing. Neither original supervisor service is replaced.

## QKV prefetch addressing

The previous SASS audit found a runtime modulo in zero-lookahead bulk-prefetch addressing. `bulk_address.py` specializes that case to the CTA index while preserving the original projection arithmetic and dependency waits. It also supplies an experimental constant-span mode for full tiles. Both still enforce the original pointer alignment and shape constraints.

The rotating 36-layer screen compares sixteen configurations, including original/wrapper controls, individual stage changes, and combined prefix fractions. All **1,744 chain/intermediate/private-graph comparisons** match exactly. QKV-only mode 1 has 35.379 µs median chain time versus 35.567 µs control, with 0.127 µs median paired gain over six rounds. Applying the specialization to gate/up and down does not yield a consistent additional gain, so only QKV changes in the integrated `--bulk-prefetch` path.

Twenty alternating complete-request pairs measure **77.181 → 77.028 ms median**, p95 **78.355 → 77.582 ms**, with **0.188 ms median paired gain** and 18/20 pairs faster. All forty complete PCM streams and paired final CUDA RNG states agree, and all 48 context/head graph checks pass. Both modes use the same first-1/16 bulk hints; this comparison isolates addressing, not adding prefetch.

The [PTX bulk-prefetch specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch) defines a cache hint, not producer synchronization. The existing dependency wait remains before activation reads. Compiled evidence is retained in `results/prefill_qkv_audit_v2/`; the original `rem.u32` disappears from the selected QKV variant.

QKV uses the same 56 registers and 2,048 shared bytes with zero spills. SASS retains `UBLKPF.L2` before `ACQBULK`. The fused prefill kernel uses 27 registers, 16 shared bytes and zero spills. PTX, intermediate IR, cubins, SASS and cubin hashes are saved for all three audited kernels.

## Prefill fusion

Profiling the preceding preset found **1,023 kernel events** in the 160-token padded prefill graph, including separate Q/K normalization, rotary products/additions, concatenations and KV copies. `prefill_qkv.py` replaces Q/K preparation and KV writes with one Triton launch per layer. Its grid is tokens × forty heads: 32 query heads and eight key/value heads, each with dimension 128.

The kernel preserves the BF16 rounding after normalization, multiplication by head weights, and each rotary product. It writes K/V directly into FastLLM's existing buffers and returns Q with the original transpose strides. GEMMs, SDPA, weights and cache capacity remain unchanged. Validated internal callers construct unique, in-range positions; no device-to-host position inspection occurs during capture. The single-token decode path is unchanged.

The corrected operator comparison uses captured real activations from all 36 layers and eight alternating timing rounds. QKV preparation falls from **60.219 to 7.234 µs per layer**, with exact query bits/strides and full KV storage. This operator result is not TTFA; the complete-request comparison below measures its actual benefit.

`--prefill-qkv` installs per-layer flags before prefill graph capture in the benchmark, quality and HTTP entry points. It is optional and defaults off. Installation rejects incompatible dimensions, precision, pre-existing graphs and FP8 prefill. The independent paired harness keeps separate control/candidate prefill graphs on one model, including the eager fallback above the largest bucket.

### Signed-zero correction and qualification

The initial version passed actual model inputs and full requests but failed the synthetic all-zero probe: arithmetic negation emitted positive zero where the reference retained negative zero. Memcheck found no invalid accesses; that numerical failure is preserved in `prefill_qkv_validation_memcheck_v1.log`. The correction flips the floating-point sign bit for the first rotary half. This preserves unary negation for both zero signs.

Only the corrected **v2** source is selected. Both unfiltered Compute Sanitizer **memcheck and racecheck** pass **90 cases** each: ten lengths, three ordinary/suffix/end-of-capacity positions, and changing random/zero/spike inputs on private-stream CUDA graphs. All query bits/strides and entire poisoned KV buffers match. There are zero memory errors and zero race hazards/warnings. The address validator separately passes 66 cases under each sanitizer, including rejected misaligned weight buffers. These are changed-operator tests, not a sanitizer run over the whole HTTP service.

Sixteen complete prefill lengths—2, 3, 127, 128, 129, 145, 159, 160, 161, 255, 256, 257, 511, 512, 513 and 1023—match every text/audio logit and all 72 full KV buffers bitwise, including NaN-poisoned unused slots. The final length exceeds the 512-token graph bucket and exercises the eager path.

## Corrected complete-request measurements

Twenty pairs alternate order and use seeds 7000–7019, with one excluded warmup pair. The control already includes the qualified bulk hint, QKV address specialization, context/audio-head buckets, projection/attention dependency overlap, initial 32-step graph and clocked FP32 codec.

| Measurement | Control | Fused prefill |
|---|---:|---:|
| Median TTFA | 77.085 ms | **75.305 ms** |
| p95 TTFA | 78.437 ms | **76.775 ms** |
| Median prefill | 9.217 ms | **7.403 ms** |
| Median initial 32 audio steps | 61.698 ms | 61.699 ms |
| Median first codec frame | 4.521 ms | 4.526 ms |

Median paired gain is **1.787 ms**, with all twenty pairs faster. All forty complete PCM streams and paired final RNG states match. The twenty control streams also match the preceding experiment's selected output hashes. Timing shifts in unchanged stages and every outlier are retained; separate-process historical medians are not subtracted to estimate a gain.

TTFA starts with complete text and an already encoded voice reference. It includes text preparation, prefill, the complete initial delay schedule, FP32 codec decoding and the CPU copy of a playable **80-ms / 3,840-byte** PCM chunk. It excludes network transit and cached voice registration. The initial CUDA graph reports one aggregate duration; individual step timings inside that graph remain null.

The initial v1 comparison measured 75.452 ms median with 1.756 ms paired gain and identical full streams, but preceded the signed-zero correction. Its artifacts remain available; final claims use the corrected v2 run.

## Cloning quality

All **48 bilingual generated WAVs**, eight reference WAVs and generation metadata exactly match the preceding selected bulk-prefetch suites. Outputs are finite and complete. Existing normalized Chinese CER **5.37% / 1.18%**, English WER **0% / 0%**, and speaker cosine **0.9143 / 0.9149** therefore apply to identical original/expanded suite files. ASR is not rerun on unchanged audio.

These are four reused voices and small machine-scored diagnostic suites. Equivalence is to the selected calibrated G32 implementation, not original unquantized BF16 output or broad human quality acceptance.

The ordinary CLI independently completes five requests using the integrated options without audio-head buckets at **75.983 ms median / 76.218 ms p95**. Its full 36-step diagnostic dictionary and saved WAV match the preceding bulk-prefetch CLI run exactly. This verifies entry-point integration; it is not another paired speedup estimate.

## HTTP streaming and fresh references

Sequential temporary servers measure **81.725 → 79.066 ms median cached-voice HTTP TTFA**, with p95 **83.490 → 80.390 ms**, over twenty complete requests each. All twenty PCM hashes/frame counts match. Malformed-input checks pass, and cancellation returns 429 while occupied followed by successful 200 recovery. Both temporary servers stop afterward.

Internal engine median is **78.035 → 75.493 ms**; prefill is 9.238 → 7.405 ms and the initial audio interval stays about 61.81 ms. The full 2.66-ms HTTP change includes host/runtime variation. The alternating in-process experiment better isolates the 1.79-ms kernel gain.

One fresh-reference registration plus synthesis observation is **119.872 ms control / 123.574 ms candidate**. Registration alone is 35.881 / 39.290 ms. This is a regression in the single observation, not a distribution or demonstrated fresh-reference improvement. The clock includes server waveform parsing, reference encoding and two loopback HTTP requests; client WAV/base64 preparation precedes it. The 50-ms target is unmet for both cached and fresh voice-reference cases.

## Remaining bottleneck

The corrected post-timing trace contains **13,397 kernel events** for a truncated 34-step request with 33 LLM steps and two codec chunks. The prefill graph falls from **1,023 to 483 kernels**, removing 540 launches. Its 36 fused preparation kernels total about 0.237 ms in the trace. Separate traces are descriptive, not paired performance evidence.

Projection-only resident intervals total **39.632 ms**, projection/attention overlap **9.263 ms**, and attention-only **6.754 ms**. These include dependency waits and are not utilization or pure arithmetic percentages. The initial LLM interval remains about **61.7 ms** and is still the main obstacle to 50 ms. Prefill GEMMs and separate SiLU/multiply plus residual/normalization work are further measured opportunities. The next investigation should compare exact prefill pointwise fusion and projection scheduling against the current paired control. No result establishes an absolute latency limit or justifies lowering the codebook count.

## Reproduction

From the repository root, with the pinned checkpoints, calibration and packing plan already present:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_prefill_qkv_paired --tag new_run --pairs 20 --profile
/venv/moss-vllm/bin/python -m optimization.benchmark_prefill_qkv_http --tag new_run
compute-sanitizer --tool memcheck --error-exitcode 1 /venv/moss-vllm/bin/python -m optimization.validate_prefill_qkv --tag new_memcheck
compute-sanitizer --tool racecheck --error-exitcode 1 /venv/moss-vllm/bin/python -m optimization.validate_prefill_qkv --tag new_racecheck
```

Run GPU jobs sequentially. A temporary local serving command for the complete selected preset is:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant \
  --scaled-dp4a --norm-projection --audio-head-buckets --short-scales \
  --projection-pdl --attention-pdl --codec-clock --first-audio-graph \
  --bulk-prefetch --prefill-qkv
```

This binds localhost only. Persistent deployment on this instance uses supervisor as documented in the instance guide. The HTTP comparison starts each temporary process sequentially and terminates it in `finally`.

## Saved evidence

`results/prefill_qkv_pass_summary_v2.json` indexes the measured comparisons, sanitizer logs, quality/HTTP results, source checks and profile. `results/prefill_qkv_sources.tar.gz` and `results/prefill_qkv_source_hashes.json` retain the source snapshot and per-file SHA-256 hashes. The archive includes earlier implementation dependencies, reports and pinned plans; generated checkpoints, audio and profiler traces remain in their existing result folders.
