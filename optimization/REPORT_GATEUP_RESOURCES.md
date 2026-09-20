# CUDA register allocation and shared-memory spilling

This pass keeps **all 32 acoustic codebooks**, streaming voice cloning, calibrated G32 decode, BF16 prefill and the FP32 codec. It tests 26 gate/up resource variants without changing arithmetic. **No variant replaces the qualified `--gateup-compiler` path. The 50-ms goal remains unmet.**

The best candidate's median paired gains are **0.087 / 0.105 / 0.068 ms** across three complete-request comparisons. Its final p95 is slightly worse. The benefit is too small and uncertain to justify replacing the selected implementation in this pass. Serving code, selected binaries and supervisor services are unchanged.

## Implementation

All 945 source/archive files and the preceding archive hash were revalidated before experimentation. GPU jobs ran sequentially on the H200 NVL. The selected Torch 2.13/Triton 3.7.1 host loads cubins derived from the preceding isolated Triton 3.8 explicit-arithmetic PTX.

`gateup_resources.py` reassembles the one-CTA IG1/IR2 and IG2/IR4 layouts with CUDA 13.0 PTXAS. Each layout has an original-binary control and twelve combinations of register limits, dynamic/static shared allocation and shared-memory spilling. Only the PTX version, resource directives and shared declaration change; arithmetic, weights and dependency instructions remain fixed.

The bundle contains **104 cubins**, covering residual/no-residual and production/debug paths for 26 configurations. Original/modified PTX, pre-reassembly Gluon IR, SASS, compiler logs, resource metadata and hashes are retained. PTXAS SHA-256 is `daba837a68265cae38c832d13399b61dab811891de9b8914defddef143b849f2`. No driver was installed or changed.

[CUDA 13 shared spilling](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/) requires static shared memory. Those variants rewrite the original dynamic declaration and launch with zero dynamic shared bytes. Compiler spill bytes, driver local allocation and Triton spill counts are distinct: reassembled records deliberately set the last to **null**, not zero. The [PTX register limit](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#performance-tuning-directives-maxnreg) constrains allocation without guaranteeing speed.

Production residual resources:

| Setting | Registers | Local bytes/thread | Static + dynamic shared/CTA | Driver block limit/SM |
|---|---:|---:|---:|---:|
| Selected prior IG1/IR2 | 148 | 0 | 0 + 2048 B | 3 |
| IG1/IR2, cap 128 | 128 | 40 | 0 + 2048 B | 4 |
| IG1/IR2, cap 128, shared spilling | 128 | 0 | 5120 + 0 B | 4 |
| IG1/IR2, cap 96, shared spilling | 96 | 160 | 9216 + 0 B | 5 |
| IG2/IR4, uncapped | 134 | 0 | 0 + 1024 B | 3 |
| **IG2/IR4, cap 144** | **144** | **0** | **0 + 1024 B** | **3** |
| IG2/IR4, cap 128 | 128 | 0 | 0 + 1024 B | 4 |

These are residency limits, not measured occupancy. The 144-register limit produces more registers than uncapped IG2/IR4 because allocation/scheduling changes. Its residual production cubin SHA is `9dbcd8ed2b974e236d751eddb5ed46583dd2b1a21f4a84ca9e3231d8764d5cfc`.

## Screening and complete requests

The pilot passes **81** comparisons. The all-layer ring passes **2,916** comparisons and 27 private graphs, including normalized/quantized states, following projections and entire poisoned KV buffers. It uses all 36 weight sets and three frozen inputs per layer, nearest saved attention metadata and a synthetic layer-35 wrap to layer 0. This is a chain screen, not a full trajectory or TTFA measurement.

| Configuration | Chain median | Median paired gain | Faster rounds |
|---|---:|---:|---:|
| Selected prior | 36.557 µs | — | — |
| Same selected cubin loaded as candidate | 36.572 µs | −0.035 µs | 2/6 |
| **IG2/IR4, cap 144** | **36.412 µs** | **0.130 µs** | **6/6** |
| IG2/IR4, cap 128 | 37.159 µs | −0.603 µs | 0/6 |
| IG1/IR2, cap 128, shared spilling | 37.842 µs | −1.295 µs | 0/6 |
| IG1/IR2, cap 128 | 38.808 µs | −2.261 µs | 0/6 |
| IG1/IR2, cap 96 | 45.327 µs | −8.747 µs | 0/6 |

Moving spills to shared memory helps the capped IG1/IR2 variant relative to ordinary spilling, but still loses to the selected uncapped kernel.

Full-request modes own separate context/head/initial-audio graphs and matching eager dispatch. Weights, cloned-voice conditioning and codec are shared unchanged. One warmup group is excluded per process; all measured samples and outliers remain. TTFA includes text preparation and first codec decoding, excluding reference registration and network transit.

| Run | Control median | Cap-144 median | Cap-144 paired gain | Faster |
|---|---:|---:|---:|---:|
| Initial, 12 triplets | 71.808 ms | 71.448 ms | 0.087 ms | 9/12 |
| Repeat, 20 triplets | 73.530 ms | 73.250 ms | 0.105 ms | 13/20 |
| Final, 20 pairs | 71.238 ms | 71.178 ms | 0.068 ms | 14/20 |

Cap-128 has paired gains of 0.167/0.049 ms in the first two runs despite losing the frozen-input ring. It was not taken into the final comparison. Final cap-144 p95 is **73.779 → 73.893 ms**. Descriptive within-run paired bootstrap intervals include zero in all three runs; the final interval is approximately **−0.016 to +0.138 ms**. Serial runtime shifts, screening/selection, repeated seeds and small samples prevent a general latency guarantee. Separate processes are not pooled into an inferential estimate.

Both initial and final processes shift into a faster phase affecting unchanged stages. Final prefill is 6.682/6.679 ms and first codec 4.192/4.189 ms, versus roughly 6.82/4.53 ms in previous runs. The initial 32-step interval is 58.523/58.408 ms. **The approximately 2-ms absolute change versus older runs is not a kernel gain.**

All **136 measured complete float32 PCM streams** and final RNG states match their controls; **52 control hashes** match prior output. **240 full-model graph cases** preserve logits, IDs, RNG and all 72 entire poisoned KV buffers across cache/head boundaries. Equivalence is to preceding calibrated G32, not upstream BF16.

## Validation and next bottleneck

Memcheck, racecheck and synccheck each pass **252 cases**, with zero errors; racecheck also reports zero hazards/warnings. Four configurations cover cap 144, cap 128 and two shared-spilling alternatives. Cases exercise three real layers, real/zero/spike inputs, residual/no-residual and production/debug binaries, and chains at cache positions 0/127/1023. This is operator-chain coverage, not whole-service sanitization.

The post-timing profile has **12,101 kernel events** over 33 decode steps and two codec frames. Gate/up and down median resident intervals are **18.912 / 25.760 µs**, with summed intervals **22.725 / 31.057 ms**. These include waits and overlap; they are not utilization, critical-path percentages or a lower bound. The initial LLM delay schedule remains the largest TTFA component.

The next structural hypothesis is finer dependency handling across projections and attention, checked against intermediate tensors. [ForgeMegakernel](https://arxiv.org/html/2609.12379v1) motivates per-SM work streams and dependency counters and cautions against assuming coarse global-barrier fusion will win. [Ada-MK](https://arxiv.org/html/2605.11581v1) and [EventTensor](https://arxiv.org/html/2604.13327v2) provide other scheduling designs. Their hardware, precision and workloads differ, so they supply hypotheses, not predicted MOSS gains. No new persistent kernel or per-head readiness protocol is implemented in this pass. No evidence establishes that 50 ms is impossible; all 32 codebooks remain required.

## Reproduction

Run GPU jobs sequentially from `/workspace/MOSS-TTS`, with fresh tags:

```bash
/venv/moss-vllm/bin/python -m optimization.gateup_resources --tag NEW
/venv/moss-vllm/bin/python -m optimization.benchmark_gateup_resources --tag NEW --bundle gateup_resource_bundle_NEW --layers 36 --rounds 6
/venv/moss-vllm/bin/python -m optimization.benchmark_gateup_resources_paired --tag NEW --up-bundle optimization/results/gateup_resource_bundle_NEW --configs ir4_r144_dynamic --rounds 20 --profile
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 /venv/moss-vllm/bin/python -m optimization.validate_gateup_resources --tag NEW --bundle gateup_resource_bundle_NEW --configs ir4_r128_dynamic ir4_r144_dynamic ir2_r128_spill ir2_r96_spill
```

Repeat the last command with racecheck/synccheck and separate tags. The exporter requires the prior screen bundle and pinned assembler path. [gateup_resources_pass_summary_v1.json](results/gateup_resources_pass_summary_v1.json) indexes results; `analyze_gateup_resources.py` reconstructs it and verifies hashes. [gateup_resources_source_hashes.json](results/gateup_resources_source_hashes.json) records the source/binary archive verification.

This pass introduces no serving flag, new HTTP/fresh-reference timing claim or new broad cloning-quality claim. The preceding qualified compiler report remains the reference for those measurements.
