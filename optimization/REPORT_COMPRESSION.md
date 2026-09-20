# Lossless CUDA allocation compression

This pass keeps **all 32 codebooks**, streaming voice cloning, the selected calibrated G32 arithmetic, BF16 prefill and the FP32 codec. An optional allocation change improves cached-voice TTFA by **0.15–0.21 ms median paired gain** in two fifteen-round comparisons. This is a small gain with request-level noise; **50 ms remains unmet**. Existing supervisor services and their voice caches are unchanged.

## Hardware support and ownership

The H200 NVL reports both virtual-memory management and generic compression support. `compressed_alloc.cpp` uses the installed CUDA 12.8 Driver API to create physical allocations, reserve/map virtual addresses and grant device access. It checks the properties returned by `cuMemGetAllocationPropertiesFromHandle` for **every allocation** and rejects silent fallback. Neither the driver nor Torch's global allocator is replaced.

An owning DLPack capsule exposes each allocation to Torch. Views retain storage ownership; final storage destruction waits for the owning CUDA context before unmapping, releasing the physical handle and freeing the address reservation. Allocation occurs outside graph capture. The caller must retain tensor owners until graphs using their pointers are destroyed. The Python wrapper is restricted to the validated SM90 architecture. Copies use the caller's current stream, with ordinary stream dependencies still required.

This hardware feature is lossless and does **not** shrink an allocation's memory footprint. Actual bandwidth savings depend on data patterns. On this instance, VMM has **2 MiB minimum granularity**, which can enlarge small buffers. These allocations are external to Torch's caching-allocator statistics; native counters and per-buffer logical/allocated sizes are reported separately.

Forty-eight random-bit tests cover eight scalar dtypes, three sizes and ordinary/compressed VMM. They check exact initial copying, private-stream graph replay, changed contents, view lifetime and final cleanup. An unconsumed capsule is also released correctly. The first sanitizer run failed a test-side immediate-cleanup assertion, with zero reported memory errors. The test now explicitly drops the final view and synchronizes CUDA before checking allocation counters; the allocator implementation was unchanged. The final memcheck run passes all 48 cases with **zero errors, zero live allocations and zero cleanup errors**. Both logs are retained.

Sources: [NVIDIA's Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/), [CUDA virtual-memory documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html), and the [CUDA 12.8 compressible-memory sample](https://github.com/NVIDIA/cuda-samples/tree/v12.8/Samples/3_CUDA_Features/cudaCompressibleMemory). This implementation follows the documented API contract and uses installed headers; no sample source is vendored. It does not use nvCOMP or a Blackwell decompression engine.

## Actual-buffer screening

`benchmark_compressed_projection.py` rotates all 36 actual layer weights, scales, inputs and normalization buffers. The kernel and arithmetic stay identical. Five variants compare Torch allocations, ordinary VMM, compressed weights only, compressed scales only and both compressed. Eight timing rounds rotate and reverse order. All **576** private-stream output/intermediate comparisons and all copied buffers match exactly.

| Projection stage | Torch µs | Plain VMM µs | Compressed weights µs | Compressed scales µs | Both compressed µs |
|---|---:|---:|---:|---:|---:|
| Fused normalization/QKV | 8.698 | 8.723 | 8.708 | 8.711 | 8.728 |
| Fused normalization/gate-up | 19.379 | 19.474 | 19.630 | 19.390 | 19.658 |
| Attention output | 4.959 | 4.967 | 4.954 | 4.934 | 4.936 |
| MLP down | 10.028 | 10.024 | 10.190 | 9.864 | 10.036 |

Only FP32 output/down scales show a useful screening gain. They contain the unchanged exported BF16 values promoted to FP32, so their low sixteen bits are zero. The result is consistent with a compressible pattern, but hardware performance counters are unavailable here: no compression ratio or exact traffic reduction is measured. Dense packed INT4 weights do not benefit in this test.

The auxiliary tests reject broader compression:

- **FP32 codec:** 687 decoder/output-projection parameter and projected-codebook buffers are tested with ordinary and compressed VMM. All 66 fixture frames match exactly on a private stream. Median graph time is **4.7777 ms Torch / 4.7790 ms plain VMM / 4.8707 ms compressed**. First-frame CPU PCM time is 4.7919 / 4.7930 / 4.8911 ms. VMM rounds 3.61 GB of logical contents up to 4.83 GB. This option is not installed in serving.
- **BF16 audio heads:** 96 private-stream comparisons across twelve synthetic hidden rows and the existing 8/16/24/32 head-prefix shapes are exact. Compression gives small gains at prefixes 8 and 16 but regresses at 24 and 32; it is not selected. These are stage tests without backbone traffic, not complete-request measurements. Every emitted codec frame still requires all 32 channels.

## Complete streaming comparison

The candidate replaces only **72 FP32 output/down scale buffers**. Their combined logical and allocated sizes are both **288 MiB**: the selected buffers incur no rounding overhead. `enable_compressed_scales()` builds and verifies all replacements before mutating the model, then installs them before graph capture. No weight code, scale value, activation, head count or sampling rule changes.

`benchmark_compressed_scales_paired.py` retains three graph sets in one process: original Torch storage, ordinary VMM control and compressed VMM. Every set shares model weights and KV state; scale attributes and graph dispatch are restored together for each complete utterance, including eager steps. Order rotates/reverses, and one warmup triplet is excluded in each run. Tests use the existing 145-token Chinese prompt and cached reference voice; waveform parsing/reference encoding and network transport are excluded.

| Fifteen-round run | Torch median ms | Plain VMM median ms | Compressed median ms | Compressed paired gain ms | Faster rounds |
|---|---:|---:|---:|---:|---:|
| First | 84.571 | 84.579 | 84.430 | 0.206 | 14/15 |
| Independent repeat | 84.503 | 84.492 | 84.329 | 0.151 | 9/15 |

Compressed p95 is 84.949 ms in the first run and 84.828 ms in the repeat, versus Torch 87.011 and 84.778 ms. The repeat therefore has a slightly worse candidate p95. Plain VMM has median paired gains of −0.003 and −0.017 ms. All **90 measured complete streams** match their corresponding triplet control and finish without truncation; the two excluded warmup triplets also match.

Twelve additional rotating full-decode GPU timings in each process give median paired gains of **6.917 and 5.760 µs per decode**. Prefill, preparation and first codec timing remain similar. This supports a small decode benefit; the whole p95 difference is not attributed to compression. `compressed_scales_timing_audit_v1.json` retains request phases and graph timings. Each process also passes 96 private-stream graph checks covering context boundaries, fallback capacity and three active-prefix sizes, with every text/audio logit and sampled ID exact.

The original sixteen-utterance and additional thirty-two-utterance cloning suites, plus all eight saved reference WAVs, match their selected G32 controls byte for byte. Frame counts, text, seeds and finite/nontruncated status also match. Existing normalized diagnostic scores apply to these identical artifacts without rerunning ASR: original-suite Chinese CER **5.37%**, additional-suite CER **1.18%**, English WER **0%** on both. This is equivalence to the existing calibrated G32 path, not to original BF16, and these reused four-voice suites are not broad or human quality acceptance.

The candidate profile still assigns **32.706 + 17.254 = 49.960 ms**, or **59.42% of GPU time**, to fused normalization/projection consumers and remaining projections. This remains the principal bottleneck. Compression of the dense weight matrices is not a solution to it.

The separate five-request benchmark without head prefixes measures **85.264 ms median / 85.649 ms p95**. Its 36-step diagnostic dictionary and saved WAV match the earlier G32 norm-projection control exactly. This exercises the benchmark CLI installation path independently; it is not an additional paired speed comparison.

## HTTP, fresh-reference and cancellation checks

Two temporary loopback servers run sequentially with the same head/context buckets, text, reference and seeds. The control then compressed candidate measure **88.203 → 87.974 ms median**, p95 **90.582 → 88.969 ms**, across twenty measured complete streams each. All twenty corresponding PCM hashes and frame counts match. Input-error checks pass; cancellation/recovery returns `429` then `200` for each server. Both temporary processes are stopped, and the original supervisor service PIDs remain unchanged.

The recorded internal engine medians are **84.606 / 84.430 ms**. Initial decode step medians are 2.1578 / 2.1494 ms; prefill is 9.2084 / 9.2080 ms, and first codec decode is 4.7715 / 4.7714 ms. The sequential HTTP result is consistent with the small paired decode gain, but includes CPU/network scheduling variation. It is not used to attribute the entire p95 difference to compression.

One continuous reference-registration-plus-synthesis observation takes **134.97 / 127.64 ms**, including server WAV parsing, reference encoding and two loopback requests. Registration alone takes 37.26 / 36.15 ms. These single observations are retained for scope completeness, not claimed as a repeatable fresh-cloning speedup. Base64/WAV construction occurs before that timer. Cached-voice measurements exclude reference registration; neither metric establishes 50 ms for fresh voice cloning.

## Reproduction and scope

Use `/venv/moss-vllm` (Torch 2.13, Triton 3.7.1) and run GPU jobs sequentially with fresh result tags. Existing pinned checkpoints, calibration exports, raw input captures and fixture are required.

```bash
/venv/moss-vllm/bin/python -m optimization.validate_compressed_alloc --tag reproduce
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_compressed_alloc --tag reproduce_memcheck
/venv/moss-vllm/bin/python -m optimization.benchmark_compressed_projection --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_compressed_auxiliary codec --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_compressed_auxiliary heads --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_compressed_scales_paired --tag reproduce --rounds 15
/venv/moss-vllm/bin/python -m optimization.benchmark_compressed_scales_http --tag reproduce
```

Add `--compressed-scales` to the selected G32 norm-projection commands in `REPORT_NORM_PROJECTION.md`; `quality_generate` and `server` also support `--audio-head-buckets`. The compression flag is optional and defaults off. G128 combinations are rejected. A temporary development endpoint remains bound to loopback; a persistent service should use supervisor.

Raw results are `compressed_alloc_validation_*`, `compressed_projection_v1.json`, `compressed_codec_v1.json`, `compressed_heads_v1.json`, `compressed_scales_paired_v1/v2.json`, their logs/profiles, `http_compressed_scales_comparison_v1.json`, `http_compressed_scales_stage_summary_v1.json`, `compressed_scales_regression_comparison_v1.json`, and `quality_suite/compressed_scales_audio_equivalence.json`. `compression_pass_summary_v1.json` indexes the pass. `compression_sources.tar.gz` and `compression_source_hashes.json` preserve code/docs/plans and licenses separately from measured outputs and checkpoints.
