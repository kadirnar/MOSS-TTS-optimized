# Prefill pointwise fusion and split-normalization decode experiments

Fused BF16 prefill activation/product and residual/normalization reduce warm cached-voice TTFA to **74.730 ms median**, with **0.644 ms median paired gain** over the qualified fused-QKV preset. All 32 codebooks and streaming voice cloning remain enabled. All complete measured streams and final RNG states match their controls. **The 50 ms target remains unmet.**

The H200 NVL, original two supervisor services and selected runtime remain unchanged: PyTorch 2.13 + CUDA 13.0, Triton 3.7.1, Transformers 5.14.1 and cuDNN 9.20. GPU tests run sequentially, with profiling after timing. The selected implementation retains calibrated INT4/G32 decode, BF16 prefill and the FP32 clocked codec.

## Selected prefill changes

The preceding profile showed 36 separate SiLU kernels, 36 product kernels, and 72 residual additions followed by normalization. `prefill_pointwise.py` fuses SiLU/product into one Triton kernel and uses the existing fused residual/RMSNorm kernel in an explicit prefill backbone loop. Both preserve the BF16 intermediate rounding boundaries. GEMMs, SDPA, QKV preparation, sampling and single-token decode stay unchanged.

The activation screen tests **21 alternatives**: direct exponential/division, libdevice exponential with rounded division, and a 128-KiB lookup table containing the installed PyTorch SiLU result for every BF16 bit pattern, across seven block/warp combinations. All **819 comparisons** pass bitwise: 36 real layer tensors plus the complete 65,536-pattern gate domain with three up values for every configuration. The table is an arithmetic lookup, not cached speech or model output. It is slower and is not installed in the selected path.

| Activation preparation | Median per-layer µs |
|---|---:|
| Separate PyTorch SiLU and product | 15.584 |
| Selected direct, block 512 / four warps | **4.663** |
| Libdevice, block 512 / four warps | 4.783 |
| Best lookup-table tile, block 2048 / four warps | 5.317 |

Six rounds rotate all 36 actual activation tensors and rotate/reverse configuration order. The selected kernel uses 18 registers and no shared memory or spills. Separately, all **72 real residual/normalization sites** match bitwise; mean-site graph timing improves **5.273 → 3.313 µs**. These are operator timings, not TTFA.

The fused residual/normalization kernel uses 55 registers, 16 shared bytes and no spills. `results/prefill_pointwise_audit_v1/` retains PTX, intermediate IR, cubins, SASS and hashes for both selected kernels and the control/uncapped/capped decode experiments.

The optional `--prefill-pointwise` entry point installs the selected activation tile and explicit residual loop before graph capture. It is wired into serving, quality generation and the ordinary benchmark CLI, and defaults off. Installation rejects incompatible BF16 Qwen3-8B projection shapes, FP8 prefill and existing graphs. No lookup table is allocated by the selected option.

## Full-request ablation

One model holds four prefill graph sets: control, activation only, residual only, and both changes. Decode/codec graphs remain shared. Twenty measured rounds use seeds 7000–7019, rotating/reversing mode order; one warmup round is excluded. Every request runs to completion before the next.

| Mode | Median TTFA | p95 TTFA | Median paired gain | Faster rounds |
|---|---:|---:|---:|---:|
| Fused-QKV control | 75.317 ms | 76.289 ms | — | — |
| Activation only | 74.891 ms | 75.378 ms | 0.444 ms | 18/20 |
| Residual/normalization only | 75.224 ms | 76.010 ms | 0.188 ms | 16/20 |
| Both, selected | **74.730 ms** | **75.443 ms** | **0.644 ms** | **18/20** |

All **80 measured complete PCM streams** and final CUDA RNG states match within each round. The twenty controls also match the preceding fused-QKV pass's selected stream hashes. Both slower combined-mode pairs and all timing outliers are retained. Median differences are not substituted for the paired-gain statistic.

Combined prefill decreases **7.405 → 6.798 ms**. Initial 32-step audio generation remains **61.754 / 61.752 ms**, and first codec work **4.522 / 4.521 ms**. TTFA includes complete text preparation, prefill, the full delay schedule, first FP32 codec decode and CPU copy of a playable 80-ms PCM chunk. Voice-reference encoding is cached for this measurement; network transit is excluded.

## Correctness and cloning

Each of the three candidate modes matches the control over sixteen full-prefill lengths through 1023 tokens: **48 full-backbone comparisons**, covering graph boundaries and the eager fallback. Every text/audio logit and all 72 entire KV buffers agree bitwise, including NaN-poisoned unused slots.

Compute Sanitizer **memcheck and unfiltered racecheck each pass 36 private-stream graph cases**: changed random/zero/spike inputs at eleven lengths through 1023, plus the complete BF16 gate domain with three up values. Activation products, residual sums and normalized outputs match bitwise, including tested zero signs and NaN bits. There are zero errors, hazards or warnings. These tests cover the changed operations, not the complete HTTP service under a sanitizer.

All **48 Chinese/English cloning WAVs**, eight reference WAVs and generation metadata exactly match the preceding selected fused-QKV suites. Output is finite and complete. Prior normalized Chinese CER **5.37% / 1.18%**, English WER **0% / 0%** and speaker cosine **0.9143 / 0.9149** apply to identical audio; ASR is not rerun. This is equivalence to the selected calibrated G32 implementation, not original BF16 equivalence or broad human quality acceptance. The suites reuse four voice assets.

The independent ordinary CLI completes five requests with the integrated flags, without audio-head buckets, at **76.353 ms median / 76.542 ms p95**. Its complete 36-step diagnostic dictionary and saved WAV match the preceding fused-QKV CLI exactly. This separate-process result is slower than the preceding CLI observation and is an integration check, not a paired speedup claim.

## HTTP streaming

Twenty complete cached-voice requests through sequential temporary loopback servers measure **78.926 → 78.438 ms median TTFA**, with p95 **79.458 → 79.268 ms**. All twenty full PCM hashes/frame counts match, malformed-input checks pass, and cancellation returns 429 while occupied followed by 200 recovery. Both temporary servers stop afterward; original supervisor processes and voice caches remain intact.

Internal engine median is **75.348 → 74.882 ms**. Prefill is 7.408 → 6.835 ms, while the initial audio graph and codec remain approximately unchanged. Host/runtime variation affects sequential HTTP results; the four-mode comparison above better isolates the 0.644-ms paired kernel gain.

One fresh-reference registration plus synthesis observation is **119.701 ms control / 117.887 ms candidate**, with registration alone 37.119 / 35.472 ms. These are single observations, not distributions. The interval includes server waveform parsing, reference encoding and two loopback requests; client WAV/base64 preparation precedes timing. The 50-ms target remains unmet for both cached and fresh-reference cases.

## Decode experiment: compute normalization once

The large remaining decode interval motivated `dp4a_split_norm_pdl.py`. It computes residual addition, RMSNorm and G32 activation quantization once in a single-CTA producer, then launches projection consumers with programmatic dependency overlap. Consumers prefetch immutable weights before waiting, load quantized activations after the wait, and reuse the original exact grouped projection arithmetic. This avoids repeating normalization in every projection CTA but introduces another launch and global intermediate buffers.

The first 36-layer screen tests ten alternatives plus control: QKV, gate/up or both split at three producer trigger positions, plus a no-hint combined case. All **1,199 chain/intermediate/private-graph comparisons** match. Every alternative is slower:

| Chain | Median µs |
|---|---:|
| Selected fused-normalization control | **35.322** |
| Best QKV split | 35.817 |
| Best gate/up split | 40.802 |
| Best combined split | 41.191 |

The split gate/up projection grows from 162 to 192 registers, despite reducing shared storage from 1,024 to 64 bytes. A second eight-configuration screen tests explicit register caps 128, 144, 160, 168, 176 and 192, alongside uncapped/control cases. All **872 comparisons** pass. The best cap requests 168 registers and compiles to 167 without spills; its chain improves to **37.176 µs**, still slower than **35.378 µs** control. The 128-register cap spills and is slowest at 44.526 µs. Smaller shared memory and less redundant arithmetic do not establish lower latency.

Together these screens contain **2,071 exact comparisons**. Six representative schedules, including 160/128 register caps, pass **54 changed-input private-graph cases under each sanitizer** with zero errors/hazards/warnings. The 128-register variant spills; the 160-register variant does not. No split-normalization variant is selected, and no full-request speedup is claimed for them.

NVIDIA's [programmatic dependent launch documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) requires synchronization before consuming producer results; the experiments retain those waits. The current [CUTLASS Operator API documentation](https://docs.nvidia.com/cutlass/latest/media/docs/operators/overview.html) includes Hopper BF16 GEMMs, epilogue fusion and graph support, motivating continued scrutiny of prefill fusion opportunities. This pass implements and measures its own Triton kernels; it does not claim a new CUTLASS GEMM result. The [batch-one inference study](https://arxiv.org/abs/2605.30571) likewise motivates measuring execution schedules instead of inferring performance from nominal memory bandwidth; its other-model results are not a latency bound for MOSS.

## Remaining work

The selected post-timing trace contains **13,289 kernel events** in a truncated 34-step request with 33 LLM steps and two codec chunks. Prefill falls from **483 to 375 kernels**, removing 108 launches. The 36 fused activations total 0.141 ms and 72 fused residual/norm kernels total 0.204 ms in that trace. Profile durations are descriptive, not paired speedup evidence.

Projection-only resident intervals remain **39.704 ms**, with **9.288 ms projection/attention overlap** and **6.742 ms attention-only**. These include programmatic dependency waits; they are neither utilization nor pure arithmetic percentages. The initial LLM interval still takes about 61.75 ms, leaving the main target unmet. Further changes must reduce the projection path's actual work or traffic without unacceptable speech changes. This pass provides no proof that 50 ms is impossible and no basis to reduce the 32 codebooks.

## Reproduction

From the repository root with the pinned checkpoints, calibrated export and packing plan:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_prefill_pointwise --tag new_screen --rounds 6
/venv/moss-vllm/bin/python -m optimization.benchmark_prefill_pointwise_paired --tag new_run --pairs 20 --profile
/venv/moss-vllm/bin/python -m optimization.benchmark_prefill_pointwise_http --tag new_run
/venv/moss-vllm/bin/python -m optimization.benchmark_split_norm_pdl --tag new_split --layers 36 --rounds 6
/venv/moss-vllm/bin/python -m optimization.benchmark_split_norm_pdl --tag new_caps --layers 36 --rounds 6 --resources
```

Run GPU jobs sequentially. Both validators accept `--tag`; run each with `compute-sanitizer --tool memcheck --error-exitcode 1` and `--tool racecheck --error-exitcode 1` respectively.

The selected local server command adds `--prefill-pointwise` to the preceding preset:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant \
  --scaled-dp4a --norm-projection --audio-head-buckets --short-scales \
  --projection-pdl --attention-pdl --codec-clock --first-audio-graph \
  --bulk-prefetch --prefill-qkv --prefill-pointwise
```

This is a temporary localhost-only command. Persistent deployment uses supervisor under the instance guide. Existing services remain unchanged.

`results/prefill_pointwise_pass_summary_v1.json` indexes the timing, correctness, sanitizer, cloning, HTTP, profile and compiled-kernel evidence. `results/prefill_pointwise_sources.tar.gz` and `results/prefill_pointwise_source_hashes.json` preserve the source snapshot and per-file hashes; checkpoints, audio and traces remain in their existing result folders.
