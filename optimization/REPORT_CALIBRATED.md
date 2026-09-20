# Calibrated INT4 and integer-kernel optimization

This pass preserves **all 32 codebooks**, voice cloning, the original model revisions, and the FP32 causal codec. The 50 ms target remains unmet. None of these experiments has replaced either running service. Measurements below are warm, batch-one, in-process requests with a cached encoded 3.112-second reference, measured from complete text to the first complete 80 ms CPU PCM chunk. They include processing, prefill, delayed autoregressive generation and codec decoding; they exclude HTTP, network transit and fresh reference encoding.

## Calibration and correctness

`collect_calibration.py` generated 12 complete calibration utterances from the BF16 teacher, using four supplied voice assets and three new texts per language. The texts are disjoint from the 16-utterance evaluation suite; the voice assets are shared. Only actual generated-input rows were collected through a full causal teacher replay. Eight utterances / 866 rows train the quantizer; four utterances / 410 rows form the held-out calibration split. This small calibration set is not an independent large-domain quality evaluation.

The exporter applies GPTQ block error feedback with activation ordering, static 32-element groups, signed codes in [-7, 7], and BF16 scales. It keeps the source checkpoint untouched. The implementation follows the [authors' GPTQ algorithm](https://github.com/IST-DASLab/gptq/blob/main/gptq.py), pinned to `2d65066eeb06a5c9ff5184d8cebdf33662c67faf`; the Apache license and adaptation notice are retained. A fused Triton column quantizer replaces the scalar Python/Torch update loop. BF16 dequantization rounding is represented in the error feedback.

- All six correlated-input checks matched the straightforward Torch algorithm's integer codes and scales exactly (`gptq_algorithm_validation.json`).
- A 12-projection pilot compared damping 0.01 and 0.1. The latter reduced the worst late attention-output regression and was selected globally, before evaluation. It was not selected separately against evaluation utterances.
- The full 144-projection export improved held-out output relative MSE in 139 projections versus the same-scale round-to-nearest control. The median GPTQ/RTN error ratio is 0.442. Five late attention-output projections regressed; no selective fallback hides those results. Export configuration, per-projection statistics and SHA-256 hashes are in `results/gptq_v1_g32_d10/`.
- All 12 real-weight packing checks passed for both native Marlin and custom DP4A on a non-default CUDA stream (`gptq_backend_validation.json`). Signed-nibble round trips were exact. These are operator checks, not quality approval.

## Complete streaming measurements

Every row below measured five complete requests after one warmup; no measured request hit the 400-token limit. The attention tile is 32, embeddings and output heads remain BF16, and prefill uses the original BF16 weights. CUDA graphs are captured before timing.

| Calibrated G32 variant | Median TTFA | p95 | Median decode |
|---|---:|---:|---:|
| Native Marlin W4A16 | 130.28 ms | 132.79 ms | 3.497 ms |
| DP4A, per-vector reciprocal activation quantization | 113.60 ms | 114.97 ms | 2.976 ms |
| Above, fused normalization/quantization | 110.57 ms | 110.73 ms | 2.884 ms |
| DP4A, per-group activation quantization | 111.56 ms | 113.01 ms | 2.907 ms |
| Above, fused normalization/quantization | 109.12 ms | 113.62 ms | 2.834 ms |
| Above, paired gate/up + SiLU | 106.21 ms | 106.61 ms | 2.750 ms |
| Above, add a 160-token prefill bucket | 104.65 ms | 105.22 ms | 2.756 ms |
| Above, constrain normalization to original reduction layout | 106.43 ms | 106.69 ms | 2.770 ms |

Raw samples and profiles are `results/all32_gptq_*.json` and the corresponding `_profile.txt` files. These are single-prompt latency diagnostics, not statistically established production percentiles. Adding the bucket pads the 145-token prompt to 160 instead of 256; prefill fell from 11.07 to 9.22 ms. No voice prefix, input text, or generated audio is cached in this experiment. Smaller prefill buckets change floating-point ordering, so they need their own quality evaluation. The layout-constrained run was slower in the complete-request benchmark despite a slightly faster isolated normalization operator; retain the measured result rather than assuming microbenchmarks compose.

The original round-to-nearest Marlin G32 trial had 0.539 active top-1 agreement against upstream logits; calibrated Marlin reached 0.639. Per-vector DP4A reached 0.525; per-group activation scaling recovered 0.647, with KL 0.0280. These are 36 teacher-forced steps on one cloned-voice prompt, not equivalence guarantees.

## Kernel findings

- The per-vector DP4A profile assigned 51.3% of GPU time to integer-dot projections and 7.8% to activation quantization. Per-group scaling improves accuracy by avoiding a single extreme activation setting the precision of the whole vector. All 32 group/shape/warp checks passed against an independent NumPy quantization reference and FP32 dequantized linear algebra.
- Fusing residual addition, RMSNorm and activation quantization passed 12 synthetic producer checks for each of global and G32 scaling, including exact BF16 outputs, integer codes and FP32 scales on a non-default stream. G32 residual-normalization plus quantization fell from 3.48 to 2.69 microseconds. The implementation retains the BF16 residual, normalized activation and weighted activation rounding boundaries.
- The separate one-block SiLU/quantization fusion was correct but slower (G32: 5.45 versus 3.06 microseconds). It is disabled.
- Paired DP4A gate/up projection with the BF16 SiLU epilogue passed 12 real-projection/warp checks. The selected one-warp version matched all three checked reference outputs exactly and reduced the cold-weight operator from about 24.3 to 21.6 microseconds, including grouped input quantization. The saved complete-request WAV also matched the preceding grouped/fused variant.
- Normalization fusion matched the recorded 36-step diagnostics but changed the saved free-generation WAV relative to the unfused variant. A subsequent audit evaluated fused and unfused producers on identical real activations inside each graph replay, over two complete sampled utterances. Residual additions matched exactly, but there were 223 differing BF16 normalized elements, 76 differing INT8 codes and four differing scale values across eight normalization sites; maximum absolute BF16 difference was 0.015625. This is a numerically changed experiment, not an exact-equivalence optimization. Evidence: `dp4a_real_fusion_validation.json`. Instrumented audit timings are not used as latency measurements.
- Compiler IR showed the fused normalization using 16 elements per thread, versus eight in the original reduction. An explicit contiguity limit of eight restored the original reduction layout. The repeated real-activation audit then matched **all residuals, normalized BF16 values, INT8 codes and FP32 scales exactly**, with zero maximum difference (`dp4a_real_fusion_validation_layout8.json`). This layout-constrained variant is measured and evaluated separately from the original fusion; it does not make INT4 quantization equivalent to BF16.
- The layout constraint reduced the kernel's shared-memory allocation from 4096 to 512 bytes (`dp4a_norm_compiler_layouts.json`). Its isolated residual-normalization/quantization time was 2.55 microseconds versus 3.43 for separate producers. All 12 synthetic component checks also passed. The full-model result above includes the actual resulting memory layout and CPU overhead.

After gate/up fusion, the remaining projections plus the paired gate/up kernel still consume 53.2% of profiled GPU time. Attention and its reduction consume about 8.9%, fused normalization/quantization 6.4%, and the remaining activation quantizers 3.2%. More parallel work within these dependent backbone steps remains the central latency problem; these measurements do not prove that 50 ms is impossible.

## Voice-cloning quality

The same 16 texts, four supplied references and seeds were evaluated with Whisper-small CUDA FP16 and WavLM FP32 speaker vectors. Chinese and English text normalization matches the previous reports. All rows below have finite audio, no truncation, and the intended voice as the top-scoring reference in all 16 cases.

| Configuration | Chinese CER | English WER | Mean speaker cosine |
|---|---:|---:|---:|
| Previous custom BF16 | 4.46% | 0.78% | 0.9104 |
| Previous experimental FP8 | 7.59% | 0% | 0.9090 |
| Calibrated Marlin | 5.51% | 0% | 0.9175 |
| Calibrated DP4A, per-vector activation scale | 15.46% | 0% | 0.9157 |
| Calibrated DP4A, per-group activation scale | 8.05% | 0% | 0.9132 |
| Grouped DP4A, original norm/gate-up fusions, bucket 160 | 3.80% | 0% | 0.9070 |
| Above, layout-constrained normalization | 5.37% | 0% | 0.9143 |

Evidence: `quality_suite/evaluation_gptq_normalized.json`, `evaluation_gptq_grouped_normalized.json`, `evaluation_gptq_fused_bucket160_normalized.json`, and `evaluation_gptq_layout8_bucket160_normalized.json`, with separate raw transcripts. Both fused/bucket variants completed all 16 evaluation utterances. These are small, uncalibrated diagnostics, not human listening or a production quality acceptance. Repeated exploration on this fixed suite further limits interpreting small score differences as generalization improvements.

After the runtime edits, the default BF16/FP32 path completed five requests at 171.23 ms median in-process TTFA. Its full recorded teacher-forced validation matches the preceding regression exactly: relative RMS 0.0077181, active top-1 0.9218444, KL 0.0031524 (`all32_none_post_calibrated_regression_fr_fg.json`). Existing services remained healthy, with all 32 codebooks and their previous voice caches.

## Reproduction

Run GPU commands sequentially, with no concurrent inference traffic. Calibration collection/export uses `/venv/main`; native-kernel and complete-model trials use `/venv/moss-vllm` (Torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.29). The original BF16 service remains in `/venv/main`.

```bash
/venv/main/bin/python -m optimization.collect_calibration
/venv/main/bin/python -m optimization.validate_gptq
/venv/main/bin/python -m optimization.calibrate_gptq --tag gptq_v1_g32_d10 --group 32 --damping 0.1
/venv/moss-vllm/bin/python -m optimization.validate_calibrated_backend --calibration optimization/results/gptq_v1_g32_d10
/venv/moss-vllm/bin/python -m optimization.benchmark_int4_dp4a --reciprocal --grouped-activation
/venv/moss-vllm/bin/python -m optimization.validate_dp4a_fusions --group 32
/venv/moss-vllm/bin/python -m optimization.benchmark_dp4a_gateup
/venv/moss-vllm/bin/python -m optimization.benchmark_quantization \
  --mode none --calibration optimization/results/gptq_v1_g32_d10 \
  --calibrated-backend dp4a --tag calibrated_g32_d10 \
  --attention-block 32 --fused-residual --dp4a-reciprocal \
  --grouped-activation --fused-dp4a --dp4a-gateup \
  --norm-layout-limit 8 \
  --prefill-buckets 128 160 256 512
```

Omit `--norm-layout-limit 8` to reproduce the numerically changed 104.65 ms variant. The layout-corrected variant uses it explicitly.

Export installation requires a complete matching checkpoint revision and must occur before CUDA graph capture. All new quantization and fusion switches are opt-in. Default model precision, service configuration and codebook count are unchanged.

## Opt-in streaming server

The existing voice-registration and PCM-streaming API now accepts an explicit calibrated preset. This selects grouped DP4A, layout limit eight, paired gate/up, BF16 prefill buckets 128/160/256/512 and the FP32 codec. It keeps all 32 codebooks. It does not reconfigure an already running process.

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10
```

Use `/v1/voices` to register a reference and `/v1/audio/speech` to stream 24 kHz mono signed-16-bit PCM, as in `README.md`. The server binds only to loopback. The command above is a foreground development invocation; a lasting instance service belongs under supervisor. A temporary local instance was used for the HTTP checks and is stopped after validation.

The layout-constrained preset measured **101.53 ms median / 103.94 ms p95 loopback HTTP TTFA** over 20 warm requests, after one warmup. Requests registered and reused a reference voice, generated 54–77 complete PCM frames, and used a 400-token budget. Invalid voice, invalid token budget and prompt overflow checks passed; cancellation recovered through HTTP 429 followed by 200. The health response confirmed all 32 codebooks, FP32 codec and the explicit calibrated preset. Evidence: `http_calibrated_layout8.json` and `http_calibrated_layout8_health.json`.

Correction from the subsequent packed-kernel audit: the HTTP text is identical to the in-process fixture and produces 145 prompt tokens. The previously stated shorter-text explanation was incorrect. The historical 101.53 ms result is retained as measured; it cannot be subtracted from an independently measured in-process result to estimate HTTP overhead. A fresh controlled comparison with stage instrumentation measured this preset at 107.58 ms HTTP / 104.01 ms inside the engine; see `REPORT_PACKED.md`. The earlier FP8 endpoint's 122.17 ms remains a separate historical measurement.

A separate, contiguous fresh-reference path took **140.44 ms** from starting HTTP voice registration to receiving the first PCM chunk from the immediately following synthesis request. Registration itself took 36.04 ms, including 22.46 ms reference encoding. This is one observation, with WAV/base64 prepared before timing, not a fresh-reference latency distribution. The initial benchmark attempt discarded the streaming iterator early and correctly received HTTP 429 on the next request; its failed log is retained as `http_calibrated_layout8.invalid_fresh_drain.log`. The benchmark now drains the same iterator, and all reported HTTP measurements come from the subsequent successful run.

The temporary endpoint was shut down after testing. The default BF16 and existing FP8 supervisor services were neither restarted nor replaced. The new preset is reproducible through the explicit CLI above; the 50 ms target remains open.
