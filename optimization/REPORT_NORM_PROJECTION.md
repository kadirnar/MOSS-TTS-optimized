# Normalization/consumer fusion, retaining all 32 codebooks

On this H200 NVL, the selected calibrated path improves **87.25 → 85.48 ms median TTFA** in ten alternating, same-process cloned-voice pairs. Median paired gain is **1.77 ms**, with all ten pairs faster and every complete float32 PCM stream identical. Sequential temporary HTTP servers measure **92.44 → 88.89 ms median**, p95 **95.95 → 90.83 ms** across 20 requests. The 50 ms objective remains unmet.

These measurements retain the original 32-codebook delay schedule, BF16 prefill, calibrated INT4/G32 projections and FP32 codec. The first playable chunk contains 1,920 samples, or 80 ms at 24 kHz. Cached-voice TTFA includes text processing, generation and codec decoding; HTTP measurements additionally include loopback transit. Fresh voice registration is measured separately. Neither original supervisor service has been replaced.

## Kernel change

`dp4a_norm_projection.py` fuses residual addition, RMSNorm, BF16 rounding and G32 activation quantization directly into QKV or paired gate/up/SiLU/output-quantization projections. The small 4,096-element normalization is repeated by each consumer CTA, removing an intermediate activation buffer and a separate producer launch. Only CTA zero writes the shared residual output. Explicit Gluon layouts preserve the selected normalization reduction order; inline PTX performs scaled signed DP4A with register activation operands.

Twenty row/layout configurations were tested on nine frozen real inputs per projection. All 180 configuration/input comparisons match, including residuals, normalized BF16 activations, INT8 values, FP32 scales and projection outputs. Selected QKV R8/IG1/IR1 reduces norm-plus-projection time **8.657 → 8.213 µs**. Selected paired gate/up R32/IG2/IR4 reduces **20.660 → 19.639 µs**. Both use four warps. QKV uses 56 registers and 2,048 shared bytes; gate/up uses 162 registers and 1,024 shared bytes. Neither spills. Timings disable diagnostic intermediate stores.

Evidence: `norm_projection_v1.json`, `norm_projection_{qkv,up}_memcheck_v1.{ptx,ttgir}`. The first full-model warmup exposed a missing local `project` import in the precomputed-QKV branch. The import was corrected; `norm_projection_paired_v1.log` is retained as a failed integration run and supplies no timings.

## Numerical and streaming qualification

- Captured 3,168 raw normalization inputs at all 72 sites across four held-out calibration utterances and multiple decode positions. With zero/spike cases, **3,312** private-stream operator comparisons match exactly.
- Twelve padded/actual-shape private-stream CUDA graph cases pass Compute Sanitizer: **zero memcheck errors and zero racecheck hazards**.
- Sixteen complete-backbone graph comparisons match all text/audio logits across attention-capacity boundaries, the 1,024-position fallback and returns to smaller positions.
- All ten paired control PCM hashes match the previous selected scaled-DP4A controls. The five-request benchmark reproduces its 36-step teacher-forced diagnostics and saved WAV exactly.
- All **16 bilingual generated WAVs and four reference WAVs** match the preceding calibrated path byte for byte. Text, voice, seed, frame count, prompt length and completion metadata also match. Existing diagnostic CER 5.37%, English WER 0% and speaker cosine 0.9143 apply to these identical artifacts; ASR was not rerun. This small, repeatedly examined suite does not establish broad quality acceptance or equality with unquantized BF16.
- Twenty full HTTP PCM stream hashes/frame counts match. No measured request truncates. Invalid input, cancellation, busy response and recovery checks pass; both temporary servers are stopped.

Artifacts: `norm_projection_validation_v1.json`, `norm_projection_memory_{memcheck,racecheck}_v1.json`, sanitizer logs, `norm_projection_graph_validation_v1.json`, `norm_projection_prior_control_equivalence.json`, `norm_projection_full_equivalence.json`, and `quality_suite/norm_projection_audio_equivalence.json`.

The default BF16 path was checked separately in both installed runtimes. Torch 2.13 measures 166.54 ms and matches its previous Torch 2.13 diagnostics/WAV; Torch 2.9 measures 171.06 ms and matches its previous Torch 2.9 diagnostics/WAV. An initial cross-runtime comparison correctly failed; only the matched-runtime comparisons establish this regression result. See `norm_projection_default_regression_comparison.json`.

## Latency evidence and remaining bottleneck

| Comparison | Control median | Fused median | Candidate p95 |
|---|---:|---:|---:|
| Ten alternating in-process pairs | 87.25 ms | 85.48 ms | 87.63 ms |
| Separate five-request benchmark | — | 85.61 ms | 86.06 ms |
| Twenty sequential HTTP requests | 92.44 ms | 88.89 ms | 90.83 ms |
| Fresh registration plus synthesis, one observation each | 156.25 ms | 127.75 ms | — |

The paired benchmark reverses order each pair and uses identical references, text and seeds. Its median paired gain is 1.7657 ms. One control outlier gives a 5.57 ms paired difference; cause is unknown and that difference is not attributed to the fusion. The first pairs were also slower for both paths. All raw measurements remain in `norm_projection_paired_v2.json`.

HTTP servers run sequentially, control then fused. Median engine TTFA is 87.59 → 85.17 ms; initial decode-step median is 2.234 → 2.170 ms. Prefill is 9.216 → 9.211 ms and first codec decode 4.773 → 4.764 ms. Preparation is 1.214 → 1.086 ms; seeding is 0.240 → 0.226 ms. The entire 3.55 ms HTTP difference cannot be assigned to the kernel. Candidate HTTP includes a 105.84 ms outlier. Fresh-registration measurements are single observations, not a latency distribution; registration itself takes 35.92/35.40 ms. See `http_norm_projection_comparison_v1.json` and `http_norm_projection_stages_comparison_v1.json`.

The updated profile removes **2,376 standalone normalization launches**. Fused consumers consume 32.626 ms and remaining projections 17.437 ms, totaling **50.063 ms / 58.98%** of profiled GPU time. Previously, projections plus normalization consumed 51.537 ms. Native attention consumes 4.643 ms / 5.47%. Profiling overhead is excluded from TTFA measurements. Projection arithmetic and weight traffic remain the main targets; unchanged prefill and codec also remain material.

## Reproduction

Use `/venv/moss-vllm` (Torch 2.13, Triton 3.7.1) and run GPU jobs sequentially with fresh tags. Source checkpoint revisions are pinned in `common.py`; calibrated exports are separate. For the operator and complete paired tests:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_norm_projection --tag reproduce
/venv/moss-vllm/bin/python -m optimization.validate_norm_projection validate --tag reproduce
compute-sanitizer --tool memcheck --error-exitcode 94 \
  /venv/moss-vllm/bin/python -m optimization.validate_norm_projection_memory --tag reproduce_memcheck
compute-sanitizer --tool racecheck --error-exitcode 94 \
  /venv/moss-vllm/bin/python -m optimization.validate_norm_projection_memory --tag reproduce_racecheck
/venv/moss-vllm/bin/python -m optimization.validate_norm_projection_graphs --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_norm_projection_paired --tag reproduce --pairs 10
/venv/moss-vllm/bin/python -m optimization.benchmark_norm_projection_http --tag reproduce
```

The opt-in flag is `--norm-projection`, added after the preceding selected flags:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_quantization \
  --mode none --calibration optimization/results/gptq_v1_g32_d10 \
  --calibrated-backend dp4a --tag norm_projection_reproduce \
  --attention-block 32 --fused-residual --dp4a-reciprocal \
  --grouped-activation --fused-dp4a --dp4a-gateup --norm-layout-limit 8 \
  --prefill-buckets 128 160 256 512 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant \
  --scaled-dp4a --norm-projection
```

`quality_generate` accepts the same model flags. A foreground development endpoint uses:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant \
  --scaled-dp4a --norm-projection
```

This binds to loopback; persistent services should use supervisor. Register voices through `/v1/voices`, then stream `/v1/audio/speech`. Fusion requires the selected scaled-DP4A preset and must be enabled before graph capture. Existing service defaults and their process-local voice caches remain unchanged.

## Research context

The [Gluon layout documentation](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html) informs explicit lane/warp layout control. [Principled Coarse-Grained Acceptance for Speculative Decoding in Speech](https://machinelearning.apple.com/research/coarse-grained), [Approximate Speculative Decoding](https://arxiv.org/abs/2608.03447), and [Carryover Drafting](https://arxiv.org/abs/2609.14717) suggest separate draft/verification experiments, but are not implemented here. Their reported results on other workloads are not MOSS measurements, and approximate acceptance would need new quality qualification. None establishes an absolute 50 ms limit for this checkpoint.

`norm_projection_source_hashes.json` and `norm_projection_sources.tar.gz` preserve this pass's code/docs. Checkpoints and measured artifacts are excluded from that source archive and remain separately under `results/`.
