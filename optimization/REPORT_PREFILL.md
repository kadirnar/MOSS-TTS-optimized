# Prefix, prefill and integer-dot investigations

This pass continues the 50 ms target with **all 32 codebooks** and voice cloning. The target remains unmet. The default BF16 and experimental FP8 services described in `REPORT_REVIEW.md` remain on their previously measured configurations; the new prefix and W8A8 prefill options below are not deployed.

## Measured complete-request changes

All measurements are warm batch-one in-process requests, with the encoded voice reference cached, complete text received at the start, and time measured to the first complete 80 ms CPU PCM chunk. Five complete utterances are measured after one warmup. No generated audio or input text is cached.

| Configuration | Median TTFA | Median prefill |
|---|---:|---:|
| FP8 decode, BF16 full prefill, same-run baseline | 118.21 ms | 11.04 ms |
| Reuse first 96 conditioning tokens | 115.45 ms | 8.18 ms |
| Reuse first 120 conditioning tokens | 115.42 ms | 8.00 ms |
| FP8 decode, W8A8 full prefill | 116.90 ms | 9.67 ms |
| W8A8 prefill plus 120-token conditioning cache | **114.11 ms** | **6.90 ms** |

The last configuration's TTFA p95 is 114.40 ms. Evidence is in `prefix_cache_fp8_all.json`, `prefix_cache_fp8_all_p120.json`, `all32_fp8_all_prefill_b32w4_fr_fg_pf8.json`, and `prefix_cache_fp8_all_p120_pf8.json`. These numbers do not include HTTP handling or fresh waveform encoding. They are not a new measurement of the running FP8 HTTP endpoint.

`PrefixPrefillCache` verifies the processor's text marker before permitting reuse. The key contains all 33 channels of the conditioning prefix, including voice codes. New text is always processed. An LRU bounds stored prefixes; missing/short/unsupported prompts use full prefill. Captured suffix graphs restore prefix KV and use lower-right causal attention. Tests varied text, language, reference codes and prompt length, included a no-reference bypass, and cleared request-local KV before reuse on a non-default stream.

Splitting prefill changes GEMM/attention floating-point ordering. Against the same full-prefill custom model, the 120-token BF16-prefix experiment has 0.00355 maximum relative RMS logit error and 0.903 active top-1 agreement over 36 forced decode steps; the W8A8-prefix variant has 0.00741 and 0.868. These are diagnostics, not audio-quality approval.

## Quality decision

The combined prefix/W8A8 configuration generated 16 complete, finite bilingual utterances across four supplied voice references. Each voice prefix was primed with placeholder text before the real text was supplied. All actual generation requests hit the prefix cache; none cached generated audio.

| Configuration | Normalized Chinese CER | English WER | Mean speaker cosine |
|---|---:|---:|---:|
| Previous custom BF16 | 4.46% | 0.78% | 0.9104 |
| Previously measured experimental FP8 | 7.59% | 0% | 0.9090 |
| New prefix + W8A8 prefill | 8.51% | 0.78% | 0.9106 |

The intended voice was the top-scoring reference for all 16 new utterances, with no truncation. The small ASR diagnostic regressed, so the 114 ms configuration stays experimental and is not promoted to either service. Evidence: `quality_suite/evaluation_fp8_prefix120_normalized.json`; raw transcripts and metrics are retained separately. These means are not statistically established corpus-level quality differences or human listening scores.

## Kernel evidence

- Native FP6-LLM source `12e83379f16a4ee1494be00db6956aab56baf620` was built for SM90 in `/venv/main`. All 24 projection/split-K checks passed, including non-default-stream execution. Best gate-up/down/QKV/output timings were 37.14 / 27.73 / 19.59 / 17.87 microseconds, including BF16↔FP16 conversions. It was slower than FP8 and not integrated into the model.
- W8A8 prefill tested 20 shape/token-count combinations with both native PyTorch and vLLM CUTLASS operators. All 40 checks passed against FP32 computation on the same quantized inputs/weights. vLLM was generally faster for short prefill; single-token decode keeps custom Triton GEMV.
- Nsight Compute is installed, but the host denies hardware performance-counter access (`ERR_NVGPUCTRPERM`). This does not prevent CUDA-event timings or compiler inspection. The main fused FP8 gate/up kernel uses 38 registers per thread, 16 bytes of shared memory and no spills. Counter metrics are unavailable; the profiler failure is recorded rather than treated as a measurement.
- The first grouped INT4 kernel used scalar strided activation loads. Generated PTX showed 32 separate 16-bit loads in the four-warp variant. Reading two BF16 activations together as a 32-bit word reduced G32 gate-up from 90.70 to 37.71 microseconds, and down from 115.93 to 20.71 microseconds. Forty checks passed for each variant. They remain slower than the best FP8 combination.
- A separate INT4/INT8 activation DP4A trial uses four integer multiply-adds per instruction and FP32 group scaling. Its initial reciprocal-based activation quantizer failed one numerical test; those partial artifacts are marked `.invalid_quantizer.*`. The explicit correctly rounded division version passed all 32 kernel checks. Its G32 complete-model median was **127.51 ms**, with 3.388 ms decode, active teacher-forced top-1 agreement **0.425**, KL **0.1407**, and one of five measured requests hitting the 400-token limit. It is not deployable on this evidence. These are simple round-to-nearest weights, not GPTQ-calibrated weights.
- A subsequent quantizer explicitly defines rounding after multiplication by a rounded FP32 reciprocal, and uses an independent NumPy reference. All 32 projection checks pass. Its G32 gate-up/down/QKV/output timings are 22.95 / 15.42 / 9.85 / 7.99 microseconds, including activation quantization. This avoids per-element correctly rounded division, but has not yet been evaluated in the complete model. It is an explicit numerical variant, not a silent replacement of the failed quantizer or deployed model.

After the code changes, the default BF16 configuration passed a complete-model regression: all five requests finished and the teacher-forced relative RMS, active top-1 and KL metrics exactly match the prior recorded values (0.0077181, 0.9218444, 0.0031524). Evidence: `all32_none_post_prefill_regression_fr_fg.json`. Neither running service was restarted or reconfigured in this pass.

The next substantial quantization question is whether calibrated weights can preserve cloning quality with these faster integer kernels. The measured round-to-nearest failures do not answer that question, and the untested reciprocal full model must not be presented as a successful deployment or a measured TTFA result.

## Reproduction

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_prefix_cache --prefix-length 120
/venv/moss-vllm/bin/python -m optimization.benchmark_fp8_prefill
/venv/moss-vllm/bin/python -m optimization.benchmark_prefix_cache --prefix-length 120 --fp8-prefill
/venv/main/bin/python -m optimization.benchmark_fp6
/venv/moss-vllm/bin/python -m optimization.benchmark_grouped_int4 --vector-x
/venv/moss-vllm/bin/python -m optimization.benchmark_int4_dp4a
```

Run GPU benchmarks sequentially. Optional `fp8_prefill=True` requires `weight_quantization='fp8_all'` and the isolated vLLM environment. The default remains BF16 prefill. Prefix caching is opt-in through `PrefixPrefillCache`; warm its graphs before installing it on an exclusively owned `FastLLM`.

Primary implementation references: [PyTorch lower-right causal attention](https://docs.pytorch.org/docs/main/generated/torch.nn.attention.bias.causal_lower_right.html), [FP6-LLM authors' CUDA implementation](https://github.com/usyd-fsalab/fp6_llm), and the [MOSS delay architecture](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/README.md). These sources motivate experiments; latency and quality claims above come from this instance's saved measurements.
