# Fixed-order native CUDA attention

All 32 codebooks, streaming voice cloning, BF16 prefill and the FP32 codec remain enabled. The 50 ms target remains unmet. This pass follows `REPORT_EPILOGUE.md`; existing supervisor services have not been replaced.

## Isolating the compiler change

`compiler_audit.py` captures actual operator inputs and outputs from the selected calibrated path: the 145-token prefill, three layers (0/17/35), and teacher-forced decode steps 0/10/31. Instrumented capture is not timed. Its 83 records produce 203 tensor comparisons, covering 8,797,568 elements. Replaying under the same Torch 2.13 / Triton 3.7.1 environment reproduced every element, including compacted embedding tables. Each replay uses a private CUDA stream.

Changing only the experimental Triton environment to 3.8 produced these differences on the identical saved inputs:

| Operator family | Mismatched elements | Maximum absolute difference |
|---|---:|---:|
| Embedding, prefill RMSNorm, fused norm/quant, QK/RoPE, final residual/norm, attention reduction/quant | 0 | 0 |
| Projections | 23 / 239,616 | 0.00048828125 |
| Attention partials | 75,370 / 1,179,648 | 0.00004435 |
| Attention log-sum-exp | 87 / 9,216 | 0.00009155 |

These are sampled operator comparisons, not an exhaustive equivalence proof. The attention PTX shows the old sequential register accumulation becoming a balanced reduction tree in the new compiler, with different FMA placement. The register layout itself is unchanged. This explains a concrete source of floating-point differences; it does not individually assign the previous corpus regression to attention versus projection changes.

The [Triton 3.8 release notes](https://github.com/triton-lang/triton/releases/tag/v3.8.0) document compiler and SM90 reduction changes. The September 2026 [bitwise GPU-kernel study](https://arxiv.org/abs/2609.11356) discusses how reduction order, FMA and rounding affect reproducibility. Those sources informed the investigation; all performance and speech measurements here are from this checkpoint on this H200.

Evidence: `compiler_capture_v1/manifest.json`, `compiler_audit_triton371.json`, `compiler_audit_triton38.json`, and both compiler attention PTX/TTGIR files.

## Gluon layout trials

`attention_layout.py` removes optional Q-load and partial-output layout conversions using explicit Gluon layouts. Sixteen combinations cover four warp counts and both conversion choices. The 528 frozen/synthetic cases compare attention partials and log-sum-exp across capacities 128/256/512/1024 and cache boundaries.

Only the four-warp variants match every comparison. Their cold-cache operator timings do not improve on the preceding kernel; they remain experiments. Removing shared-memory conversions alone does not guarantee lower latency. Evidence: `attention_layout_triton371.json`, `.ptx` and `.ttgir`.

## Native CUDA implementation

`attention_native.cu` retains the selected B32/W4 arithmetic explicitly using rounded CUDA operations and inline PTX for exponential and division. Separate shared buffers for maximum, denominator and weighted values remove shared-buffer reuse barriers. The native path has three static `bar.sync` instructions, versus twelve in the preceding Triton PTX. It uses 40 registers per thread, 2,080 shared bytes and no spills.

The first version used scalar loads. A second version loads eight BF16 values with a 16-byte vector load; it preserves the same arithmetic. Both versions pass all 33 frozen/boundary comparisons exactly, including FP32 partials/log-sum-exp, final BF16 attention output, INT8 activations and FP32 quantization scales. Source-addressed build artifacts retain both CUDA versions and their compilation logs.

| Context capacity | Triton warm operator | Scalar CUDA warm | Vector CUDA warm |
|---|---:|---:|---:|
| 128 | 3.67 µs | 2.90 µs | 2.70 µs |
| 256 | 4.39 µs | 3.42 µs | 2.88 µs |
| 512 | 6.03 µs | 5.00 µs | 3.30 µs |
| 1024 | 9.62 µs | 8.19 µs | 4.54 µs |

Each synthetic timing uses the final position of the stated capacity. Cold-cache results are also retained: at capacity 256, the vector kernel takes 8.19 µs versus 9.89 µs for Triton. Operator timings are not full-request TTFA savings; the real prompt activates only part of its capacity bucket.

The native library is compiled with CUDA 12.8 for SM90 and called on the caller's current stream. The opt-in installer validates the architecture and B32/W4 attention configuration, builds outside graph capture, and the launcher validates tensor metadata. No host driver change is involved.

Compute Sanitizer memcheck and racecheck each passed 16 private-stream CUDA-graph cases, with zero errors and zero hazards. Future KV slots contain NaNs to detect accidental masked reads. All partial outputs remain finite; unused splits are zero with negative-infinity log-sum-exp. Separate full-backbone graph validation checks context boundaries, position 1023 and returns to smaller positions with an initialized long prefix.

Evidence: `attention_native_v1.json`, `attention_native_vector_v2.json`, `attention_native_build/`, `attention_native_memcheck.log`, `attention_native_racecheck.log`, and `attention_native_graph_validation*.json`.

## Full-model and voice-cloning checks

The scalar CUDA version measured **91.02 ms median / 91.25 ms p95** in-process TTFA over five complete requests. The vector version's independent run measured **92.51 / 93.19 ms**; its request preparation and decode timings were higher despite a slightly lower profiled attention cost. Both observations are retained. The vector operator's isolated gain does not establish an additional full-request improvement over scalar CUDA.

To resolve the variation, `benchmark_attention_paired.py` alternates separately captured Triton and vector-CUDA graphs inside **one process and one model**, reversing execution order every pair. Each pair uses identical text, reference and seed; one warmup pair is excluded. Over ten measured pairs, control TTFA is **92.23 ms median / 92.62 ms p95** and native TTFA is **90.55 / 90.72 ms**. The median of the ten paired differences is **1.71 ms**, and every pair improves (range 1.39–2.15 ms). All ten complete float32 PCM streams match exactly. Preparation (1.15/1.16 ms) and prefill (9.21/9.21 ms) are closely matched. This supports the native-vector gain over Triton under the stated cached-voice workload; it does not compare vector against scalar CUDA or include HTTP/fresh-reference encoding. Evidence: `attention_paired_vector_v2.json`.

Both versions exactly reproduce the preceding calibrated path's 36-step teacher-forced diagnostics and saved benchmark WAV. Each also reproduces all **16 bilingual generated WAVs and four reference WAVs** byte for byte, with matching prompt, seed, frame and termination metadata. Existing diagnostic scores apply to these identical artifacts: Chinese CER 5.37%, English WER 0%, speaker cosine 0.9143. No new ASR evaluation is claimed, and this is not equality with the original unquantized BF16 model or broad quality acceptance.

The vector profile attributes 4.627 ms to native attention across 1,188 calls (3.895 µs each), versus the preceding Triton attention profile's 6.325 ms. Projection kernels remain the largest cost: 45.957 ms, about 51.7% of the profiled GPU time. Fused normalization/quantization takes another 7.4%. Further work should target those costs, with the same exact-output and speech checks.

Evidence: `all32_gptq_dp4a_native_attention_v1*.json`, `all32_gptq_dp4a_native_attention_vector_v2*.json`, corresponding profiles, and `quality_suite/native_attention*_audio_equivalence.json`.

## Streaming HTTP comparison

Fresh sequential temporary servers used the same current runtime, source, voice reference, text and seeds 501–520. Both enabled the preceding attention-output quantizer fusion; only the native partial-attention implementation differed. Each configuration completed 20 measured requests after warmup.

| Configuration | HTTP median / p95 | Engine median | Initial decode step |
|---|---:|---:|---:|
| Triton control | 95.84 / 96.85 ms | 92.18 ms | 2.391 ms |
| Native vector CUDA | 95.54 / 97.55 ms | 91.02 ms | 2.336 ms |

The HTTP median change is only **0.30 ms**, and native HTTP p95 is **0.70 ms higher**. The measured engine median improves by 1.16 ms. Preparation differs (1.09 versus 1.44 ms); prefill and the first codec frame remain approximately 9.21 and 4.78 ms. These observations support a kernel/engine gain, not a robust HTTP-tail improvement or a latency guarantee.

All **20 complete PCM stream hashes match**, with matching frame counts and no truncation. Input-error and cancellation/recovery checks passed; both recover through 429 then 200. Single contiguous fresh-reference registration-to-PCM observations are 135.09 ms native and 136.09 ms control, including registration HTTP costs of 37.29 and 35.61 ms. Single observations do not define a fresh-reference distribution.

Both temporary servers were stopped after testing. The existing BF16 and FP8 supervisor services retain their original processes. Evidence: `http_native_attention.json`, `http_native_attention_control.json`, `http_native_attention_comparison.json`, `http_native_attention_health.json`, and both stage-metrics JSONL files.

## Reproduction

Run GPU workloads sequentially using the existing Torch 2.13 / Triton 3.7.1 environment:

```bash
/venv/moss-vllm/bin/python -m optimization.compiler_audit replay --tag triton371
/venv/moss-triton38/bin/python -m optimization.compiler_audit replay --tag triton38
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_layout --tag triton371
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_native --tag vector_v2
/venv/moss-vllm/bin/python -m optimization.validate_attention_quant_graphs \
  --native-attention --tag vector_v2
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_paired \
  --tag vector_v2 --pairs 10
compute-sanitizer --tool memcheck --error-exitcode 91 \
  /venv/moss-vllm/bin/python -m optimization.validate_attention_native_memory
compute-sanitizer --tool racecheck --error-exitcode 92 \
  /venv/moss-vllm/bin/python -m optimization.validate_attention_native_memory
/venv/moss-vllm/bin/python -m optimization.benchmark_quantization \
  --mode none --calibration optimization/results/gptq_v1_g32_d10 \
  --calibrated-backend dp4a --tag native_attention_vector_v2 \
  --attention-block 32 --fused-residual --dp4a-reciprocal \
  --grouped-activation --fused-dp4a --dp4a-gateup --norm-layout-limit 8 \
  --prefill-buckets 128 160 256 512 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention
```

Use a new result tag to preserve historical measurements. To recreate frozen inputs, call `compiler_audit capture --folder <new-folder>` under the old compiler. `quality_generate` accepts the same model flags. The optional local development server is:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention
```

Register a reference through `/v1/voices`, then stream `/v1/audio/speech` as documented in the README. The first PCM chunk is still 80 ms decoded from all 32 codebooks. This foreground development command binds only to loopback; a persistent service should use supervisor.

`native_attention_source_hashes.json` and `native_attention_sources.tar.gz` preserve this pass's source snapshot. Older source-addressed native build artifacts retain the scalar and vector implementations.
