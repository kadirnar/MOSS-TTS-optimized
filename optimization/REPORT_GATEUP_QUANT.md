# Gate/up and grouped activation-quantizer fusion

All 32 codebooks remain enabled, including the first playable PCM chunk. Streaming voice cloning, BF16 prefill and the FP32 codec are retained. The 50 ms target remains unmet. This pass builds on `REPORT_NATIVE_ATTENTION.md`; existing supervisor services are unchanged.

## Kernel search and numerical correction

The preceding profile spent about half of GPU time in projections and still launched a separate G32 activation quantizer after gate/up/SiLU in every layer. `dp4a_gateup_quant.py` computes a complete group of 32 output neurons within a CTA, preserving the BF16 gate, up and SiLU roundings before producing grouped INT8 activations and FP32 scales. The downstream projection consumes those tensors directly. These neuron groups are independent of the codec's 32-codebook setting.

Thirty-six initial combinations cover larger row tiles, paired projection plus separate quantization, fused quantization, standard versus direct PTX activation loads, and warp counts. The fastest initial fused configuration takes 17.69 µs versus a 19.30 µs reference, but changes some intermediate values. Its first 18 checks happen to preserve all consumed quantized values. Expanded testing across all 36 layers catches **14 changed INT8 values, one changed scale and 3,606 changed downstream elements** across 504 cases. That unconstrained variant is not selected. The original failing numerical results remain in `gateup_quant_consumer_validation.json`.

Compiler IR shows that the larger row tile changes the scale/reduction layout from four to eight contiguous groups per thread. A second sweep tests 16 scale-contiguity/row/warp combinations. Constraining contiguity to **four** recovers the previous arithmetic while retaining the speed improvement. The selected configuration is direct PTX loads, 32 rows, four warps, and scale-contiguity limit four. It measures **17.77 µs versus 19.32 µs** for gate/up plus quantization on an eight-weight ring, about 8% faster. Activations are prequantized before timing, matching the fused decoder.

The corrected kernel passes all **504 checks**: 36 layers times 12 recorded activations plus zero and spike inputs, using a private stream. BF16 intermediate values, INT8 activations, FP32 scales and final BF16 down-projection outputs all match exactly. It uses 168 registers per thread, 2,048 shared bytes and no spills. The selected PTX/TTGIR and all timings are retained.

Six smaller-row-tile experiments compute 8 or 16 neurons at a time before combining a quantization group. None improves the selected operator in the paired sweep; the existing configuration remains fastest at 17.88 µs versus the 19.25 µs reference in that run. This brings the gate/up search to 59 recorded configurations, including its repeated control. These experiments are informed by the [CUDA hardware-multithreading documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-multithreading) and [Triton's inline-assembly API](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html); hardware-counter attribution remains unavailable on this instance.

Evidence: `gateup_quant_kernels.json`, `gateup_quant_scale_kernels.json`, `gateup_quant_row_kernels.json`, `gateup_quant_consumer_validation_scale4.json`, and `gateup_quant_selected_scale4.{ptx,ttgir}`.

## Full-model validation and latency

Five complete requests measure **89.04 ms median / 89.58 ms p95 TTFA** with a cached cloned voice. The 36-step teacher-forced diagnostics and saved WAV exactly match the preceding native-attention path. Sixteen full-backbone comparisons use separately captured old/fused graphs, an initialized 1,023-token prefix, capacity boundaries, the full fallback and returns to earlier positions. All text/audio logits match on a private stream; both sides use native attention.

All **16 bilingual generated WAVs and four reference WAVs** match the preceding path byte for byte, with matching prompt/seed/frame/termination metadata. Existing small-suite diagnostics apply to the identical artifacts: Chinese CER 5.37%, English WER 0%, speaker cosine 0.9143. No new ASR run or equivalence to the original unquantized BF16 model is claimed.

A same-process comparison alternates the preceding and fused graph sets, reversing order each pair and using the same text, reference and seed. One warmup pair is excluded. Ten measured pairs give:

| Configuration | Median TTFA | p95 TTFA |
|---|---:|---:|
| Native attention, separate gate/up quantizer | 90.45 ms | 90.73 ms |
| Native attention, fused gate/up quantizer | **88.88 ms** | **89.11 ms** |

The median paired difference is **1.51 ms**; all pairs improve by 1.40–1.83 ms. All ten complete float32 PCM streams match. Preparation and prefill are closely matched at about 1.2 and 9.21 ms. This includes text processing, all-32-codebook generation, FP32 decoding and CPU PCM copy; it excludes HTTP, network transit and fresh reference encoding.

Compute Sanitizer memcheck passes eight private-stream graph cases, covering actual dimensions and padded K/output tiles, with zero errors. Each fused output is independently requantized to check its INT8 values and scales.

The unchanged default BF16 path also completes five original-runtime requests at 172.86 ms median / 173.95 ms p95. Its complete 36-step validation dictionary and saved WAV exactly match the preceding default-path regression. Evidence: `gateup_quant_default_regression_comparison.json` and `all32_none_post_gateup_quant_regression_fr_fg.json`.

The new profile eliminates the remaining 1,188 standalone activation-quantizer launches in the profiled 33-step request. Fused gate/up/quant consumes 24.1% of GPU time, QKV plus down 21.7%, attention output projection 7.0%, fused normalization/quantization 7.6%, and native attention 5.3%. Projections therefore remain the main bottleneck despite the removed launch and memory round trip.

Evidence: `all32_gptq_dp4a_gateup_quant_v1*.json` and profile, `gateup_quant_graph_validation_v1.json`, `quality_suite/gateup_quant_audio_equivalence.json`, `attention_paired_gateup_quant_v1.json`, and `gateup_quant_memcheck.log`.

## HTTP streaming and fresh voice registration

Two sequential temporary loopback servers used identical text, cloned voice and seeds 501–520, with 20 complete measured requests per configuration. Both used native attention; the candidate added gate/up quantization fusion. All 20 complete PCM hashes and frame counts matched, no request truncated, and input-validation and cancellation/recovery checks passed (busy response 429, then successful response 200).

| Measurement | Separate quantizer | Fused quantizer |
|---|---:|---:|
| HTTP TTFA median | 95.79 ms | 88.86 ms |
| HTTP TTFA p95 | 96.91 ms | 90.45 ms |
| Engine TTFA median | 92.13 ms | 84.26 ms |
| Preparation median | 1.59 ms | 1.44 ms |
| Prefill median | 9.22 ms | 8.85 ms |
| Initial decode-step median | 2.360 ms | 2.146 ms |
| First codec decode median | 4.78 ms | 4.41 ms |
| Fresh-registration-to-first-PCM, one observation | 139.01 ms | 135.56 ms |

**The 6.93 ms HTTP difference cannot be attributed entirely to fusion.** Unchanged prefill and codec stages also ran faster in the candidate process. The same-process paired gain of 1.51 ms is the better estimate of this kernel change's contribution. Cross-process variability remains unexplained; CPU placement and clock behavior were not recorded for those HTTP runs. Fresh registration alone took 39.75 ms for the control and 38.21 ms for the candidate. Each fresh total is a single observation, not a latency distribution.

The boundary includes text preparation, full-codebook generation, FP32 codec decoding, CPU PCM conversion and local HTTP delivery. Cached-voice timing excludes initial reference encoding; fresh totals include voice registration. Both temporary servers were stopped. Existing supervisor services retain their previous configurations.

Evidence: `http_gateup_quant_comparison.json`, `http_gateup_quant{,_control,_health}.json`, and the corresponding server logs and stage-metric JSONL files.

## Follow-up bottleneck experiments

A 27-configuration sweep tests larger row/warp tiles for QKV, attention-output and down projections. Output/down do not improve over the selected plan. The best exact QKV variant (eight rows/eight warps) measures 6.19 versus 6.33 microseconds in the weight-ring microbenchmark, and all 504 expanded QKV comparisons match. Its ten same-process complete-generation pairs, however, give only **0.097 ms median paired gain**, with five improvements and five regressions. All ten complete PCM streams match. The wider plan remains experimental; the selected model keeps QKV at four rows/four warps. Evidence: `dp4a_wide_kernels.json`, `dp4a_wide_qkv_validation.json`, `dp4a_wide_qkv_plan.json`, and `attention_paired_wide_qkv_v1.json`.

The H200's PCI device reports NUMA node 2, while the benchmark initially permits all 256 logical CPUs across four nodes. A separate experiment keeps one model and the same graphs/weight addresses, changing only the calling thread's CPU affinity between requests. It rotates the five placement modes, excludes one warmup round, and measures five requests per mode. Same-seed complete float32 PCM matches across all placements.

| Calling-thread placement | Median TTFA | Median preparation |
|---|---:|---:|
| Original unrestricted affinity | **91.77 ms** | **2.46 ms** |
| Node 0 | 94.44 ms | 4.77 ms |
| Node 1 | 94.07 ms | 4.84 ms |
| GPU-local node 2 | 93.02 ms | 3.84 ms |
| Node 3 | 92.56 ms | 4.02 ms |

This change does not help and is not integrated into serving. It does not explain the earlier HTTP variation or test initial memory placement: model setup occurs before affinity changes, other existing threads remain unbound, and no memory-binding policy changes. [NVIDIA's affinity guidance](https://github.com/NVIDIA/gpu_affinity) instead recommends placement before significant computation or CUDA-context creation; a startup-placement experiment would therefore be a distinct test. The benchmark restores its original affinity and leaves other processes unchanged. Evidence: `cpu_affinity_v1.json` and `benchmark_cpu_affinity.py`.

## Reproduction

Run GPU work sequentially in the tested Torch 2.13 / Triton 3.7.1 environment. Use new result tags to preserve historical artifacts.

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_gateup_quant
/venv/moss-vllm/bin/python -m optimization.benchmark_gateup_quant --scale-sweep
/venv/moss-vllm/bin/python -m optimization.benchmark_gateup_quant --row-sweep
/venv/moss-vllm/bin/python -m optimization.validate_gateup_quant_consumers --scale-mode 4
/venv/moss-vllm/bin/python -m optimization.validate_attention_quant_graphs --gateup-quant --tag v1
compute-sanitizer --tool memcheck --error-exitcode 93 \
  /venv/moss-vllm/bin/python -m optimization.validate_gateup_quant_memory
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_paired \
  --gateup-quant --tag gateup_quant_v1 --pairs 10
/venv/moss-vllm/bin/python -m optimization.benchmark_dp4a_wide
/venv/moss-vllm/bin/python -m optimization.validate_dp4a_wide_qkv
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_paired \
  --wide-qkv --tag wide_qkv_v1 --pairs 10
/venv/moss-vllm/bin/python -m optimization.benchmark_cpu_affinity --tag v1 --rounds 5
/venv/moss-vllm/bin/python -m optimization.benchmark_quantization \
  --mode none --calibration optimization/results/gptq_v1_g32_d10 \
  --calibrated-backend dp4a --tag gateup_quant_v1 \
  --attention-block 32 --fused-residual --dp4a-reciprocal \
  --grouped-activation --fused-dp4a --dp4a-gateup --norm-layout-limit 8 \
  --prefill-buckets 128 160 256 512 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant
```

`quality_generate` supports the same model flags. The optional local development server is:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant
```

Register a reference through `/v1/voices`, then stream `/v1/audio/speech` as documented in the README. The first chunk is still 80 ms of PCM decoded from all 32 codebooks. This foreground development command binds only to loopback; a persistent service should use supervisor. The fusion is opt-in, requires compatible interleaved G32 weights/BF16 scales, and must be enabled before graph capture.

`gateup_quant_source_hashes.json` and `gateup_quant_sources.tar.gz` preserve this pass's source snapshot. Prior source snapshots and failing/negative experiment artifacts remain available.
