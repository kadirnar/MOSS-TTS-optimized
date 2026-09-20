# Exact scaled-integer INT4 unpacking

All 32 codec codebooks remain enabled, including the first streamed PCM chunk. Voice cloning, BF16 prefill and the FP32 codec are retained. The 50 ms target remains unmet. This pass follows `REPORT_GATEUP_QUANT.md`; existing supervisor services are unchanged.

## Integer instruction change

`dp4a_scaled.py` moves each signed INT4 nibble into the high four bits of an INT8 byte. The resulting byte represents sixteen times the original weight. Two signed DP4A instructions compute eight products; an arithmetic right shift removes the factor of sixteen before conversion to FP32. The two tested modes shift either each eight-product result or the complete 32-product group.

This is a change in execution, not weight precision or calibration. Even the worst signed-byte inputs fit: the scaled 32-product sum has magnitude at most 524,288, comfortably inside INT32. The selected kernel preserves the original floating reduction, scale arithmetic and BF16 rounding boundaries. The [PTX ISA description of signed DP4A](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a) specifies the byte interpretation used here.

The original unpack sequence uses seven shifts/masks/sign-extension operations per eight weights. The scaled form uses three shifts/masks and one shift either per dot or per group. Static PTX for fused gate/up has zero `mad.lo.u32` instructions versus 512 previously, with the same 512 DP4A instructions, 256 activation-load instructions and 38 barriers. This is a compiler-artifact comparison, not a hardware-counter measurement.

A 62-configuration search covers both shift placements, row tiles and warp counts across all four projection shapes. Each candidate is checked on three layers and six real/edge inputs. Selected configurations are:

| Projection | Rows / warps | Shift placement | First ring timing, old → new | Independent repeated ring, old → new |
|---|---|---|---:|---:|
| Gate/up + SiLU + G32 quantization | 32 / 4 | After group sum | 17.86 → 16.73 µs | 17.76 → 17.11 µs |
| QKV | 4 / 4 | After group sum | 6.39 → 6.07 µs | 6.43 → 6.10 µs |
| Attention output | 4 / 4 | After each dot | 5.41 → 4.75 µs | 5.43 → 5.12 µs |
| Down | 4 / 2 | After each dot | 10.20 → 9.77 µs | 10.24 → 10.13 µs |

Each timing uses eight distinct weight/scale pairs and prequantized inputs, with a median of nine graph replays. The selected kernels use respectively 168/32/30/128 registers per thread, 2048/2048/2048/1024 shared bytes, and no spills. Expanded checks cover **2,016 cases**: all 36 layers, four projection families, 12 recorded activations plus zero/spike inputs. Every BF16 output and fused INT8/FP32-scale tensor matches the previous selected implementation exactly, using private CUDA streams.

The integer test additionally checks all 16 weight values against all 256 signed activation values, in each of eight independent lanes and in all eight lanes together, plus 65,537 random dots. **102,401 dots per shift mode** match both a separate integer reference and the previous assembly. Compute Sanitizer memcheck also passes 16 private-stream graph cases covering actual/padded dimensions, both modes and masked activation pointers, with zero errors.

Evidence: `dp4a_scaled_kernels_v1.json`, `dp4a_scaled_validation_v1.json`, `dp4a_scaled_plan_v1.json`, `dp4a_scaled_{up,qkv,out,down}_v1.{ptx,ttgir}`, `dp4a_scaled_memory_validation.json`, and `dp4a_scaled_memcheck.log`.

## Complete streaming model

Five complete requests measure **87.61 ms median / 87.72 ms p95 TTFA**, with median decoder-step time 2.219 ms. The entire 36-step teacher-forced diagnostic dictionary and saved WAV match the preceding gate/up-fusion path. Sixteen full-backbone comparisons across context boundaries, the 1024-position fallback and returns to earlier positions match all text and audio logits on a private stream.

Ten same-process comparisons alternate separately captured old/new graph sets, reversing order each pair and using identical text, voice and seed. One warmup pair is excluded:

| Path | Median TTFA | p95 TTFA |
|---|---:|---:|
| Previous fused gate/up path | 88.91 ms | 89.00 ms |
| Scaled-integer unpacking | **87.67 ms** | **87.89 ms** |

The **median paired gain is 1.23 ms**, with all ten pairs faster and all ten full float32 PCM streams identical. A late shift in execution speed affects both configurations: unchanged prefill falls from about 9.21 to 8.84 ms and codec decode from 4.77 to 4.41 ms. One pair spans that shift and records a 6.87 ms difference. All raw samples are retained; that outlier is not attributed to unpacking, and the fastest 82.07 ms observation is not reported as the selected median.

All **16 generated bilingual cloning WAVs and four references** match the preceding calibrated implementation byte for byte. Prompt, seed, frame-count, finite-output and termination metadata also match. The existing small-suite diagnostics therefore apply to identical artifacts: Chinese CER 5.37%, English WER 0%, mean speaker cosine 0.9143. This does not establish equality with the original unquantized BF16 model or corpus-wide quality acceptance.

The profile still attributes **51.9%** of GPU time to projections (44.87 ms across 4,752 calls), followed by fused normalization/quantization at 7.7%. The remaining layout-conversion barriers in projection PTX motivate a future native CUDA implementation with explicit arithmetic order; no such speedup is claimed here.

Evidence: `all32_gptq_dp4a_scaled_dp4a_v1*.json` and profile, `attention_paired_scaled_dp4a_v1.json`, `dp4a_scaled_graph_validation_v1.json`, and `quality_suite/scaled_dp4a_audio_equivalence.json`. An initial graph-validator CLI omission failed before loading the model, was corrected, and remains in `dp4a_scaled_graph_validation_v1.argparse_failed.log`.

## HTTP streaming and fresh cloned voices

Sequential temporary servers measure 20 complete requests each with identical text, reference and seeds 501–520. Both paths use native attention and gate/up quantization fusion; only the candidate enables scaled unpacking. Every full PCM hash and frame count matches, no measured request truncates, and validation-error plus cancellation/recovery checks pass (429 while busy, then 200).

| Measurement | Control | Scaled unpacking |
|---|---:|---:|
| HTTP TTFA median | 94.81 ms | 90.24 ms |
| HTTP TTFA p95 | 97.51 ms | 90.69 ms |
| Engine TTFA median | 90.96 ms | 87.19 ms |
| Preparation median | 1.93 ms | 1.12 ms |
| Prefill median | 9.22 ms | 9.21 ms |
| Initial decode-step median | 2.307 ms | 2.233 ms |
| First codec decode median | 4.79 ms | 4.77 ms |
| Request seeding median | 0.383 ms | 0.230 ms |
| Fresh-registration-to-first-PCM, one observation | 141.52 ms | 131.81 ms |

The 4.57 ms HTTP difference includes unrelated preparation/seeding variation and exceeds the controlled 1.23 ms paired gain. The prior pass's 88.86 ms HTTP run was faster than this candidate run; these sequential measurements should not be presented as a monotonic historical improvement. Fresh registration alone takes 39.15 / 35.62 ms, and each fresh total is a single observation. Cached timing excludes reference encoding; fresh totals include it. All measurements use the actual original codec and all 32 codebooks.

The test harness terminated both temporary servers, and port 18084 is free. Original supervisor processes remain unchanged. Evidence: `http_scaled_dp4a_comparison_v1.json`, `http_scaled_dp4a_stages_comparison_v1.json`, and control/scaled server logs, health snapshots and stage JSONL files.

## Request seeding

The first HTTP requests spend several milliseconds in global random seeding, but measured warm medians are much smaller (0.23–0.38 ms in the comparison above). The installed implementation and [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.manual_seed.html) distinguish seeding the current CUDA device from seeding every device. This server's sampling and captured RNG state use its single GPU worker's current CUDA device.

Two hundred alternating standalone calls after model warmup measure 0.1143 ms median for `torch.manual_seed` and 0.0023 ms for `torch.cuda.manual_seed`. Ten full-generation same-process pairs, now timing **from before seeding through first PCM**, measure 87.34 versus 87.18 ms median; median paired gain is **0.199 ms**, with nine improvements and one regression. All ten complete float32 PCM streams match. Seed-call medians inside those requests are 0.245 and 0.014 ms.

An opt-in server flag, `--cuda-only-seed`, retains that scoped seeding behavior. Its separate 20-request HTTP validation matches every previous PCM hash/frame count, has no truncation, and passes validation errors and cancellation/recovery. HTTP median/p95 is 91.43/92.98 ms, **slower than the preceding 90.24/90.69 ms run** despite a 0.014 ms median seed cost. Engine median also rises to 88.71 ms; no end-to-end HTTP improvement is claimed from that separate process. Fresh registration plus first PCM takes 131.20 ms in one observation. The temporary server is stopped, and global seeding remains the default. Evidence: `request_seed_v1.json`, `http_cuda_seed_comparison_v1.json`, and `http_cuda_seed_stage_summary_v1.json`.

## Additional hardware experiment and current research

`dp4a_pair.py` shares one activation load between gate and up integer dot products. Six row/warp variants all pass their 18 numerical checks, but none beats the selected kernel. Best timing is **17.58 versus 17.06 µs**; the largest low-warp tiles regress severely. The experiment is retained without integration. Evidence: `dp4a_pair_kernels_v1.json`.

The September 2026 [multi-shell Leech-lattice study](https://arxiv.org/abs/2609.02652) distinguishes the stored quantization rate from the actual GPU execution format and shows that packing/decode cost can erase nominal compression gains. The September [Qwen3-8B ternarisation study](https://arxiv.org/abs/2609.09240) likewise reports quality costs and a preliminary packed GEMV slower than FP16 cuBLAS. Neither result establishes a performance or quality limit for this speech checkpoint. This pass therefore measures exact changes to the existing calibrated format.

Current [PTX 9.4 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix) also introduces `ldmatrix.s8.s4` and lists `sm_90a` support. It is relevant future work, but the installed Triton assemblers report CUDA 12.8 and 12.9, and the host driver advertises CUDA 13.0. No unsupported toolkit, host-driver change or untested expanding-load kernel is installed or claimed.

## Reproduction

Run GPU commands sequentially. Use fresh tags/output names to preserve historical measurements. Selected runtime is Torch 2.13 / Triton 3.7.1 in `/venv/moss-vllm`.

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_dp4a_scaled --tag reproduce
/venv/moss-vllm/bin/python -m optimization.validate_dp4a_scaled \
  --sweep-tag reproduce --tag reproduce
compute-sanitizer --tool memcheck --error-exitcode 94 \
  /venv/moss-vllm/bin/python -m optimization.validate_dp4a_scaled_memory
/venv/moss-vllm/bin/python -m optimization.validate_attention_quant_graphs \
  --scaled-dp4a --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_paired \
  --scaled-dp4a --tag scaled_dp4a_reproduce --pairs 10
/venv/moss-vllm/bin/python -m optimization.benchmark_request_seed --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_quantization \
  --mode none --calibration optimization/results/gptq_v1_g32_d10 \
  --calibrated-backend dp4a --tag scaled_dp4a_reproduce \
  --attention-block 32 --fused-residual --dp4a-reciprocal \
  --grouped-activation --fused-dp4a --dp4a-gateup --norm-layout-limit 8 \
  --prefill-buckets 128 160 256 512 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant --scaled-dp4a
```

`quality_generate` accepts the same model flags. `benchmark_scaled_http --tag reproduce` owns two sequential temporary loopback servers and terminates each after measurement. For a manually controlled development endpoint:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant --scaled-dp4a
```

Register a reference through `/v1/voices`, then stream `/v1/audio/speech` as documented in the README. The first chunk remains 80 ms of PCM decoded from all 32 codebooks. The foreground development command binds only to loopback; use supervisor for a persistent service. The new flag is opt-in, requires the tested interleaved G32/scale configuration and fused gate/up quantizer, and must be enabled before graph capture.

Add `--cuda-only-seed` to test the separately qualified single-device seeding option. Its small same-process gain does not eliminate the observed HTTP variability. `scaled_dp4a_source_hashes.json` and `scaled_dp4a_sources.tar.gz` preserve this pass's source snapshot.
