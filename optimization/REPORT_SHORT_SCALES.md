# Exact BF16 scale storage

This pass retains **all 32 codebooks**, streaming voice cloning, calibrated G32 weights, BF16 prefill and the FP32 codec. Exact BF16 storage for output/down scales gives **0.52 / 0.44 ms median paired TTFA gains** over ordinary FP32 storage in two independent fifteen-round comparisons. Candidate medians are **84.19 / 84.13 ms**. Relative to the preceding compressed-FP32 option, median paired gains are **0.32 / 0.20 ms**. These are small improvements; **50 ms remains unmet**.

The optional `--short-scales` flag is available in benchmark, quality generation and serving. It defaults off and is mutually exclusive with `--compressed-scales`. Both original supervisor services retain their previous configuration and process-local voice caches.

## Storage and arithmetic

The selected calibration export stores BF16 scale values. Output/down kernels previously used copies promoted to FP32 because earlier implicit compiler layouts changed floating reduction order with BF16 inputs. This pass revisits the storage with explicit Gluon integer and floating layouts. `short_scales.py` checks that every FP32 source value is exactly representable as BF16, creates all replacements before mutating the model, and installs them before graph capture. Non-G32, missing producer fusions and incompatible scale configurations are rejected.

The 72 selected buffers shrink from **288 MiB to 144 MiB**. This is an actual storage reduction, distinct from the preceding CUDA allocation compression experiment. No weight code or scale value changes. The integer DP4A result converts into the fixed floating layout before loading scales and performing the established multiply/FMA/reduction order. Output uses four rows/four warps; down uses four rows/two warps. Both use integer-group and integer-row parameters of one. No special allocator or new native extension is required by this option.

The compiled-kernel audit saves the actual SM90 cubins, PTX, Gluon IR and SASS for both selected shapes. Output changes from 30 to 32 registers, retains 2,048 shared bytes and reduces static barrier instructions from five to three. Down retains 128 registers, 1,024 shared bytes and twenty static barriers. Both variants have **zero spills**. Static DP4A, FMA and floating-add counts match their FP32 counterparts; BF16 down uses narrower scale loads. These are static code observations, not measured traffic or dynamic instruction counts, and the gain cannot be assigned to storage alone. See `short_scales_codegen_v1/summary.json` and `audit_short_scales_codegen.py`.

## Screening and correctness

The pilot compares 36 BF16 configurations plus FP32/compressed-FP32 controls across the two projection families. Its timing ring uses all 36 actual layer weights, scales and input vectors. Four rounds rotate/reverse option order. Three sampled layers, four recorded vectors and zero/spike cases yield **684 output comparisons, 664 exact**. The twenty failures belong to rejected compiler/layout configurations; no failed variant is installed. Some faster Triton variants change a few BF16 outputs despite unchanged scale values, demonstrating why storage equivalence alone is insufficient.

| Projection | Ordinary FP32 µs | Compressed FP32 µs | Selected BF16 µs |
|---|---:|---:|---:|
| Attention output | 4.955 | 4.930 | 4.857 |
| MLP down | 10.031 | 9.853 | 9.760 |

Expanded checks cover all 72 actual projections with twelve recorded vectors, four random magnitude ranges, zero and spike: **1,296 comparisons, all exact**. Sixteen padded/actual-row cases exercise private-stream CUDA graphs with N=1/3/4/5/37/4095/4096/4097 and both input widths. **Memcheck reports zero errors; racecheck reports zero errors and zero warnings.** These results are exact against the preceding calibrated G32 arithmetic, not against original unquantized BF16.

All sixteen original-suite and thirty-two additional-suite cloned-voice WAVs, their generation metadata and all eight saved reference files match their G32 controls byte for byte. Existing normalized diagnostics therefore apply without another ASR run: Chinese CER **5.37% / 1.18%** and English WER **0% / 0%**, respectively. These are reused four-voice diagnostic suites, not broad or human quality acceptance. Evidence is in `quality_suite/short_scales_audio_equivalence.json`.

## Complete streaming comparison

`benchmark_short_scales_paired.py` retains three graph sets in one process, using ordinary FP32, compressed FP32 and short BF16 scales. Each has the selected context and audio-head prefix buckets. Buffer attributes and dispatch flags are restored for eager execution as well as graph replay. Each independent run rotates/reverses three-way order for fifteen measured rounds after an excluded warmup triplet. The workload uses complete text, 145 prompt tokens and an already encoded 3.112-second Chinese reference. TTFA includes text processing, generation, first codec decode and CPU float32 PCM transfer; reference registration and HTTP are separate.

| Run | Ordinary FP32 median ms | Compressed FP32 median ms | BF16 median ms | BF16 paired gain vs ordinary ms | Faster rounds |
|---|---:|---:|---:|---:|---:|
| First | 84.780 | 84.548 | 84.192 | 0.524 | 15/15 |
| Independent repeat | 84.592 | 84.401 | 84.127 | 0.441 | 14/15 |

Candidate p95 is **84.704 / 84.664 ms**, better than ordinary FP32 (**85.488 / 84.803**) but slightly worse than compressed FP32 (**84.684 / 84.602**). All ninety measured complete float32 PCM streams match their triplet controls, with no truncation; both excluded warmup triplets also match. Each process additionally passes 96 private-stream full-graph comparisons covering context boundaries, fallback and three active head prefixes: **192 exact logit and sampled-ID checks** in total. Every emitted frame still uses all 32 codebooks.

Twelve rotating complete-decode GPU timing rounds in each run show **9.160 / 9.293 µs median paired gains** over ordinary FP32. Each timing samples 100 graph replays. Candidate graph medians are 2.11424 / 2.11445 ms, versus 2.12398 / 2.12386 ms. Preparation, prefill and first codec timings remain similar. Request noise and outliers are retained; the full HTTP or p95 difference must not be attributed solely to scale traffic. `short_scales_timing_summary_v1.json` preserves stage and paired summaries.

The updated candidate profile groups fused normalization/projection consumers and these Gluon projections under `_kernel`: **4,752 calls, 49.442 ms / 59.25% of GPU time** in the independent repeat. Backbone projections remain the principal bottleneck. This small storage gain does not establish an absolute limit on further acceleration.

The independent five-request benchmark without head-prefix graphs measures **85.281 ms median / 88.469 ms p95**, including an 89.074 ms request. Its complete 36-step diagnostic dictionary and saved WAV match the preceding G32 norm-projection control exactly. This validates the separate CLI path; it is not another paired improvement estimate. `short_scales_regression_comparison_v1.json` retains every timing and comparison.

## HTTP and fresh voice cloning

Two temporary loopback servers run sequentially, **compressed FP32 control first, BF16 short candidate second**, with identical text, reference and seeds 501–520. Twenty measured complete streams per server give **88.643 → 87.988 ms median HTTP TTFA**, p95 **91.358 → 88.225 ms**. All twenty PCM hashes and frame counts match; measured streams finish without truncation. Input-error checks pass, and cancellation/recovery returns `429` then `200` in both cases. Both temporary servers are stopped.

Internal engine medians are **84.675 / 84.253 ms**, initial decode steps **2.1502 / 2.1393 ms**, prefill **9.2181 / 9.2157 ms**, and first codec decode **4.7726 / 4.7659 ms**. Preparation also changes from 1.1663 to 1.0755 ms and seeding from 0.2481 to 0.2243 ms. The control's first several requests are slower. These scheduling/phase differences are retained; the whole sequential HTTP gain is not a kernel-only estimate. The independent same-process paired comparisons provide stronger attribution.

One continuous fresh-reference registration plus synthesis observation takes **130.77 / 129.13 ms**. Registration alone takes **36.37 / 37.40 ms**. These are single observations, not latency distributions or evidence of a repeatable fresh-cloning gain. The timer includes server WAV parsing/reference encoding and two loopback requests, but client base64/WAV construction occurs beforehand. Cached-voice TTFA excludes reference registration. Neither boundary reaches 50 ms.

Evidence: `http_short_scales_comparison_v1.json`, `http_short_scales_stage_summary_v1.json`, raw per-request JSON/JSONL files and server/client logs.

## Reproduction

Use `/venv/moss-vllm` (Torch 2.13, Triton 3.7.1) and the pinned source checkpoints/calibration captures. Run GPU jobs sequentially with fresh tags:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_short_scales --tag reproduce
/venv/moss-vllm/bin/python -m optimization.validate_short_scales all --tag reproduce
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_short_scales memory --tag reproduce_memcheck
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_short_scales memory --tag reproduce_racecheck
/venv/moss-vllm/bin/python -m optimization.benchmark_short_scales_paired --tag reproduce --rounds 15
/venv/moss-vllm/bin/python -m optimization.benchmark_short_scales_http --tag reproduce
/venv/moss-vllm/bin/python -m optimization.audit_short_scales_codegen --tag reproduce
```

Add `--short-scales` to the selected G32 norm-projection benchmark/quality commands in `REPORT_NORM_PROJECTION.md`. Quality and serving also accept `--audio-head-buckets`. A temporary foreground development endpoint is:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant \
  --scaled-dp4a --norm-projection --audio-head-buckets --short-scales
```

It binds to loopback. Persistent services should use supervisor. Voice registration and streaming use `/v1/voices` and `/v1/audio/speech` as documented in `README.md`.

The related [INT4 PTX experiment](REPORT_INT4_PTX.md) is rejected: exact nibble decomposition compiles into INT8 IMMA/conversion sequences on this H200 and loses substantially to DP4A. Instruction-format support is not evidence of useful native throughput. The short-scale gain instead comes from a measured exact-storage/layout change.

Artifacts include `short_scales_v1.json`, `short_scales_all_v1.json`, `short_scales_memory_{memcheck,racecheck}_v1.json`, `short_scales_paired_v1/v2.json`, their logs/profiles, and the cloning-equivalence file. `short_scales_pass_summary_v1.json` indexes the pass. `short_scales_sources.tar.gz` and `short_scales_source_hashes.json` include both short-scale and INT4-PTX code, reports and licenses, separately from checkpoints and measurements.
