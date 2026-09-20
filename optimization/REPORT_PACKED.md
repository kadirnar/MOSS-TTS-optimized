# Packed INT4 projections and bounded decode graphs

The 50 ms target remains open. This pass keeps **all 32 codebooks**, voice cloning, BF16 prefill, and the FP32 streaming codec. The previous calibrated work supplied a tested 106.43 ms in-process variant and a historical 101.53 ms HTTP measurement. Stage instrumentation below corrects the earlier claim that HTTP used a shorter prompt: the fixture text is identical and both produce 145 prompt tokens. The existing BF16 and FP8 supervisor services remain unchanged.

## Projection layout experiments

The prior profile assigned 54.2% of GPU time to DP4A projections and fused gate/up. `dp4a_packing.py` evaluates two ways to reduce the conversion cost:

- A PTX `prmt` path expands four adjacent signed nibbles with fewer scalar bit operations. The signed-byte interpretation follows [NVIDIA's PTX instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt).
- An interleaved representation stores eight values in one 32-bit word: four in the low nibbles and four in the high nibbles. One 64-bit activation load supplies two DP4A inputs. The original checkpoint and exported integer values are unchanged; only the inference buffer layout changes.

The calibrated scales were originally BF16 values promoted to FP32. Selected projections now store those scales in BF16 and convert to FP32 for arithmetic. Installation checks exact value preservation before changing any buffers; this is not additional scale quantization.

The tuner measured 192 combinations across four real projection shapes, two layouts, two scale-storage types, three row tile sizes and four warp counts. Eight distinct weight **and scale** pairs exceed L2 capacity; grouped activation quantization is included. All checks passed a relative-RMS tolerance against the separately validated original kernels. Exhaustive tests covered all 65,536 four-nibble words, including -8, and 65,536 interleaved packing rows. A further 72 real/edge input checks passed on a non-default stream.

| Projection | Previous operator | Fastest packed | Selected matching-layout operator |
|---|---:|---:|---:|
| QKV | 10.03 µs | 8.27 µs | 8.30 µs |
| Attention output | 8.15 µs | 7.05 µs | 7.05 µs |
| Gate/up + SiLU | 21.63 µs | 19.40 µs | 19.42 µs |
| MLP down | 15.02 µs | 12.38 µs | 14.25 µs |

The fastest plan matched 54/72 additional outputs exactly, with maximum relative RMS 0.00004252. Small reduction-order differences nevertheless changed every generated evaluation WAV. Its normalized Chinese CER increased from 5.37% to 7.59%, English WER from 0% to 0.78%, and mean speaker cosine fell from 0.9143 to 0.9007. All 16 utterances finished and retained the intended top-scoring reference, but this plan was not selected for the new serving test.

The next selection filters every measured configuration through up to 18 exact comparisons per projection. The chosen plan passes all 72 and uses interleaving throughout, with `(rows, warps)` of QKV `(4,4)`, output `(2,2)`, gate/up `(4,4)`, down `(2,2)`. QKV and gate/up scales use BF16 storage; the others use FP32. The plan is `results/dp4a_packing_exact_plan.json`. The name describes the selection checks, not a universal mathematical equivalence guarantee on other runtimes or GPUs.

Evidence: `dp4a_packing_kernels.json`, `dp4a_packing_validation.json`, `dp4a_packing_exact_selection.json`, `packed_kernel_resources.json`, and separate compiler PTX/IR artifacts.

## Decode attention capacity

The original decode graph launches attention splits for the full 1,024-token KV allocation, even when most splits are beyond the current position. `DecodeContextBuckets` captures additional graphs for capacities 128, 256 and 512, selecting the smallest capacity strictly greater than the current zero-based position. The 1,024-token graph remains the long-context fallback. KV storage and every valid causal entry remain intact.

All 16 attention/KV boundary checks matched exactly. All 12 complete-backbone checks also matched exactly, including transitions at 127/128, 255/256 and 511/512, full-capacity position 1023, return to smaller capacities, and non-default-stream execution. Invalid positions are rejected. At position 144, the isolated cold-cache operation fell from 15.36 to 14.14 µs.

Evidence: `decode_bucket_kernels.json` and `decode_graph_validation.json`.

## Complete-request results

These rows use the same 145-token cloned-voice fixture: warm batch one, cached encoded 3.112-second reference, complete text to first complete 80 ms CPU PCM chunk. Processing, prefill, generation, codec and CPU copy are included. HTTP, network transit and fresh waveform encoding are excluded. Each row has five measured complete requests after one warmup, with all 32 codebooks and a 400-token budget.

| Configuration | Median TTFA | p95 | Median decode |
|---|---:|---:|---:|
| Previous calibrated layout-constrained path | 106.43 ms | 106.69 ms | 2.770 ms |
| Fastest packed layout | 95.76 ms | 96.15 ms | 2.479 ms |
| Fastest packing + bounded attention graphs | 94.69 ms | 94.88 ms | 2.449 ms |
| Matching-layout packing + bounded attention graphs | 98.69 ms | 100.32 ms | 2.517 ms |

The bounded-graph change preserved the recorded 36-step diagnostics and saved generated WAV relative to the corresponding unbounded fast-packing run. The matching-layout candidate recovered the prior calibrated path's entire recorded teacher-forced validation and saved WAV: relative RMS 0.0138037, active top-1 0.6571485 and KL 0.0285592 versus upstream BF16. These remain quantized-model diagnostics, not BF16 equivalence.

The fastest packing profile still spends 49.9% of GPU time in projections, 7.1% in fused normalization/quantization and 10.0% in attention plus its reduction. The continued projection bottleneck motivates further memory-layout and arithmetic work; the new graph capacities address a measured secondary cost. The timing results do not establish an absolute lower bound for 50 ms.

The selected matching plan uses 40/48/40/125 registers per thread for QKV/output/gate-up/down, with zero spills. Its shared-memory allocations are 4096/2048/4096/2048 bytes. The faster down kernel used 128 registers, one warp and a different reduction order; register count alone did not determine the selection. Compiler evidence is recorded rather than inferred from bit width.

Raw evidence is in `all32_gptq_dp4a_packed*.json` and their profiles. Full quality and HTTP checks are recorded separately so a faster microbenchmark is not presented as a complete deployment.

## Cloning regression

The matching plan plus bounded decode graphs generated all 16 bilingual samples without truncation. Every output WAV and all four reference WAVs were **byte-identical** to the prior layout-constrained calibrated configuration. Text, seed, voice, frame count and termination metadata also match. Evidence: `quality_suite/packed_exact_audio_equivalence.json`.

The existing diagnostic results therefore apply to those identical files: Chinese CER 5.37%, English WER 0%, mean speaker cosine 0.9143, intended reference top-1 in all 16 cases. This is an equivalence check against the previous calibrated path, not a new ASR run or equivalence to the original BF16 checkpoint. The corpus remains small and repeatedly explored; original BF16's earlier Chinese CER was 4.46%.

The quality generation exercised the 256- and 512-token graphs across 1,564 decode calls. The separate full-backbone boundary checks cover 128 and 1024 as well. All 32 codebooks and valid causal KV history are retained. No input text or generated audio is cached.

## HTTP comparison and timing audit

The first packed serving check measured 101.34 ms median / 104.83 ms p95 over 20 complete cloned-voice requests, with one fresh-reference-to-PCM observation of 144.95 ms. This was close to the historical calibrated HTTP result and did not establish a serving gain by itself.

Optional local JSONL stage diagnostics were then added behind `--metrics-file`. They record timing and completion metadata after generation, without text, audio or voice identifiers. A controlled sequential comparison used the same current source, runtime, 145-token prompt, reference WAV and seeds 501–520 in fresh temporary servers:

| Configuration | HTTP median / p95 | Engine median | Initial decode step | First codec frame |
|---|---:|---:|---:|---:|
| Previous calibrated layout-constrained preset | 107.58 / 109.61 ms | 104.01 ms | 2.759 ms | 4.767 ms |
| Matching packed plan + bounded decode | 100.25 / 101.17 ms | 96.52 ms | 2.526 ms | 4.769 ms |

Each row contains 20 complete requests after a warmup; none truncated. Preparation was approximately 1.09 ms, prefill 9.21 ms, seed setup 0.22–0.23 ms and total time outside the engine approximately 3.7 ms. This supports a decode improvement, with similar prefill and codec costs. One fresh registration-to-first-PCM observation was 152.93 ms calibrated and 145.34 ms packed; those are individual observations, not latency distributions. Cancellation/recovery and error checks passed for both temporary endpoints, which were stopped afterward. Existing supervisor services were left running unchanged.

Evidence: `http_packed_exact_buckets.json`, `http_packed_stage_probe.json`, `http_calibrated_stage_probe.json`, their JSONL stage records, and `http_packed_stage_comparison.json`. The earlier report's explanation that HTTP used shorter text was checked against `fixture.pt` and corrected: it is the same text and prompt length. Historical independent timing runs are retained, but cannot be used to subtract HTTP overhead or prove an improvement between configurations.

## Direct activation loads inside DP4A

Compiler PTX showed shared-memory transfers to convert the separately loaded activation tensor into the weight's register layout. `dp4a_direct.py` embeds predicated 64-bit activation loads and both four-byte integer dot products in one inline PTX expression. It retains the same signed integer values, group scaling, BF16 rounding and 32-codebook output. Invalid padded K positions initialize both activation words to zero and skip the load.

Forty-eight row/warp candidates were measured on eight distinct weight-and-scale pairs. Checks used three real layers, four recorded inputs per layer, plus zero and spike inputs, on a private stream. The selected plan enables direct loads only for candidates passing 18/18 exact comparisons and exceeding 2% operator gain. This selected QKV `(rows=4, warps=4)` and MLP down `(4,2)`; output and gate/up retain the preceding matching plan.

| Projection | Previous matching operator | Direct selected | Registers/thread | Shared bytes |
|---|---:|---:|---:|---:|
| QKV | 8.25 µs | 7.97 µs | 32 | 2048 |
| MLP down | 14.21 µs | 12.02 µs | 128 | 1024 |

Neither selected kernel spills registers. The static PTX barrier count falls from 7 to 5 for QKV and 26 to 20 for down; these are compiler instruction counts, not hardware-counter measurements. QKV/down selected checks match exactly in 36/36 cases. All candidate errors remain below the diagnostic tolerance; the fastest down candidate has only 13/18 exact matches and was not selected.

The first five-request full-model run measured 94.64 ms median but **134.94 ms p95**, including one 144.55 ms request with a 52.50 ms decode step. That outlier is retained; its cause was not established. A ten-request repeat measured **94.04 ms median / 96.07 ms p95**, with all requests complete. Both runs reproduce the preceding calibrated teacher-forced diagnostics and saved WAV exactly. This is evidence of a gain on the tested workload, not a latency guarantee.

All 16 bilingual cloning WAVs and four reference WAVs match the prior selected packed path byte for byte; frame, text, seed, voice and termination metadata also match. The prior diagnostic CER 5.37%, WER 0% and speaker cosine 0.9143 therefore apply to these identical files, with the same small-suite limitations. The new profile still assigns 50.0% of GPU time to projection kernels, 7.2% to normalization/quantization and 9.1% to attention/reduction. All 32 codebooks remain enabled.

Evidence: `dp4a_direct_kernels.json`, `dp4a_direct_exact_plan.json`, `direct_*.ptx`, `all32_gptq_dp4a_direct_exact_buckets_{v1,repeat}*.json`, their profiles, and `quality_suite/direct_exact_audio_equivalence.json`.

Compute Sanitizer memcheck reported **zero errors** on eight additional cases with seven output rows, K lengths 32/96/160/384, paired and unpaired projections, padded reduction lanes, and private-stream execution. These targeted inline-load checks are recorded in `dp4a_direct_padding.json` and `dp4a_direct_padding_memcheck.log`; the full model was not run under memcheck.

The direct-load temporary HTTP service measured **91.11 ms median / 93.29 ms p95** across 20 complete requests with a cached cloned voice. Its stage medians were 87.40 ms inside the engine, including 1.10 ms preparation, 8.84 ms prefill, 2.264 ms per initial decode step and 4.401 ms for the first codec frame. One contiguous fresh-reference registration-to-PCM request took **130.33 ms**, including 34.88 ms registration. All 32 codebooks remained enabled; error, cancellation and recovery checks passed.

Because this gain appeared larger than the in-process gain, the preceding packed plan was repeated afterward in another fresh temporary server. It measured **93.21 ms median / 101.70 ms p95** over the same 20 seeds, versus its earlier 100.25 ms median. Its engine/prefill/codec medians were 90.15/8.84/4.400 ms, and initial decode averaged 2.349 ms. Thus the closer comparison supports a **2.10 ms HTTP median improvement**, not the apparent 9.14 ms difference against the earlier control. The cause of timing variation was not isolated; both control runs and all tail samples are retained. Neither absolute results nor their differences are a service-level guarantee.

Evidence: `http_direct_stage_probe.json`, `http_direct_stage_summary.json`, `http_packed_control_probe.json`, `http_direct_control_comparison.json`, and the corresponding JSONL stage records. Both temporary endpoints were stopped; the original BF16 and FP8 supervisor services remain unchanged. `packed_source_hashes.json` records the source snapshot for this pass.

## Reproduction

Run GPU measurements sequentially. The trials use the existing isolated `/venv/moss-vllm` environment: Torch 2.13.0+cu130, Triton 3.7.1 and the H200 NVL. They consume the fixed calibrated export from `REPORT_CALIBRATED.md`.

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_dp4a_packing
/venv/moss-vllm/bin/python -m optimization.validate_dp4a_packing
/venv/moss-vllm/bin/python -m optimization.select_exact_packing
/venv/moss-vllm/bin/python -m optimization.benchmark_decode_buckets
/venv/moss-vllm/bin/python -m optimization.validate_decode_graphs
/venv/moss-vllm/bin/python -m optimization.benchmark_quantization \
  --mode none --calibration optimization/results/gptq_v1_g32_d10 \
  --calibrated-backend dp4a --tag packed_exact_buckets_v1 \
  --attention-block 32 --fused-residual --dp4a-reciprocal \
  --grouped-activation --fused-dp4a --dp4a-gateup --norm-layout-limit 8 \
  --prefill-buckets 128 160 256 512 \
  --packing-plan optimization/results/dp4a_packing_exact_plan.json --decode-buckets
```

The streaming server accepts the same optional plan and bucket flags in addition to `--calibration`. Packing must happen before any graph capture; decode buckets capture after the full-capacity graph. Defaults do not enable either experiment.

For the direct-load extension, run `optimization.benchmark_dp4a_direct`, then use `results/dp4a_direct_exact_plan.json` instead of `results/dp4a_packing_exact_plan.json`, with a new result tag. The selected plan is fixed by operator timing and exact-comparison criteria before full-model quality evaluation. The same plan path works with `optimization.server` and `optimization.quality_generate`.

For a foreground local serving test:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json --decode-buckets
```

Register a reference through `/v1/voices` and stream `/v1/audio/speech` using the README API. A persistent instance service should use supervisor. The command does not expose a public port and does not replace either existing endpoint.

## Research direction

The [codec multi-token/speculative decoding paper](https://arxiv.org/abs/2410.13839) and the newer [RACER retrieval/logit speculation paper](https://arxiv.org/abs/2604.14885) motivate investigating draft acceptance as a separate experiment. MOSS's existing 32 delayed codebook heads are not trained future-token prediction heads, so their presence alone does not implement those algorithms. The [2026 block-wise speech architecture](https://arxiv.org/abs/2604.12438) uses a different FastSpeech/Mimi system; its published latency is not evidence that this MOSS checkpoint has reached 50 ms. No model substitution or codebook reduction was made here.
