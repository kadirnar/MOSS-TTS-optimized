# G64 weight and activation experiments

All experiments retain **32 acoustic codebooks**, streaming and voice cloning. G64 below means **64 weights per quantization scale**, not fewer audio codebooks. The 50-ms target remains unmet. Both faster G64 candidates regress Chinese transcription diagnostics and remain unselected; the previously qualified G32 preset remains at **74.730 ms median warm cached-voice TTFA** in its qualifying paired run.

Selective G64 gate/up and QKV projections reach **74.061 ms median warm cached-voice TTFA**, versus **74.779 ms** for their same-process G32 control. Their **0.733-ms median paired gain** comes with worse Chinese transcription diagnostics, so that candidate is not promoted. Gate/up alone measures **74.308 ms**, with **0.439-ms paired gain**, but also regresses on both cloning suites.

## Calibration and numerical scope

Added group 64 to the calibration CLI and activation-weighted scale search. The first attempt encountered the older scale-search group guard before exporting weights; its failed log is retained. The corrected run exports all **144 projections** to `results/gptq_v1_g64_diag_d10`, using the same 866 training and 410 held-out activation rows, damping 0.1, activation ordering, static BF16 scales and signed codes −7 through 7. Scale search tests 33 clipping fractions from 0.5 to 1. Original model checkpoints are unchanged.

Six independent PyTorch scale-search comparisons—MSE and activation-diagonal losses on random, outlier and zero inputs—and two PyTorch-versus-Triton GPTQ integer-code comparisons match exactly. These validate calibration implementation, not speech quality.

Mean held-out projection MSE rises from **0.00221656 to 0.00240027**, about **8.3%**. Per-family means are:

| Projection | Selected G32 | G64 | Relative increase |
|---|---:|---:|---:|
| QKV | 0.00062003 | 0.00072175 | 16.4% |
| Attention output | 0.00353992 | 0.00390170 | 10.2% |
| Gate/up | 0.00110423 | 0.00128051 | 16.0% |
| Down | 0.00360205 | 0.00369711 | 2.6% |

`dp4a_group64.py` retains G32 activations and the SM90 programmatic dependency waits/hints. Its nominal expression repeats each G64 scale over two activation groups. A second expression factors the shared scale after combining the two activation-weighted integer sums. This deliberately changes FP32 operation order. The initial nominal pilot also differs in one BF16 intermediate despite no explicit factoring; nominal source expressions do not guarantee identical compiler lowering. The failed exact assertion is retained, and no G64 variant is described as bit-identical to selected G32.

`dp4a_group64_a64.py` additionally tests G64 activation groups, independent input/output group sizes, and register/thread layouts. Arithmetic checks use separately packed original DP4A projections plus standalone normalization and quantization. Whole-chain relative RMS includes propagation and quantization thresholds; the recorded screening bounds are not speech acceptance criteria. Fused output quantization matches standalone quantization of each candidate's own output.

## Kernel screens

GPU jobs run sequentially on the same H200 NVL, PyTorch 2.13/CUDA 13.0/Triton 3.7.1 runtime. Each full screen rotates all 36 actual layer weight sets and reverses/rotates configuration order. Timings use nine CUDA-graph samples with three layer-ring repetitions. Graph edges and private-stream replay are checked. These are three-projection chain timings, not TTFA.

| Screen | Configurations including control | Control µs | Best relevant alternative µs |
|---|---:|---:|---:|
| G64 weights, G32 activations, all three stages replaced | 15 | 35.366 | 35.799, gate/up factored |
| G64 activation and mixed-group layouts | 21 | 35.380 | 36.517, QKV-only A64 |
| Selective G64 weights, G32 activations retained | 17 | 35.420 | **34.989**, gate/up + QKV factored |

The first screen includes an original-kernel reference with G64 scales repeated in memory. It takes 35.416 µs. Nominal compact G64 raises gate/up registers from **162 to 201**, and down from **128 to 161**, slowing the chain to 40.940 µs. Factoring gate/up reduces its register count to **154**. Wider G64 activation tiles do not compensate for layout costs; the default full-A64 gate/up spills **92 registers** and the chain takes 51.207 µs. All wider-activation candidates remain unselected.

The selective screen isolates useful stages. Gate/up alone gains **0.332 µs** paired; gate/up plus QKV gains **0.438 µs**. Attention-output and down weights remain G32 in both complete-model candidates. No complete-model speed claim is made for the other layouts.

## Complete streamed requests

`group64_norm_experiment.py` provides an explicitly experimental per-model norm dispatch. It owns separate packed G64 buffers keyed by the original projection buffer; original BF16 prefill and G32 attention-output/down projections remain available. Control and candidates hold independent context/head graphs and first-32-audio graphs. The benchmark switches eager dispatch and graph sets together before each complete request.

Twelve measured rounds rotate/reverse control, gate/up-only, and gate/up-plus-QKV requests with matching seeds; one warmup round is excluded. All requests finish and produce finite audio. All twelve control PCM hashes match the prior qualified preset. Candidate PCM is different and is evaluated separately. Ninety-six private-stream eager/graph comparisons cover context boundaries through position 1023, head counts and returns to earlier positions; all IDs and logits match each candidate's own eager computation.

| Mode | Median TTFA | p95 | Median paired gain | Faster rounds |
|---|---:|---:|---:|---:|
| Selected G32 control | 74.779 ms | 75.356 ms | — | — |
| G64 gate/up only | 74.308 ms | 75.055 ms | 0.439 ms | 11/12 |
| G64 gate/up + QKV | 74.061 ms | 74.713 ms | 0.733 ms | 11/12 |

The initial 32-step LLM interval is **61.802 / 61.251 / 61.021 ms** respectively. BF16 prefill stays about 6.83 ms and first FP32 codec decoding about 4.52 ms. This supports locating the gain in decode projections. Outliers and slower pairs are retained. TTFA includes text preparation and the first playable 80-ms PCM chunk; voice encoding is cached and network transit excluded. No fresh-reference or HTTP improvement is claimed for these candidates.

## Cloning diagnostics

Both candidates generate all 48 bilingual test utterances each without truncation: **96 new WAVs**. The selected control is rescored in the gate/up-plus-QKV evaluation run with CUDA-FP16 Whisper-small and pinned WavLM; the gate/up-only evaluation uses the same runtime/models. All prompt text, seeds, voice assets, encoded prompt lengths and reference WAVs match their controls. Chinese scores use OpenCC and number normalization; English uses the existing text normalizer. Raw transcripts and scores remain saved.

| Suite / model | Normalized Chinese CER | English WER | Mean speaker cosine |
|---|---:|---:|---:|
| Original 16, selected G32 | 5.37% | 0% | 0.914313 |
| Original 16, G64 gate/up + QKV | **6.29%** | 0% | 0.914119 |
| Original 16, G64 gate/up only | **5.83%** | 0% | 0.915891 |
| Expanded 32, selected G32 | 1.18% | 0% | 0.914909 |
| Expanded 32, G64 gate/up + QKV | **1.93%** | 0% | 0.922514 |
| Expanded 32, G64 gate/up only | **1.85%** | 0% | 0.917104 |

Both candidates regress Chinese CER in both suites and remain unselected despite their sub-millisecond gains. All cases choose the intended reference among the four test assets, but that and higher speaker cosine in some comparisons do not establish speech equivalence. These reused-voice, machine-scored suites are not human listening or broad quality acceptance. Transcription differences include homophones and a short Chinese substitution; this is a diagnostic regression, not a claim that every differing transcript proves an audible defect.

## CUDA memory and concurrency checks

Compute Sanitizer **memcheck and unfiltered racecheck each pass 72 cases**, with zero errors, hazards or warnings. Three actual layers exercise five chained plans—nominal/factored G32 activations, the selective candidate, mixed activations and a spilled full-A64 tile—plus three attention-output variants. Private-stream graphs replay after real/zero/spike input changes. All returned buffers are finite and bit-identical to each candidate's own eager output. These checks cover the experimental operators, not a complete HTTP service or equivalence to G32 weights.

The full screens record **5,739 numerical comparisons**, including fifteen private-graph rows in the first screen; the other two screens additionally validate 38 private graphs. All candidates satisfy their documented screening bounds, and all 4,104 fused output-quantizer comparisons in the latter two screens match their standalone quantizers exactly. These tolerances do not convert the changed-precision candidates into exact replacements.

`results/group64_audit_v1/` retains PTX, Gluon IR, cubins, SASS and resource/hash records for control, nominal/factored G64, and wider/spilled activation plans. `group64_scale_validation_v1_provenance.json` clarifies that the validator's inherited rounding-history field describes the earlier G128 investigation; the successful G64 result is preserved unchanged.

## Documentation and next work

The reviewed [vLLM Marlin source documentation](https://docs.vllm.ai/en/v0.16.0/api/vllm/model_executor/layers/quantization/utils/marlin_utils/) lists group 64 among its supported sizes. This is compatibility context, not a new Marlin measurement or a claim that version 0.16 is current. Earlier full vLLM/SGLang comparisons remain documented in `REPORT_REVIEW.md`.

The recent [Fast NF4 dequantization paper](https://arxiv.org/html/2604.02556v1) uses a shared lookup table to accelerate NF4-to-FP16 decoding. Our selected uniform-integer DP4A path has no NF4 lookup table or separate FP16 weight expansion, so its reported gains do not transfer directly. The current experiments measure scale grouping and instruction/register costs in the actual MOSS kernels.

The selected profile still identifies the initial LLM decode, particularly projections, as the largest remaining interval. No experiment here proves 50 ms impossible or justifies reducing codebooks. A concrete next hardware experiment is sampled CTA timestamps around dependency waits and projection work: current resident intervals include waits, so separating them can guide exact-arithmetic scheduling. Instrumentation overhead must itself be measured before interpreting those timestamps.

## Reproduction

Use unique tags and run all GPU commands sequentially:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_group64_pdl --tag new_a32 --layers 36 --rounds 6
/venv/moss-vllm/bin/python -m optimization.benchmark_group64_a64 --tag new_a64 --layers 36 --rounds 6
/venv/moss-vllm/bin/python -m optimization.benchmark_group64_a64 --tag new_selective --layers 36 --rounds 8 --weight-only
/venv/moss-vllm/bin/python -m optimization.benchmark_group64_norm_paired --tag new_request --rounds 12
```

The quality-only flag `--g64-norm-projections up` or `up_qkv` is added to the complete selected quality command in `REPORT_PREFILL_POINTWISE.md`, with `--bulk-prefetch --prefill-qkv --prefill-pointwise`. Expanded prompts use `quality_prompts_expansion_v1.json` and seed base 8400. No G64 serving flag is installed.

`results/group64_pass_summary_v1.json` indexes the evidence. `results/group64_sources.tar.gz` and `results/group64_source_hashes.json` preserve the source snapshot. Both original supervisor services remain healthy with 32 codebooks and their existing voice caches; no temporary server is left running. No HTTP deployment or fresh-reference benchmark was performed for these rejected candidates.
