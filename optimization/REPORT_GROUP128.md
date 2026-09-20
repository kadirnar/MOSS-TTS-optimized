# G128 projection and calibration experiments

All experiments retain the requested MOSS-TTS-v1.5 8B Delay model, **all 32 codebooks**, streaming PCM and voice-reference conditioning. G128 denotes the INT4 **weight quantization group size**, not the number of audio codebooks. BF16 prefill, embeddings and output heads and the FP32 codec remain in use. The 50 ms target remains unmet. G128 is experimental and has not replaced either supervisor service or the selected G32 path.

## Implementation and calibration

`dp4a_group128.py` adds interleaved packed INT4 projections using signed integer DP4A, with INT8 activation groups of either 32 or 128 elements. It includes a fused attention reduction/activation quantizer for A128. `dp4a_group128_norm.py` fuses residual/RMSNorm/activation quantization into QKV and paired gate/up/SiLU projections, optionally including the following activation quantizer. The benchmark and quality entry points accept `--g128-plan`; serving has no G128 preset. The checkpoint files are unchanged, and G128 exports are separate from G32.

The first full export uses the same eight calibration utterances (866 rows), four held-out utterances (410 rows), activation ordering and 0.1 damping as the preceding G32 export. GPTQ improves held-out projection MSE over round-to-nearest in 140/144 projections; the median ratio is 0.4436. Its median MSE is nevertheless 1.5357 times G32's. See `group128_calibration_comparison_v1.json`.

Added optional `--scale-search mse|diagonal` to calibration. The Triton kernel searches 33 clipping fractions from 1 to 0.5, using BF16-rounded scales and reconstruction. The diagonal objective weights squared weight error by training activation energy; it is a proxy, not full AWQ or a full output-error optimum. No held-out rows enter scale selection or GPTQ feedback.

Plain MSE scale selection improves nine of twelve pilot projections, but makes layer-zero down and QKV held-out MSE 3.39 and 4.25 times worse. It is rejected and has only a partial twelve-projection export. Activation-weighted selection improves eleven of twelve pilot cases and, after expansion, **140/144 projections**. Median held-out MSE is **0.8514 times max-scale G128**, or **1.3000 times max-scale G32**. Lower projection error is not assumed to imply better generated speech. The complete export is `gptq_v1_g128_diag_d10`; statistics are in `g128_diagonal_full_comparison.json`.

## Operator experiments and checks

Across Triton 3.7.1 and isolated Triton 3.8.0, the projection search measures **174 configurations / 3,132 input checks**, with no compilation failures. All are within the 0.001 relative RMS pilot threshold, but only 2,353 checks are bitwise exact. These kernels are not presented as arithmetic-preserving replacements for G32. The microbenchmarks rotate eight separate weight/scale buffers and report nine graph-timing rounds.

The producer/consumer fusion search covers another **64 configurations / 576 checks**. A128 fusion is slower and is omitted from its final plan. A32 QKV fusion improves 8.595 to 8.036 microseconds. Gate/up plus following activation quantization improves 22.256 to 19.951 microseconds. All checked producer intermediates match; some projection outputs differ slightly. Raw sweeps, including differences, remain in `group128_kernels_*.json` and `group128_norm_*.json`.

Twenty private-stream CUDA graph cases cover padded/actual projection dimensions, attention split counts, poisoned future attention slots, and residual/no-residual producer fusions. Independent integer-dot references and attention/quantizer references pass. Compute Sanitizer reports zero memcheck errors and zero racecheck hazards under Triton 3.7.1. These sampled checks do not replace complete speech evaluation.

The scale-search validator passes twelve independent Torch grid-search cases and two GPTQ feedback comparisons. Memcheck reports zero errors. An initial reference implementation failed one BF16 scale comparison because Python-scalar `/7` used reciprocal multiplication, moving an exact rounding tie. Tensor-valued division agrees with explicit Triton `div.rn` and the float64 diagnostic. The search kernel was unchanged. The failed fixture and correction are preserved in `scale_search_initial_disagreement.pt` and `scale_search_reference_audit.json`.

## Complete-request measurements

Each row below is a separate five-request process: warm batch one, complete Chinese text, cached 3.112-second cloned-voice reference, 145 prompt tokens, first complete 80 ms PCM chunk, all 32 codebooks. Processing, prefill, decoding, codec and CPU copy are included; fresh reference encoding and network are excluded. These are **not alternating paired controls**; small differences may include runtime variation. Output-head prefix graphs are disabled in these rows.

| G128 experiment | Median TTFA ms | p95 ms |
|---|---:|---:|
| Original-layout DP4A, max scales, A128 | 101.74 | 102.07 |
| Packed DP4A, max scales, A128 | 89.04 | 89.10 |
| Plus attention quantizer fusion, A128 | 87.06 | 87.52 |
| A32, producer fusion, separate final quantizer | 88.24 | 89.03 |
| A32, also fuse final gate/up quantizer | 84.33 | 84.54 |
| Same family, Triton 3.8 tuned plan | 83.15 | 85.56 |
| Activation-weighted scales, A32, Triton 3.7 | 84.89 | 85.34 |
| Activation-weighted scales, A32, Triton 3.8 | **83.39** | **83.80** |

The final row's initial decode step is 2.0947 ms median. The isolated compiler environment retains Torch 2.13/cuDNN 9.20 and overrides the declared Triton dependency. It is a compiler experiment, not a supported full vLLM/Inductor environment. Compiler-dependent arithmetic changes require separate quality evaluation.

The post-change selected G32 regression measures **85.52 ms median / 85.72 p95**, without head prefixes. All 36-step diagnostic values and the saved WAV exactly match the preceding G32 norm-fusion artifact. See `post_g128_regression_comparison_v1.json`. Its prior head-prefix paired result remains 84.84 ms and prior HTTP median 88.72 ms; neither is replaced by an unpaired G128 measurement. No G128 HTTP or fresh-registration latency is claimed.

## Speech and cloning diagnostics

All five evaluated G128 variants finish all sixteen original-suite utterances with finite output and no truncation; WavLM ranks the intended reference first for all samples. Evaluation uses the same cached Whisper-small CUDA FP16/beam-five and WavLM versions as prior results. Chinese CER uses OpenCC, number and punctuation normalization; English WER uses the existing English normalizer.

| Variant | Normalized Chinese CER | English WER | Mean speaker cosine |
|---|---:|---:|---:|
| Selected G32, original suite | 5.37% | 0% | 0.9143 |
| G128 max scales, A128, Triton 3.7 | 5.83% | 0% | 0.9148 |
| G128 max scales, A32, Triton 3.7 | 9.63% | 0% | 0.9177 |
| G128 weighted scales, A128, Triton 3.7 | 5.37% | 0% | 0.9156 |
| G128 weighted scales, A32, Triton 3.7 | 7.39% | 0% | 0.9115 |
| G128 weighted scales, A32, Triton 3.8 | 3.80% | 0% | 0.9119 |

Correction: an earlier progress message compared **raw** G128/A32 CER of 19.01% against **normalized** G32 CER of 5.37%. That comparison was invalid and was explicitly corrected. Comparable G128/A32 max-scale CER is 9.63%. Both raw and normalized transcripts/scores are preserved; normalization did not rerun ASR. The max-scale Triton 3.8 timing variant has no separate original-suite quality result and must not inherit the weighted-scale variant's score.

This sixteen-sample suite has been reused across many experiments. It cannot establish general quality equivalence, and speaker cosine is an uncalibrated proxy, not human listening. A separately frozen set of sixteen new texts, spoken by the same four references, adds 32 utterances per variant. `quality_prompts_expansion_v1.json` is fixed before evaluation; its hash and seed base 8400 are recorded in every manifest. This expansion still does not test unseen speakers or constitute a production acceptance study.

| Variant on the additional 32 utterances | Normalized Chinese CER | English WER | Mean speaker cosine |
|---|---:|---:|---:|
| Custom BF16, Torch 2.13 | 2.79% | 0% | 0.9199 |
| Selected G32, with head-prefix graphs | **1.18%** | **0%** | 0.9149 |
| G128 weighted scales, A32, Triton 3.8 | 4.89% | 0.30% | 0.9270 |

All 96 generated utterances are finite, finish without truncation, and rank the intended speaker first among the four references. Despite the candidate's favorable original-suite scores, the additional texts expose a regression. In particular, its `zh0_6` transcript contains a spurious tail and scores 30.30% CER. Without listening, this does not distinguish a synthesis defect from ASR behavior, but it fails the diagnostic comparison. **The full G128/Triton 3.8 candidate is not selected.** Raw and normalized evaluation is in `quality_suite/evaluation_expanded_v1*.json`.

This finding motivates a separate, narrower experiment: keep the selected G32 QKV/gate-up consumer fusions and use weighted G128 only for attention output and MLP down projections. `dp4a_group128_residual.py` validates and installs these 72 projections before graph capture, retaining G32 activation grouping and the original compiler. The opt-in flags are `--g128-residual-calibration` and `--g128-residual-plan`, supplied together with the full G32 norm-projection preset.

The residual-only candidate measures **84.01 ms / p95 84.07** in five separate-process requests without head-prefix graphs. Its original-suite Chinese CER is **7.86%** (G32: 5.37%), English WER 0%, speaker cosine 0.9136. On the additional texts it scores **1.96% CER** (G32: 1.18%), English WER 0%, speaker cosine 0.9204. All 48 samples are finite, finish without truncation and rank the intended reference first. Head-prefix graphs are enabled in these quality runs. Because both Chinese diagnostic comparisons worsen, this candidate is also **not selected**. The additional texts are reused for this follow-up after seeing the full-G128 result; they are no longer an untouched evaluation set for further selection.

`benchmark_group128_residual_paired.py` isolates its latency effect using one model/process, shared BF16 prefill/codec, separately retained G32/G128 output/down buffers and independently captured graphs. It restores module buffers for eager pre-audio steps as well as graph dispatch, and reverses request order every pair. Both paths use head-prefix graphs and all 32 codebooks. A fixed-seed control WAV remains byte-identical to the preceding G32 artifact. Forty-eight private-stream tests match eager and captured candidate output, including sampled IDs, across attention capacity boundaries and three head-prefix sizes.

Ten measured pairs give **84.84 → 83.76 ms median TTFA**, p95 **85.19 → 84.50**, and **1.013 ms median paired gain**. Every pair improves, by 0.387–1.741 ms. Median initial decode step falls 2.1625 → 2.1260 ms; prefill stays 9.217/9.218 ms and codec 4.773/4.772 ms. Different quantized models produce different PCM, so cross-model PCM equality is neither expected nor claimed. This establishes a small latency gain, not quality acceptance. See `group128_residual_paired_v1.json` and `quality_suite/evaluation_g128_residual_v1_normalized.json`.

## Remaining bottleneck

In the weighted G128/A32 Triton 3.8 profile, fused norm/projection kernels take 32.630 ms and remaining projections 15.724 ms: **58.61% of 82.507 ms profiled GPU time**. Matched-shape G32 profiling records 32.648 and 17.445 ms, respectively. Native attention is nearly unchanged at 4.638/4.641 ms. This is profile attribution, not end-to-end TTFA or a proof of an absolute latency limit. Weight traffic and projection execution remain the principal targets; this pass provides no basis for reducing codebooks.

## Reproduction and artifacts

Run GPU jobs sequentially. Existing model snapshots and `calibration_v1` inputs are required. Retain unique tags to preserve measurements.

```bash
/venv/moss-vllm/bin/python -m optimization.calibrate_gptq \
  --group 128 --damping .1 --scale-search diagonal --tag gptq_v1_g128_diag_d10
/venv/moss-triton38/bin/python -m optimization.benchmark_quantization \
  --mode none --calibration optimization/results/gptq_v1_g128_diag_d10 \
  --calibrated-backend dp4a --tag g128_reproduce \
  --attention-block 32 --fused-residual --dp4a-reciprocal \
  --grouped-activation --fused-dp4a --dp4a-gateup --norm-layout-limit 8 \
  --prefill-buckets 128 160 256 512 --decode-buckets --native-attention \
  --g128-plan optimization/results/group128_a32_triton38_plan_v1.json
```

The quality command accepts the same preset arguments, omitting benchmark-only `--runs`, and optionally `--prompt-file optimization/quality_prompts_expansion_v1.json --seed-base 8400`. Use `/venv/main/bin/python -m optimization.quality_evaluate --asr-device cuda --tags <tags> --output <name>` followed by `/venv/main/bin/python -m optimization.normalize_quality_scores --input <name>` for matched evaluation. Original-suite artifacts are `quality_suite/evaluation_g128_v1_normalized.json` and `quality_suite/evaluation_g128_diag_v1_normalized.json`; `group128_pass_summary_v1.json` indexes timings and sweep counts. Reproduce the residual-only paired comparison with `/venv/moss-vllm/bin/python -m optimization.benchmark_group128_residual_paired --tag reproduce --pairs 10`.

`group128_sources.tar.gz` and `group128_source_hashes.json` preserve the code, documentation, frozen additional prompts and selected plans, separately from calibration tensors and measured outputs. The new archive explicitly includes the extensionless `vendor/GPTQ_LICENSE`; the prior source archive's suffix filter omitted it. Historical archives remain unchanged.

## Sources informing the experiments

- [GPTQ quantizer implementation](https://github.com/IST-DASLab/gptq/blob/main/quant.py) motivates optional clipping-range search. Our BF16 static reconstruction and diagonal objective are described above.
- [AWQ](https://arxiv.org/abs/2306.00978) motivates considering activation importance; this implementation is not a reproduction of full AWQ.
- [Triton releases](https://github.com/triton-lang/triton/releases) and [integer dot documentation](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) inform compiler and integer-kernel experiments. Triton 3.8 is tested in isolation with separate numerical checks.
- [FireQ](https://arxiv.org/abs/2505.20839) provides a potential INT4/FP8 activation direction; its methods are not implemented here and its results are not measurements of this checkpoint.
- The [June MOSS-TTS Local v1.5 optimization article](https://www.lmsys.org/blog/2026-06-17-moss-tts-local-v15) concerns a different architecture/checkpoint. Its 12-codebook setup is not substituted for the requested 32-codebook model.
