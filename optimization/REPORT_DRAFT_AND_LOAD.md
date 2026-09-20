# Layer-skipping drafts and CUDA load-policy experiments

The selected 32-codebook streaming/cloning path remains unchanged. This pass finds no useful new request-level speedup: untrained layer-skipping drafts have insufficient joint acceptance, and a promising isolated QKV load-policy result mostly disappears under representative model traffic. The 50 ms goal remains active and unmet. No reduced-codebook mode, replacement checkpoint or new service deployment is introduced.

## Draft acceptance measured on this checkpoint

`audit_layer_drafts.py` generates eight first-frame histories with the selected calibrated G32 model: two texts for each of four supplied reference voices. Each history covers 32 autoregressive decode steps. Six layer-skipping layouts then prefill and decode with their **own** retained layers and independent KV caches, consuming the target-generated history. This does not provide the draft with the target's hidden/KV states.

The audit reconstructs the existing temperature/top-k/top-p distributions and includes every active audio channel plus the unforced text slot in a joint proposal. For each of 256 contexts per layout, it estimates `E_q[min(1, p_joint/q_joint)]` with 8,192 Monte Carlo proposals. It also records the support-intersection upper-bound formula and product-of-marginal-overlaps lower-bound formula, computed in floating point. Monte Carlo values have sampling error and can exceed a computed bound slightly. The identical full-model control reproduces all 256 probability tensors exactly and has acceptance one; independent identical/disjoint distribution checks also pass.

| Retained layers | Repeated decode graph ms | Mean joint acceptance estimate | Mean support upper bound |
|---|---:|---:|---:|
| 36, full control | 2.1302 | 100% | approximately 100% |
| 34, evenly spaced | 2.0247 | 8.456% | 9.887% |
| 30, evenly spaced | 1.8162 | 4.305% | 5.181% |
| 24, evenly spaced | 1.5077 | 0.207% | 0.241% |
| 18, evenly spaced | 1.1943 | 0.076% | 0.074% |
| 12, evenly spaced | 0.8826 | 0.138% | 0.169% |
| 24, middle twelve omitted | 1.5054 | 0.473% | 0.539% |

The 34-layer draft's mean estimated acceptance falls from 53.54% over the first four steps to 13.79% over the next four and 0.16% over steps 8–15. No accepted proposals were observed in its Monte Carlo samples for steps 16–31. That last observation is not proof of exactly zero probability. Truncation to top-k/top-p makes support overlap especially restrictive when many codebooks are active together.

Even an optimistic one-proposal cost proxy—verification costs only one ordinary target decode, one bonus token is free, and draft prefill/bookkeeping are excluded—gives speed ratios of 0.56–0.71 for these six drafts. This is a screening calculation, **not measured speculative throughput**. Teacher-forced acceptance on eight histories does not establish block acceptance under free-running proposals, and it says nothing definitive about a trained drafter or different layer selection. No block verifier, speculative serving path or 50 ms impossibility claim follows from this experiment.

Artifacts: `layer_draft_audit_v1.json`, `layer_draft_histories_v1.pt`, and `layer_draft_summary_v1.json`. All 1,792 layout/context probability comparisons are retained. This experiment uses Torch 2.13, Triton 3.7.1, the existing G32 kernels and native CUDA attention; it makes no new codec or HTTP timing claim.

## CUDA load-policy implementation and sweep

`dp4a_norm_memory.py` copies the selected fused normalization/QKV and gate/up arithmetic into an experimental kernel with controls for weight/scale cache operators, L1 eviction priority, explicit PTX L2 prefetch, CTA ordering and register limits. Every choice retains the calibrated weight codes and all 32 audio channels. The final benchmark/quality opt-in `--qkv-load-policy` chooses early L1 eviction for weight loads and L2-only scale loads in QKV. It is **not a serving preset**.

Two sweeps measure 152 configuration attempts, using layers zero and seventeen for timing. Sixteen attempts use cache/eviction combinations rejected by the installed PTX toolchain; these are now rejected by the Python guard. The remaining 136 attempts complete 1,224 numerical checks; 1,134 are exact. The initial ninety failed checks come from ten gate/up CTA-order variants: output values used the remapped block index but activation scales still used the original index. The scale-index bug was fixed, and every corresponding check passes in the second sweep. Failed first-sweep results and compiled IR remain available. No such variant was deployed.

Explicit L2 prefetch is slower. Tight register limits introduce spills and large regressions; QKV at a 32-register cap takes about 21 microseconds, versus roughly 8 at the selected limit. The corrected gate/up block-order variants offer no useful gain. Cache policy leaves the chosen QKV kernel at 56 registers, zero spills and 2,048 shared bytes.

The first timing ring rotates eight weight/scale copies while reusing one layer's normalization inputs. It shows QKV improving **8.204 → 7.385 microseconds**, with all nine sampled outputs exact. A twelve-round comparison with reversed/rotating order reproduces **8.139 → 7.392 microseconds**, or 0.745 microseconds median paired gain. Early eviction alone gives nearly the same result; L2 scale loads add no clear incremental gain. Layer-seventeen timing still shows a similar isolated improvement, ruling out the missing residual addition in layer zero as its sole explanation.

The stronger test rotates **all 36 actual QKV weight matrices, scales, normalization weights, input states and residuals**. Under that traffic, the medians are **8.688 → 8.658 microseconds**, and median paired gain is just **0.034 microseconds**. The initial shared normalization buffers made the microbenchmark unrepresentative of model execution. This comparison does not isolate which cache or buffer causes the difference, but it changes how future producer-fusion candidates should be screened.

Artifacts: `norm_memory_v1.json`, `norm_memory_layer17_v2.json`, `norm_memory_repeat_v1.json`, `norm_memory_layer_ring_v1.json`, and per-configuration PTX/TTGIR files. The selected policy matches all **1,656** all-layer real/zero/spike input checks, including producer intermediates, on private streams. Twelve padded/actual-size private-stream CUDA graph cases pass both memcheck and racecheck with zero errors/hazards.

## Complete streaming checks

The paired test retains separate full decode/context/head-prefix graphs and switches the load flag for eager pre-audio work as well as graph dispatch. Both paths use the same G32 weights, BF16 prefill, FP32 codec, 145-token Chinese prompt and cached voice. Order reverses every pair; one warmup pair is excluded.

| Same-process comparison | Control median ms | Candidate median ms | Median paired gain ms | Faster pairs |
|---|---:|---:|---:|---:|
| First ten pairs | 85.75 | 85.70 | 0.110 | 6/10 |
| Follow-up twenty pairs | 84.66 | 84.59 | 0.134 | 12/20 |

These results do not establish a worthwhile request-level optimization. Twelve additional alternating full-graph GPU timings give 2.12344/2.11921 ms medians and **0.00366 ms median paired gain**, consistent with only a small total effect. The twenty-pair p95 values are 88.56/87.70 ms. All thirty paired full float32 PCM streams are identical between paths; neither side truncates. Each run also passes 48 private-stream full-graph comparisons across context boundaries and three active-head-prefix sizes, including all logits and sampled IDs.

A late speed shift affects **both paths and unchanged stages** in the twenty-pair run. Prefill falls from approximately 9.22 to 8.83 ms and first codec decode from 4.77 to 4.40 ms. The 79.95 ms fastest candidate observation is not selected as a new latency result. The cause is unknown; no clock, affinity or service change was made. All twenty pairs remain in the main statistics; `qkv_load_timing_audit_v1.json` records the descriptive early/late split.

The separate five-request benchmark without head prefixes measures 86.77 ms median. Its previous 36-step diagnostic dictionary and saved WAV match exactly. Both the original sixteen-utterance suite and the additional thirty-two-utterance suite, plus all eight saved reference WAV files, are byte-identical to their G32 controls. Existing normalized CER/WER scores therefore apply to the identical artifacts without rerunning ASR: 5.37%/0% on the original suite, 1.18%/0% on the additional texts. This is not equality with the original BF16 checkpoint or a broad/human quality guarantee.

The experimental server flag and unrun HTTP harness were removed after the full-model timing decision. `server.py` matches the preceding source snapshot exactly. No new HTTP, cancellation or fresh-reference latency result is claimed. Existing supervisor services and voice caches are unchanged.

## Remaining work and reproduction

The latest profile still assigns approximately **59.5% of GPU time** to fused norm/projection consumers plus remaining projections. Neither these cache-policy changes nor the tested untrained drafts remove that bottleneck. Larger arithmetic/weight-traffic changes, or a drafter trained for the joint acoustic distribution, remain separate investigations. Every next path must continue using all 32 codebooks and measure complete playable PCM latency.

Run GPU jobs sequentially from the repository root:

```bash
/venv/moss-vllm/bin/python -m optimization.audit_layer_drafts --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_norm_memory --tag reproduce --timing-layer 17
/venv/moss-vllm/bin/python -m optimization.validate_norm_memory layer_ring --tag reproduce
/venv/moss-vllm/bin/python -m optimization.validate_norm_memory all --tag reproduce
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_norm_memory memory --tag reproduce_memcheck
/venv/moss-vllm/bin/python -m optimization.benchmark_qkv_load_paired --tag reproduce --pairs 20
```

The existing model snapshots, G32 calibration export, raw normalization capture, fixture and expanded-suite manifest/reference files are required. `--qkv-load-policy` can be added to the G32 norm-projection benchmark/quality commands in `REPORT_NORM_PROJECTION.md`; it remains experimental. `draft_load_sources.tar.gz` and `draft_load_source_hashes.json` archive code/docs/plans and licenses separately from weights, audio and measured outputs.

The [speech coarse-grained acceptance paper](https://machinelearning.apple.com/research/coarse-grained), [Approximate Speculative Decoding](https://arxiv.org/abs/2608.03447), and [Carryover Drafting](https://arxiv.org/abs/2609.14717) motivate measuring acceptance and distinguish trained or approximate methods from the untrained exact-joint audit here. None of their reported speedups is a measurement of this checkpoint. Cache/prefetch implementation follows the [Triton load API](https://triton-lang.org/main/python-api/generated/triton.language.load.html) and [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch); only instructions accepted by the installed SM90 toolchain are used.
