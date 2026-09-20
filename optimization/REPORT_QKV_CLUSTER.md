# Clustered QKV and attention preparation

The corrected SM90 cluster kernel measures **74.144 ms median warm cached-voice TTFA**, compared with **74.227 ms** for the preceding selected register-preload preset. The median paired gain is **0.106 ms**, with 16/20 pairs faster. This is a small optional improvement, not a result near 50 ms. Every first PCM chunk retains all **32 acoustic codebooks**. The model remains calibrated G32 decode, BF16 prefill, FP32 codec, and streaming voice cloning.

The measurement starts when complete text is submitted to the engine and ends at the first playable PCM chunk. It includes text preparation, prefill, initial generation and codec decoding, and excludes network transit and reference registration. Fresh-reference and HTTP observations are reported separately below. The preceding selected model's quality qualification is the comparison target; bit equality here does not claim equality to the original BF16 model.

## Implementation

`qkv_cluster_prepare.py` assigns one cluster to each of 48 Q/K/V heads. Eight four-warp CTAs divide the 128 projection rows. Q/K values are exchanged through distributed shared memory before head normalization, rotary position handling and KV writes; V values are written directly. This replaces the separate normalization/QKV and attention-preparation kernels with one launch per layer. The native attention, attention reduction/quantization and register-preloaded output projection remain separate.

The selected schedule hints a 1/16 weight prefix before its original dependency wait and releases dependent work after projection, before head preparation. Every consumer still waits before reading producer data. Cluster dimensions are `(8,1,1)` and scheduling preference is SPREAD, matching the compiler runtime. Production and diagnostic kernels use **64 registers per thread, 2 KiB shared memory per CTA and zero spills**. The original head-width, residual rounding boundaries, activation groups, scales and all acoustic codebooks are preserved.

Triton 3.7.1 cannot lower the tested distributed-shared-memory conversion: its `nvvm.mapa` result uses address space 3 where the verifier requires generic or cluster space. Only the new fused kernel is compiled in the already isolated Triton 3.8 environment. Its SM90 cubins are exported with pointer signatures, specializations, resource metadata, hashes and source snapshots. `qkv_cluster_binary.py` loads these through the CUDA driver in the selected Torch 2.13/Triton 3.7.1 runtime. Other model and codec kernels keep their selected compiler/runtime. This is not a vLLM or whole-environment upgrade.

`qkv_cluster_model.py` installs a per-model decode callable before graph capture. Prefill delegates to the original method. The optional `--qkv-cluster` flag in serving, quality generation and benchmarking requires the selected bulk-prefetch and register-preload preset. It defaults off and uses `results/qkv_cluster_bundle_v6`. No module-global dispatch is changed and no running supervisor service is replaced.

## Numerical failures and corrections

These failures are retained rather than treated as successful trials:

- Initial CGA layout construction and SSA type mismatches were corrected before compilation. The selected compiler then exposed the `nvvm.mapa` lowering failure. The first binary export also assumed a removed `cluster_dims` metadata field; export now records the runtime's fixed dimensions explicitly.
- An initial 52-configuration screen inside Triton 3.8 passed 5,616 comparisons and found a 0.511 µs isolated chain gain. Transferring the binaries to the selected runtime exposed a one-byte QKV difference at layer 2, column 1604. Saved normalization/activation quantization outputs matched. Emitted PTX showed that Triton 3.8 balanced a four-product local sum that Triton 3.7 folded sequentially. Explicit local FMA ordering corrected this difference.
- That correction passed all-layer fixtures and 48 full-model graph checks, but failed the first complete-stream PCM comparison. A request-trajectory audit stopped at layer 30 on the fifth decode call, after 174 exact projections. Triton 3.8 also balanced the normalization's 32-value local sum. The new kernel now explicitly reproduces the selected local FMA chain before the warp/four-warp reductions. The failing input is saved in `qkv_trajectory_v1/mismatch.pt` and included in regression checks.
- The first memcheck run reported zero memory errors but failed a zero-input cache comparison. QKV and V matched, while 416 Q and 104 K elements differed only in zero sign. The selected decode PTX uses `sub.rn.f32(0,xs)`; a sign-bit flip has different signed-zero behavior. Bundle v6 reproduces that subtraction exactly.
- Fresh installation initially assumed that context-bucket setup had already created `_decode_capacity`. The integrated warmup test exposed this before generation. The callable now uses physical cache capacity when that optional attribute is absent. The failed quality-start log is retained.

`qkv_cluster_compiler_diagnostics_v1.json` indexes the arithmetic evidence. Earlier nonidentical binaries are not selected. Matching frozen fixtures alone was insufficient; complete trajectories and changed-input tests found additional failures.

## Screens and full requests

The initial screen covers 1/2/4/8 CTAs, integer layouts, prefix hints and dependency-release points. Later screens compare corrected arithmetic, leader-only head preparation, four/eight CTAs, and 1/8 versus 1/16 versus 1/32 hints. Leader-only preparation and the larger/smaller prefixes do not improve the selected full-request result.

The corrected, balanced-order all-layer ring (`qkv_cluster_binary_ring_v6.json`) compares actual weights and three saved normalization inputs from each of 36 layers. It passes **216** complete chain/intermediate/cache comparisons and private graph checks. Control and fused medians are **19.063 / 18.849 µs**; median paired gain is **0.205 µs**, all six rounds faster. The attention fixtures are the nearest saved layer 0/17/35 captures, not an actual complete trajectory. Early rounds run in a faster runtime phase in both modes. All samples are retained. Earlier two-mode rings accidentally canceled reversal with rotation and always ran control first; the final ring and corrected full-request comparisons use a balanced order.

| Complete-request run | Control median | Fused median | Median paired gain | Faster pairs |
|---|---:|---:|---:|---:|
| v2, explicit projection and norm order | 74.275 ms | 74.086 ms | 0.237 ms | 9/12 |
| v3, nominal prefix | 74.261 ms | 74.083 ms | 0.128 ms | 15/20 |
| v3, larger 1/8 prefix | 74.261 ms | 74.201 ms | 0.101 ms | 12/20 |
| **v4, corrected zero semantics, selected** | **74.227 ms** | **74.144 ms** | **0.106 ms** | **16/20** |

The final p95 is **74.850 → 74.719 ms**. The first 32 LLM steps measure **61.208 → 61.080 ms**; prefill is **6.826 / 6.830 ms** and the first codec chunk **4.524 / 4.528 ms**. Those unchanged-stage differences are runtime variation, not codec gains. Across v2/v3/v4, all **124 measured complete streams** and final RNG states agree; all **52** controls match preceding selected PCM hashes. Each candidate has independent context/head/initial-audio graph sets and restored eager dispatch. The final binary passes **48** full-model graph comparisons of IDs, logits, sampling state and all 72 entire KV buffers after poisoning the current positions. Failed v1 is not counted as a complete-stream pass.

## Memory and synchronization validation

Bundle v6 passes **72 cases under each of memcheck, racecheck and synccheck**. Each suite covers production and diagnostic cubins, real/zero/spike activation changes, private CUDA graphs, positions 0/127/1023, three frozen layers and the saved layer-30 divergence. Complete output bytes and entire poisoned KV caches match the selected path. Memcheck and synccheck each report zero errors; racecheck reports zero hazards, errors and warnings. These are fused-chain checks, not whole-service sanitizer coverage. The initial signed-zero failure remains in `qkv_cluster_memcheck_v1.log`.

## Cloning, HTTP and entry points

The integrated flag generates all **48** original/expanded Chinese and English cloning WAVs. Every generated WAV, eight reference copies and generation metadata match the preceding selected register-preload suites exactly. The output folders are `quality_suite/qkv_cluster_v2` and `quality_suite/expanded_qkv_cluster_v2`; `qkv_cluster_quality_exact_v2.json` contains hashes. Existing identical-audio normalized Chinese CER **5.37% / 1.18%**, English WER **0% / 0%**, and speaker cosine **0.9143 / 0.9149** therefore apply. No ASR rerun or human listening is claimed; four reference assets remain a small diagnostic set.

Sequential temporary loopback servers measure **77.816 → 77.716 ms median HTTP TTFA**. The p95 **regresses from 78.141 to 78.600 ms**. All twenty complete PCM streams match; missing-voice, malformed-budget and over-capacity errors pass, and cancellation recovers through the expected **429 → 200** sequence. Internal engine medians are **74.141 / 74.018 ms**, with initial LLM intervals **61.192 / 61.040 ms**. Separate server processes include host/runtime variation; this is integration evidence alongside the in-process paired result, not a broad tail-latency improvement.

One fresh-registration-plus-synthesis observation is **116.811 / 112.970 ms**. Reference-registration HTTP time itself changes **35.604 → 32.458 ms**, despite an unchanged encoder. This host/runtime variation explains most of the fresh-reference difference; it is not attributed to QKV fusion. Both temporary servers stop, port 18084 is free, and the original supervisor PIDs 54030 / 56179 remain ready with 32 codebooks.

The independent ordinary benchmark CLI, without audio-head buckets, completes five requests at **74.791 ms median / 74.956 ms p95**. Its entire 36-step diagnostic dictionary and saved WAV match the preceding selected run. This separate-process result is slower than the prior 72.768-ms CLI observation and is not a speedup claim. All three entry-point help checks pass; all 273 top-level Python modules parse. The four selected cubins are SHA-verified and retain paired cluster barriers, cluster reads, a bulk hint, dependency wait/release and explicit rotary subtraction.

## Remaining work on the critical path

The final post-timing 34-step profile has **12,101 kernel events**, removing 1,188 separate head-preparation launches from the prior profile's 13,289 events. Each of the 1,188 fused preparations has a median resident interval of **9.568 µs**. Gate/up and down still have the largest decoder projection intervals: **19.008 / 25.504 µs** median, with summed intervals **22.850 / 30.617 ms**. These intervals include dependency waits and overlap; they are not utilization, additive critical-path percentages or a lower bound.

Initial LLM generation remains the dominant measured stage at about 61.08 ms. Further cluster scheduling/resource placement and the gate/up/down projection chain are concrete hardware targets. The current measurements do not prove that 50 ms is impossible, and do not justify reducing codebooks.

## Reproduction and sources

Use `/venv/moss-vllm/bin/python` for the selected runtime and `/venv/moss-triton38/bin/python` only for export. All GPU commands are run sequentially.

```bash
/venv/moss-triton38/bin/python -m optimization.qkv_cluster_binary --tag NEW --configs c8_t2_exact
/venv/moss-vllm/bin/python -m optimization.benchmark_qkv_cluster --tag NEW --layers 36 --rounds 6 --binary-bundle optimization/results/qkv_cluster_bundle_NEW
/venv/moss-vllm/bin/python -m optimization.benchmark_qkv_cluster_paired --tag NEW --rounds 20 --configs c8_t2_exact --binary-bundle optimization/results/qkv_cluster_bundle_NEW --profile
compute-sanitizer --tool memcheck --error-exitcode 1 /venv/moss-vllm/bin/python -m optimization.validate_qkv_cluster --tag NEW_MEM --binary-bundle optimization/results/qkv_cluster_bundle_NEW
compute-sanitizer --tool racecheck --error-exitcode 1 /venv/moss-vllm/bin/python -m optimization.validate_qkv_cluster --tag NEW_RACE --binary-bundle optimization/results/qkv_cluster_bundle_NEW
compute-sanitizer --tool synccheck --error-exitcode 1 /venv/moss-vllm/bin/python -m optimization.validate_qkv_cluster --tag NEW_SYNC --binary-bundle optimization/results/qkv_cluster_bundle_NEW
```

The installed flag intentionally selects the qualified bundle v6; exporting another bundle does not silently replace it. Add `--qkv-cluster` to the complete register-preload serving/quality/benchmark command in `REPORT_ASYNC_WEIGHTS.md`.

The cluster decomposition was informed by [ClusterFusion, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/3760d0ea4709a913a4804f4b4c073836-Paper-Conference.pdf) and [Triton's multi-CTA tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/multicta.html). This implementation is independent and preserves this model's selected quantized arithmetic. Paper FP16/H100 results are not performance predictions for this H200 workload. [LLVM's NVPTX documentation](https://llvm.org/docs/NVPTXUsage.html) and the installed compiler/driver sources informed the address-space and launch-ABI audit.
