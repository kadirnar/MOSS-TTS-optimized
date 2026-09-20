# CUDA projection dependency overlap

The selected PDL candidate improves warm cached-voice TTFA to **83.46 / 83.47 ms median** in two independent fifteen-round comparisons. Median paired gains over the preceding exact short-scale path are **0.76 / 0.64 ms**; every candidate round is faster and every complete PCM stream matches. **50 ms remains unmet.** All 32 codebooks, streaming voice cloning, calibrated G32 weights, BF16 prefill and the FP32 codec are retained.

`--projection-pdl` is an optional addition to the selected `--short-scales` preset. It uses per-model/per-projection configuration and defaults off. Existing supervisor services are unchanged. This is a measured reduction in dependency overhead, not a new quantization scheme or a claim that all model operations run concurrently.

## Dependency contract and implementation

CUDA Programmatic Dependent Launch permits a consumer to start independent work before its producer retires. The consumer must wait before reading produced data; a launch hint alone does not establish memory visibility. Overlap is opportunistic. The [NVIDIA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) documents both stream capture and explicit graph edges. The installed Triton 3.7.1 launcher sets the CUDA Driver API programmatic-serialization attribute when `launch_pdl=True`.

New Gluon kernels retain the established integer dots, fixed floating layouts and BF16 boundaries. QKV/gate-up kernels wait before activation/residual loads, then issue their dependent-launch hint. Output/down kernels load only immutable BF16 weight scales before waiting; dependent activation/scaling loads remain afterward. They issue the hint before output stores. Every consumer synchronizes with the predecessor; no code depends on concurrent progress or uses a spin-wait protocol. Allocation owners and static inputs outlive their CUDA graphs.

`projection_pdl.enable()` checks SM90, the complete norm/projection preset, exact short-scale buffers and absence of the separate experimental QKV load policy, then installs flags before capture. The native/codec/sampling paths retain their existing dependency behavior. The benchmark's first comparison temporarily replaced process-local function references; the repeat uses the integrated per-model flags. The first harness source is preserved with its result.

## Trigger and preload screening

The actual-buffer ring contains each of 36 layers' normalization inputs, residuals, norm weights and quantized matrices. Each chain executes fused norm/gate-up, its dependent down projection, and the following norm/QKV projection. These are actual-shape dependency chains, not complete attention layers or full TTFA. Adjacent chains begin from distinct saved inputs; their results are not fed through omitted attention operations. The next layer's norm/weights are used, including a wrap for the last ring entry. Six timing rounds rotate/reverse order, with nine graph-event samples per option. Full-model comparisons below measure the actual intervening dependencies.

Twelve trigger configurations test implicit launch completion, hints after the initial wait, after normalization and before stores. Two controls retain ordinary dependencies, including a cloned kernel body with PDL disabled. The 36-layer sweep passes **1,404 chain/intermediate comparisons**, all exact. Best trigger placement reduces median chain time **37.967 → 36.208 µs**. Hints placed after normalization can regress to about 40.98 µs; they are not selected.

A second sweep tests sixteen combinations of independent scale and/or packed-weight preloads for norm consumers and output/down. It passes **1,944 comparisons**, all exact. Selected scale-only output/down preload measures **36.060 µs**, versus 36.195 for basic PDL and 37.873 for the ordinary control in that sweep. Preloading full matrices regresses to 54.57–66.88 µs. All configurations and both one-layer pilots are retained, including slow variants. These are screening measurements, not claims of equivalent full-model speedups.

For every captured screening graph, `cuGraphGetEdges_v2` confirms two programmatic edges with source port one/type one for PDL, versus default zero-port/type-zero edges for controls. Private-stream graph outputs match the reference. A separate changing-input validator covers three actual layers, three recorded inputs and zero/spike/random cases for both candidate variants: **36 cases**. It rewrites inputs between replays. **Memcheck reports zero errors; racecheck reports zero errors and zero warnings.**

## Complete streaming results

Both runs use one shared model with separate context/head-prefix graph sets for ordinary short scales, basic PDL and PDL plus scale preloads. Each retains its own captured dispatch, and eager settings are restored alongside graph selection. One warmup triplet is excluded per process; all fifteen measured triplets remain. The request supplies complete text and cached 3.112-second Chinese cloned-voice conditioning, with 145 prompt tokens. CPU PCM transfer and codec decoding are included; registration and HTTP are separate.

| Run | Control median ms | Basic PDL median ms | Selected preload median ms | Selected paired gain ms | Faster rounds |
|---|---:|---:|---:|---:|---:|
| First | 84.197 | 83.872 | 83.461 | 0.757 | 15/15 |
| Integrated repeat | 84.224 | 83.907 | 83.466 | 0.639 | 15/15 |

Basic PDL's median paired gains are 0.387 / 0.324 ms, with 15/15 and 14/15 faster rounds. Scale preloads add another 0.361 / 0.369 ms median paired improvement over basic PDL. Candidate p95 is **84.597 / 83.984 ms**, versus control **85.403 / 89.411 ms**. Outliers and slower phases in the controls are retained; the full p95 difference is not attributed to PDL.

All **ninety measured full float32 PCM streams** match their triplet controls and finish without truncation. Both excluded warmup triplets also match. Each run passes 96 private-stream complete-graph comparisons of logits and sampled IDs across context boundaries, fallback capacity and head prefixes: **192 exact checks** in total. Every emitted codec frame still contains all 32 channels.

All **48 original/additional cloning-suite WAVs**, generation metadata and **eight reference WAVs** remain byte-identical to the immediately preceding short-scale path. Existing normalized Chinese CER **5.37% / 1.18%**, English WER **0% / 0%** and speaker diagnostics apply to identical files without rerunning ASR. These are reused four-voice diagnostic sets, not broad or human quality acceptance; equivalence is to the calibrated G32 path, not unquantized BF16. Evidence is in `quality_suite/projection_pdl_audio_equivalence.json`.

Twelve alternating full-decode graph measurements per process give **17.13 / 16.66 µs median paired gain** for the selected candidate, with graph medians **2.09449 / 2.09481 ms** versus control **2.11085 / 2.11164 ms**. Basic PDL gains 8.73 / 10.11 µs. Preparation, prefill and first codec times remain similar; the repeated graph measurements support a small decode gain. Raw phase summaries are in `projection_pdl_timing_summary_v1.json`.

The separate five-request benchmark without head-prefix graphs measures **84.605 ms median / 84.762 ms p95** and 2.1322 ms median decode step. Its complete 36-step teacher-forced diagnostic dictionary and saved WAV match the prior short-scale benchmark exactly. This exercises CLI installation and the upstream diagnostic path independently; it is not another paired improvement estimate. See `projection_pdl_regression_comparison_v1.json`.

## Machine code and profiler interpretation

Actual cubins, PTX, Gluon IR and SASS are retained in `projection_pdl_audit_v1/`. The selected QKV/up/output/down kernels use **56 / 162 / 32 / 128 registers**, **2,048 / 1,024 / 2,048 / 1,024 shared bytes**, and **zero spills**, matching their ordinary counterparts. Output/down contain one/eight static global-load instructions before the dependency wait. In the rejected all-weight preload, gate-up reaches 255 registers and reports `n_spills=212`; QKV rises to 96 registers. Extra register pressure and spilling are consistent with its slowdown.

SM90 disassembly names the wait/hint lowering `ACQBULK` / `PREEXIT`, consistent with the [NVIDIA binary-utilities instruction reference](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-binary-utilities/index.html). The first audit parser looked for the PTX mnemonic and returned null wait positions. It was corrected using the saved SASS; the original summary remains archived and no kernel/timing was changed.

PDL changes how profiler durations must be interpreted. In three profiled chain replays, the control has 114.24 µs summed kernel duration and the same interval union. Basic PDL has **167.84 µs summed / 111.65 µs union**; selected scale preload has **162.85 / 109.54 µs**. Resident intervals overlap and include dependency waits. Their sum therefore double-counts time, and interval overlap does not prove simultaneous useful work. The new full-stream profile's inflated projection percentage must not be compared directly with the previous 59.25% non-PDL breakdown. CUDA graph-event and complete-request timings remain the performance evidence; profiling is excluded from them.

Projection arithmetic/traffic and the remaining ordinary boundaries remain the principal next targets. This pass does not establish an absolute 50 ms bound or justify reducing codebooks.

## HTTP streaming and fresh references

Two temporary loopback servers run sequentially, ordinary short-scale control then selected PDL, using the same text/reference and seeds 501–520. Twenty measured complete streams each give **88.304 → 87.586 ms median HTTP TTFA**, p95 **90.327 → 87.921 ms**. All twenty full PCM hashes and frame counts match. Input validation checks pass, and cancellation/recovery returns `429` then `200` for both servers. Both temporary servers are stopped.

Internal engine medians are **84.418 / 83.768 ms**, initial decode **2.1463 / 2.1277 ms**, prefill **9.2221 / 9.2225 ms**, first codec **4.7694 / 4.7696 ms**, preparation **1.0791 / 1.0769 ms**, and seeding **0.2282 / 0.2258 ms**. The stage result is consistent with the paired decode gain. Sequential HTTP still includes CPU/network scheduling variation; its entire p95 difference is not a kernel-only claim.

Single continuous fresh-registration-plus-synthesis observations take **130.74 / 129.81 ms**, including server WAV parsing, reference encoding and two loopback requests. Registration alone is **38.81 / 38.57 ms**. Client WAV/base64 construction precedes the timer. These are single observations, not a fresh-cloning latency distribution or repeatable gain estimate. Neither cached nor fresh-reference measurements meet 50 ms. See `http_projection_pdl_comparison_v1.json` and `http_projection_pdl_stage_summary_v1.json`.

## Reproduction and research context

Use `/venv/moss-vllm` and run GPU work sequentially with fresh tags:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_projection_pdl --tag reproduce --rounds 6
/venv/moss-vllm/bin/python -m optimization.benchmark_projection_pdl --tag preload_reproduce --rounds 6 --prefetch
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_projection_pdl --tag reproduce_memcheck
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_projection_pdl --tag reproduce_racecheck
/venv/moss-vllm/bin/python -m optimization.benchmark_projection_pdl_paired --tag reproduce --rounds 15
/venv/moss-vllm/bin/python -m optimization.audit_projection_pdl --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_projection_pdl_http --tag reproduce
```

Append `--projection-pdl` to the exact short-scale benchmark, quality or loopback-server commands in `REPORT_SHORT_SCALES.md`. The flag requires `--short-scales`; persistent services should use supervisor. Existing endpoints have not been replaced.

The [Triton PDL tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/11-programmatic-dependent-launch.py) and installed GDC/launcher source document the mechanism. Recent [PDL megakernel reconstruction](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction) distinguishes matched real-weight end-to-end and synthetic-body measurements on an H100/Llama workload; its results are not MOSS measurements. The [August Ling decode engineering note](https://staging.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell) uses independent preamble loads on Blackwell. Our preload sweep demonstrates that whole-matrix hoisting can lose badly in this H200 kernel. No third-party kernel implementation is vendored by this pass.

Raw artifacts are `projection_pdl_*`, `quality_suite/projection_pdl_*`, `quality_suite/expanded_projection_pdl_v1`, and `http_projection_pdl_*`. Source and evidence indexes are recorded in `projection_pdl_pass_summary_v1.json`, `projection_pdl_sources.tar.gz` and `projection_pdl_source_hashes.json`.
