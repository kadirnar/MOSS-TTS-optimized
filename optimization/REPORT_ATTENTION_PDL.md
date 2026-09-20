# CUDA dependency overlap through attention

The optional `--attention-pdl` path preserves all 32 codebooks and improves complete cloned-voice streaming latency. It extends the already qualified projection PDL path through Q/K normalization and rotary cache updates, native split attention, and attention reduction/activation quantization. Calibrated G32 weights, BF16 prefill, the FP32 codec, sampling, and emitted frame contents are unchanged. **The 50 ms target remains unmet.**

The selected implementation uses the simpler `q1_a2_r1` configuration. Q/K and reduction issue their launch hints immediately after waiting for their producers; native attention issues its hint after computing attention logits. The separate Q/K norm-preload candidate remains available to the experiment harness, but does not show a consistent additional request-level gain across runtime phases. Default flags remain off, and existing supervisor services are unchanged.

## Dependency and memory contract

The [NVIDIA CUTLASS dependency guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/dependent_kernel_launch.md) requires both a compatible kernel synchronization protocol and the dependent-launch attribute. Every changed consumer waits before reading producer outputs. No kernel relies on concurrent progress or spins on application flags. The native CUDA launcher uses `cudaLaunchKernelEx` with programmatic stream serialization; Triton uses its installed `gdc_wait`, `gdc_launch_dependents`, and `launch_pdl=True` support. CUDA's [PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) documents the memory visibility and opportunistic scheduling rules.

`attention_pdl.enable()` requires SM90, the qualified projection PDL preset, native B32/W4 attention, G32 output quantization, and installation before graph capture. Configuration belongs to each model/attention module. Both eager and captured decode paths use it. Position and context capacity remain dynamic through the existing bounded graphs and 1024-token fallback.

The preload experiment moves only immutable Q/K normalization weights before the wait. QKV activations, position, current-frame rotary values, and cache data remain after it. The selected variant performs no such Q/K preloads. Existing output/down projection scale preloads remain enabled for every control and candidate.

## Operator screening and correctness

The five-kernel chain is norm/QKV projection → QK/RoPE/cache → split attention → reduction/G32 quantization → output projection. A 25-option sweep includes the established control, an ordinary-launch clone, sixteen attention/reduction trigger pairs, four Q/K preload/trigger combinations, and three individually enabled stages.

The ring uses all 36 layers' actual calibrated projection weights, norm weights, and saved normalization inputs. Q/K norm weights come from the pinned original checkpoint. Representative frozen KV/rotary fixtures are taken from the nearest captured layer 0/17/35. These are isolated chains, not a complete model trajectory. Full-model measurements below test the actual intervening dependencies.

All **2,592** all-layer/intermediate comparisons are exact, as are the pilot's 72 comparisons. Six rotating/reversed graph-event timing rounds give:

| Configuration | Median chain µs |
|---|---:|
| Qualified projection-PDL control | 22.848 |
| Ordinary-launch clone | 22.842 |
| Selected `q1_a2_r1` | 19.626 |
| Q/K preload candidate | 19.568 |
| Native-attention-only PDL | 23.520 |

Enabling just one stage is not sufficient to reproduce the combined gain. All options, including regressions, are retained. The microsecond difference is not presented as an equivalent TTFA improvement.

CUDA graph edge inspection confirms four programmatic edges for the complete candidates, versus three ordinary edges and one existing projection-PDL edge for the control. A separate validator covers **168 cases per sanitizer run**: two candidates, three actual layers, four capacities, and seven changing input/position replays. It includes positions 0/1/31/32, capacity boundaries, return to zero, real/zero/spike/random inputs, NaN-poisoned future cache entries, guarded allocations, independent reference caches, and private streams. Complete chain intermediates and both entire KV caches match. **Memcheck reports zero errors; racecheck reports zero hazards, errors, or warnings.**

## Complete cloned-voice measurements

Three independent processes compare one shared model with separate graph sets for control, selected late attention hint, and the Q/K preload candidate. Each process runs fifteen rotating/reversed triplets and excludes one warmup triplet. Text is complete at request start; the cloned voice uses cached 3.112-second reference conditioning and a 145-token prompt. Text processing, codec decode, and PCM CPU transfer are included. Registration and HTTP are measured separately.

| Run | Control median ms | Selected median ms | Selected p95 ms | Median paired gain ms | Faster pairs |
|---|---:|---:|---:|---:|---:|
| `v2` | 83.642 | 79.707 | 80.487 | 4.002 | 15/15 |
| `v3`, runtime transition | 83.414 | 77.394 | 79.836 | 3.612 | 15/15 |
| `v4`, later runtime phase | 79.374 | 77.497 | 80.055 | 1.860 | 14/15 |

The second process changes timing phase during round seven. Unchanged prefill shifts from about 9.21 to 8.84 ms, and the unchanged codec also becomes faster. The cause is undetermined. All rounds are retained; the difference between aggregate medians in that transitional run is **not** attributed wholly to attention PDL. The third process confirms a smaller but positive paired improvement in the later phase. Its one slower selected request, 85.157 ms, is retained in p95 and the 14/15 result. Before/after clock observations do not establish what caused the transition.

Complete repeated decode graph medians are **2.0959 → 1.9706 ms**, **2.0964 → 1.9709 ms**, and **1.9814 → 1.9210 ms**. Median paired graph gains are 0.1256, 0.1254, and 0.0603 ms per step. These independently support reduced decode cost while showing the runtime-phase dependence.

The alternative preload medians are 79.764 / 79.476 / 77.387 ms, with median paired gains over the selected variant of −0.026 / +0.039 / +0.090 ms. This is not a stable additional improvement across phases, so serving selects the simpler variant. Both configurations remain exact in the tests.

All **135 measured complete float32 PCM streams** match their triplet controls; all excluded warmup streams match too. No request truncates. **288 complete-graph logits and sampled-ID comparisons** pass across context boundaries, fallback capacity, and audio-head prefixes. Every emitted codec frame retains all 32 channels.

One initial harness launch (`v1`) was terminated before measurement because the sanitizer process had not yet exited. It produced no paired timing or audio result and is excluded. Its log and an explicit aborted-run record are preserved. Completed latency measurements run sequentially without concurrent GPU experiments.

## Voice quality and code generation

All **48 original/additional bilingual cloning WAVs**, generation metadata, and **eight reference WAVs** are byte-identical to the preceding qualified projection-PDL path. This reuses the preceding normalized Chinese CER **5.37% / 1.18%**, English WER **0% / 0%**, and speaker diagnostics on identical audio; ASR is not rerun. These are small four-voice diagnostic suites, not broad or human quality acceptance. Exactness here is relative to the selected calibrated G32 path, not original BF16 output.

SASS inspection finds no additional register pressure in the selected Q/K or reduction kernels: Q/K retains 30 registers and 16 shared bytes; reduction uses 26/25/31/32 registers for capacities 128/256/512/1024 and 2048 shared bytes. Triton reports zero spills. Native attention retains 40 registers, zero stack/spill loads/stores, and 2080 bytes of user shared memory in the `ptxas` log. The separate `cuobjdump` resource report prints 3104 shared bytes; both raw reports are preserved rather than conflating their accounting.

The Q/K preload variant has four static global-load instructions before the wait and 26 registers; the selected variant has none before its wait. Native arithmetic and reduction ordering remain unchanged. The SM90 wait/hint lowerings are inspected in the saved SASS alongside the graph edges.

## Remaining bottleneck

PDL kernel durations overlap and include waits, so their sums are not critical-path percentages. Three profiled five-kernel replays have control/selected/preload interval unions **48.45 / 46.72 / 44.48 µs**, while their summed durations are **48.77 / 71.87 / 69.57 µs**. These are tiny profiled chains, not performance-selection measurements.

A separate trace of the selected complete 34-step request contains 14,250 kernel events. Sweeping resident intervals yields **39.17 ms with only projection kernels resident**, **10.68 ms with projection and attention both resident**, and **6.18 ms with only attention resident**. Other kernels occupy 24.68 ms. This still points to projection work as the largest remaining area to investigate. Resident intervals include waits and are not hardware utilization or pure arithmetic time; profiler gaps are not assigned a cause. The unchanged first-frame codec remains about 4.4–4.8 ms across the observed runtime phases. No absolute 50 ms lower bound has been established, and no codebook reduction is justified by these results.

## HTTP streaming and independent regression

Sequential temporary loopback servers measure **87.353 → 83.613 ms median HTTP TTFA**, with p95 **88.103 → 83.981 ms**, over twenty complete cached-voice requests per variant. All twenty PCM hashes and frame counts match. Input validation passes, and cancellation/recovery returns `429` then `200` for both servers. Both temporary processes are stopped; original service PIDs and voice caches are unchanged.

Internal engine medians are **83.812 → 79.841 ms**, while initial 32-step mean decode latency is **2.1276 → 2.0039 ms**. Preparation is 1.099 / 1.090 ms, prefill 9.219 / 9.224 ms, and first codec decode 4.766 / 4.765 ms. The unchanged stages support attributing the main improvement here to decode. Sequential HTTP still includes scheduling variation, so its full difference is not a kernel-only estimate.

One continuous fresh-registration-plus-synthesis observation is **125.28 / 124.76 ms**, including WAV parsing, reference encoding and two loopback requests. Registration alone is **34.12 / 37.39 ms**. Client WAV/base64 construction precedes timing. These are single observations, not a fresh-reference distribution or a reliable gain estimate. Cached and fresh cloning both remain above 50 ms.

The initial HTTP comparison's method string accidentally named the alternative Q/K preload candidate. It was corrected to the executed server defaults, `q1/a2/r1/preload=False`; the original label and harness source are retained. No timing/audio data or executed kernel was changed by that metadata correction.

The independent benchmark CLI, which omits audio-head buckets, measures **80.538 ms median / 80.660 ms p95** across five requests. Its full 36-step teacher-forced diagnostic dictionary and saved WAV are identical to the preceding projection-PDL CLI result. This checks integration and output regression; it is not a separately paired gain estimate.

## Reproduction

Use `/venv/moss-vllm`, fresh result tags, and sequential GPU execution:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_pdl --tag reproduce --layers 36 --rounds 6
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_attention_pdl --tag reproduce_memcheck
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_attention_pdl --tag reproduce_racecheck
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_pdl_paired --tag reproduce --rounds 15
/venv/moss-vllm/bin/python -m optimization.audit_attention_pdl --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_pdl_http --tag reproduce
```

Append `--attention-pdl` to the qualified `--projection-pdl` benchmark, quality-generation, or loopback-server commands. The flag requires `--projection-pdl --native-attention --attention-quant`; the projection preset in turn requires exact short scales. Persistent services should use supervisor as documented in the instance guide.

Evidence is indexed by `attention_pdl_*`, `quality_suite/attention_pdl_*`, `quality_suite/expanded_attention_pdl_v1`, and `http_attention_pdl_*`. The pass index, static checks, source archive, and hashes are `attention_pdl_pass_summary_v1.json`, `attention_pdl_static_checks_v1.json`, `attention_pdl_sources.tar.gz`, and `attention_pdl_source_hashes.json`.
