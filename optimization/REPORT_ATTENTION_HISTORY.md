# Historical KV preloads and attention overlap

Optional **`--attention-history`** reduces warm cached-voice TTFA from **73.636 to 71.856 ms** in the twenty-pair repeat, with **1.685 ms median paired gain** and 19/20 requests faster. Paired loopback HTTP measures **76.436 → 75.022 ms**, with **1.919 ms paired gain** and all twenty pairs faster. The **50-ms goal remains unmet**.

All **32 acoustic codebooks**, streaming voice cloning, calibrated G32 decode, BF16 prefill and the FP32 codec remain enabled. This pass changes no weight precision or attention context. All 48 bilingual cloning WAVs remain byte-identical. The option defaults off; existing supervisor services are unchanged.

## What changed

The preceding register-resource sweep was progress: it ruled out replacements with only uncertain gains. All 1,565 files and its source-archive hash were revalidated before this pass. GPU jobs ran sequentially on the H200 NVL.

Native attention previously waited for the entire QKV producer before loading any cache data. The new CUDA kernel loads only **historical rows with index strictly below the current position** before the existing dependency wait. It retains the wait before reading Q or the current K/V row. Position and historical cache writes precede the QKV producer; that producer completes its own upstream dependency wait before releasing attention. Cache ownership remains sequential per request. No spin flags, omitted waits or requirement for concurrent execution are introduced.

This follows the independent-work opportunity described by [CUDA Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html). Launch overlap is opportunistic; triggering alone does not make dependent data safe. The selected schedule releases QKV dependents after its upstream wait and releases attention dependents after the attention wait. Selected reduction/G32 quantization and output-weight preloading remain unchanged.

The sweep compares key-only, value-only and combined preloads, source-level packed/FP32 staging, and two QKV/attention release positions. It includes no-preload controls. The first build used ordinary read-only loads, which the compiler moved after the wait. Explicit volatile PTX global-vector loads retain the intended order. SASS now contains four/eight historical vector loads plus a position load before the wait; current-token and query consumption remain after it. The first build supplies no performance claim.

The selected native kernel uses **98 registers, 2,080 bytes shared memory and zero reported spills**. The early-release clustered QKV production kernel uses **72 registers, 2 KiB shared and zero spills**. The host remains Torch 2.13/Triton 3.7.1; QKV cubins use isolated Triton 3.8, and native attention uses CUDA 12.8. Early-release producer cubins were freshly exported from the current exact source. An initial guard rejected an older bundle that predated the signed-zero correction; no measurement used that bundle.

`attention_history_model.enable()` installs a per-model callable before capture and requires the qualified `--gateup-compiler`, `--down-tile8`, clustered-QKV and attention-PDL preset. Serving, quality generation and the ordinary benchmark expose the flag. Original global callables remain intact. Prefill delegates to the preceding implementation.

## Operator and full-request measurements

Thirty schedules include the selected control, an early-QKV-only control and 28 native variants. The pilot passes 90 intermediate/cache comparisons and 240 private-graph boundary checks. The all-layer ring passes **3,240 comparisons**, another **240 graph checks**, and thirty graph-edge inspections. It rotates all 36 weight sets and three frozen inputs per layer with nearest saved attention metadata. These isolated chains are not full model trajectories or TTFA measurements.

| Schedule | Chain median | Median paired gain |
|---|---:|---:|
| Selected prior control | 18.816 µs | — |
| Earlier QKV release only | 18.868 µs | −0.054 µs |
| Earlier QKV/attention release, no preloads | 18.831 µs | −0.015 µs |
| **Both historical K/V, FP32 staging, both early releases** | **17.104 µs** | **1.709 µs** |
| Same preloads, preceding QKV release | 17.508 µs | 1.296 µs |
| Same preloads, preceding attention release | 17.793 µs | 1.024 µs |
| Key-only FP32 staging, preceding QKV release | 17.731 µs | 1.081 µs |

All six ring rounds improve for the selected preload schedule. Earlier release alone does not establish a gain; the selected result combines the loads and release schedule.

Each full-model mode owns independent context/head/initial-audio graphs and matching eager dispatch. One warmup group is excluded per process, and all measured samples remain. TTFA includes text preparation and codec decoding, excluding reference registration and network transit.

| Comparison | Control median | Selected median | Median paired gain | Faster |
|---|---:|---:|---:|---:|
| Initial, 12 triplets | 73.423 ms | 71.706 ms | 1.750 ms | 12/12 |
| Repeat, 20 pairs | 73.636 ms | 71.856 ms | 1.685 ms | 19/20 |

The initial alternate retaining the preceding QKV release measures 72.068 ms, with 1.526 ms paired gain. Repeat p95 is **73.853 → 73.058 ms**; the selected path's **80.543-ms outlier remains included**. These small samples do not establish a general tail guarantee.

The repeat's initial 32 LLM steps fall **60.409 → 58.587 ms**. Unchanged prefill is **6.833 / 6.835 ms** and first codec **4.531 / 4.532 ms**. All **76 measured complete PCM streams** preserve final sampling RNG; all **32 control hashes** match prior output. **144 full-model graph cases** preserve logits, IDs, RNG and all 72 entire poisoned KV buffers across context/head boundaries. Equivalence is to the selected calibrated G32 path, not upstream BF16.

## Cloning and integration validation

Memcheck, racecheck and synccheck each pass **360 changed-history replay cases**, with zero errors; racecheck reports zero hazards and warnings. Graphs copy position and entire staged caches before QKV, so historical data changes inside the captured dependency chain. Current slots are poisoned, positions move forward/backward, and real/zero/spike inputs exercise three layers and 128/1024 capacities. Four schedules cover packed/FP32 and key/both-cache preloads. Those runs expose intermediate QKV stores for comparison; a separate production-producer check is recorded alongside them.

The first memcheck attempt exited on a host validator bug for the first layer's absent residual. Its zero-error banner does not count as a successful check. The corrected completed run is `memcheck_v2`. This is operator-chain coverage, not sanitization of the whole service.

Supplementary runs with the selected **production QKV binary**, without intermediate debug stores, pass **90 additional cases per sanitizer**, again with zero errors/hazards/warnings. [Production sanitizer results](results/attention_history_production_sanitizers_v1.json) preserve this separate coverage; aggregate coverage is 450 cases per tool.

All **48 generated Chinese/English WAVs**, eight reference copies and generation metadata match the preceding gate/up compiler suites byte for byte. Existing diagnostics apply to the identical files; no new ASR run or human-quality claim is made. The ordinary five-request CLI, without audio-head buckets, measures **72.211 ms median / 72.352 ms p95** and preserves its entire 36-step diagnostic dictionary and saved WAV. It is an integration check, not another paired gain estimate.

## HTTP and fresh voice cloning

One temporary loopback server alternates two graph sets on one shared model, codec and reference encoder. It switches the matching eager hidden callable as well. The client measures through the first complete **3,840-byte/80-ms PCM chunk** and drains every measured utterance.

| Workload | Control median | Candidate median | Median paired gain | Faster |
|---|---:|---:|---:|---:|
| Cached voice, 20 pairs | 76.436 ms | 75.022 ms | 1.919 ms | 20/20 |
| Fresh registration plus synthesis, 10 pairs | 112.128 ms | 110.665 ms | 2.757 ms | 9/10 |

Cached p95 is **78.111 → 76.552 ms**; fresh p95 is **130.608 → 113.455 ms**, including the retained 142.181-ms control outlier. Fresh timing begins before reference-registration HTTP and includes subsequent synthesis. Client WAV/base64 preparation precedes timing, and every fresh request re-encodes the reference.

Internal engine paired gains are **1.931 ms** for both workloads. Cached initial generation is **60.397 / 58.489 ms**. Reference-registration medians are **34.757 / 34.566 ms**, with **0.461 ms paired difference despite unchanged encoder code**; the full fresh gain is not attributed solely to the new kernel.

All **60 measured HTTP streams** retain complete PCM/RNG, and twenty cached controls match earlier HTTP hashes. Invalid variant/voice checks and cancellation recovery **429 → 200** pass. The temporary server stops afterward. A separate smoke test of the actual production `--attention-history` CLI registers a fresh voice, verifies a complete stream against prior PCM, and stops its server.

## Remaining work and reproduction

The post-timing profile contains 12,101 kernel events across 33 decode steps and two codec frames. All 1,188 chronological QKV/attention pairs have overlapping resident intervals, median **7.904 µs**. This includes dependency waits and is not a measurement of simultaneous useful work. Gate/up and down remain the largest projection intervals, medians **18.976 / 25.216 µs** and summed resident times **22.823 / 30.230 ms**. These overlap and must not be added as critical-path shares, utilization or a lower bound.

LLM generation remains the principal cached-voice bottleneck; reference encoding adds substantial time for fresh cloning. Further projection fusion and dependency scheduling remain open. Nothing establishes that 50 ms is impossible, so all 32 codebooks remain required.

Append `--attention-history` to the complete qualified `--gateup-compiler` command. Run GPU jobs sequentially with fresh tags:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_history --tag NEW --layers 36 --rounds 6
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_history_paired --tag NEW --configs q1_a1_m3_float --rounds 20 --profile
/venv/moss-vllm/bin/python -m optimization.benchmark_history_http --tag history_NEW --rounds 20 --fresh-rounds 10
/venv/moss-vllm/bin/python -m optimization.validate_history_server --tag NEW
```

Primary evidence: [attention_history_pass_summary_v1.json](results/attention_history_pass_summary_v1.json), [quality equivalence](results/attention_history_quality_exact_v1.json), [HTTP samples](results/paired_http_history_v1.json), and [source/binary archive hashes](results/attention_history_source_hashes.json). `analyze_attention_history.py` reconstructs the summary while validating its inputs. Compiler snapshots, the ineffective first build, failed guards, all timing samples and profiles remain under `results/`.
