# Exact gate/up compiler and cluster experiments

Optional `--gateup-compiler` improves warm cached-voice TTFA from **73.884 to 73.502 ms** in the final twenty-pair comparison, with **0.238 ms median paired gain** and 14/20 pairs faster. Paired loopback HTTP measures **77.401 → 77.102 ms**, with **0.210 ms paired gain** and 13/20 faster. The selected change is a **single-CTA** gate/up kernel; the tested multi-CTA alternatives are slower.

All **32 acoustic codebooks**, streaming voice cloning, calibrated G32 decode, BF16 prefill and the FP32 codec remain enabled. All 48 bilingual cloning WAVs match the preceding selected path byte for byte. **The 50-ms target remains unmet.** Existing supervisor services are unchanged; the new option defaults off.

## Experiment and implementation

The preceding MLP-layout/paired-HTTP pass made progress. Before continuing, all 384 files and the archive hash in `mlp_schedule_source_hashes.json` were revalidated. GPU experiments ran sequentially on the same H200 NVL.

`gateup_cluster.py` shards one 32-value gate/up output-quantization group across one, two, four or eight CTAs. Each CTA retains the original residual/RMS normalization and G32 input quantization, computes its assigned projection rows, applies the original BF16 SiLU/product rounding, and cooperates on the output quantizer. G32 activation grouping is separate from the model's 32 acoustic codebooks.

Two communication schemes were tested: gather the 32 BF16 outputs before quantization, or reduce the absolute maximum across CTAs and quantize the distributed outputs directly. Integer layouts, prefix-prefetch sizes and producer-release positions were also varied. All variants retain the original dependency wait. Implicit release does not remove synchronization.

The kernels are compiled with isolated Triton 3.8 and loaded through a checked CUDA Driver ABI in the selected Torch 2.13/Triton 3.7.1 host. Explicit FP32 normalization and four-product reduction sequences preserve the selected compiler's operation order. Every other kernel remains on the selected runtime. Two initial exports failed strict layout-type checks on reshaped quantized values; explicit layout conversion corrected them. Failed source snapshots/logs remain separate and supply no performance evidence.

`gateup_cluster_bundle_screen_v1` contains all 112 production/debug, residual/no-residual cubins for 28 configurations, with PTX, Gluon IR, SASS, resource metadata and hashes. The smaller `gateup_exact_bundle_v1` copies only the four selected binaries unchanged. Selected production residual cubin SHA-256 is `1d108a9b31f6288edb002412c239c3aff802b25f7f1827cdc3b353319652716e`.

`gateup_compiler.enable()` installs a per-model dispatch before graph capture. It requires the qualified G32, clustered-QKV, eight-row-down preset, all 32 codebooks and the selected host runtime. Serving, quality generation and the ordinary benchmark expose `--gateup-compiler`; other options and defaults retain their behavior.

## Cluster screen and selected layout

The pilot passes 15 comparisons. Its first all-layer ring passes **540** comparisons and five private graphs. The expanded ring passes **3,132** comparisons and 29 private graphs with no failures. Checks include normalized values, quantized inputs/scales, gate/up outputs and output quantizer, downstream projections and whole KV caches after poisoning. All 36 weight sets and three frozen inputs per layer are used; the following attention metadata is the nearest saved fixture and layer 35 wraps synthetically to layer 0. These are dependency-chain screens, not TTFA or complete model trajectories.

The expanded screen uses six balanced rotated/reversed rounds:

| Gate/up variant | Registers/thread | Shared/CTA | Chain median | Median paired gain |
|---|---:|---:|---:|---:|
| Selected prior control | 162 | 1 KiB | 36.660 µs | — |
| One CTA, IG2/IR4 | 134 | 1 KiB | 36.602 µs | 0.056 µs |
| **One CTA, IG1/IR2** | **148** | **2 KiB** | **36.552 µs** | **0.075 µs** |
| Two CTAs, IG2/IR4 | 80 | 1 KiB | 38.252 µs | −1.597 µs |
| Two CTAs, IG1/IR2 | 80 | 2 KiB | 38.132 µs | −1.459 µs |
| Two CTAs, distributed maximum | 80 | 1 KiB | 38.801 µs | −2.133 µs |
| Four CTAs, distributed maximum | 72 | 1 KiB | 42.420 µs | −5.761 µs |
| Eight CTAs, distributed maximum | 70 | 1 KiB | 51.403 µs | −14.735 µs |

These rows have zero reported spills. The worst one-CTA IG4/IR1 layout uses 255 registers and 90 reported spills, taking 114.925 µs. The initial ring also rejects four/eight-CTA gathered-output variants at 42.281/50.371 µs versus 36.713 µs control.

The two-CTA kernel's PTX contains cluster barriers and a distributed shared-memory load. Reduced register counts do not establish a speedup or actual occupancy: the tested clustered schedules lose once communication, duplicated normalization and additional CTAs are included. The selected one-CTA variant changes both compiler scheduling and integer layout while explicitly retaining arithmetic. Its gain is not attributed to either change alone.

## Complete streamed requests

Each mode has independent context/head/initial-audio graphs and matching eager dispatch. One warmup group is excluded per process; all measured samples and outliers remain.

| Comparison | Control median | IG2/IR4 median | IG1/IR2 median | IG1/IR2 paired gain | Faster |
|---|---:|---:|---:|---:|---:|
| Initial, 12 triplets | 73.568 ms | 73.514 ms | 73.396 ms | 0.113 ms | 8/12 |
| Repeat, 20 triplets | 73.702 ms | 73.600 ms | 73.562 ms | 0.265 ms | 17/20 |
| Final, 20 pairs | 73.884 ms | — | 73.502 ms | 0.238 ms | 14/20 |

The final p95 is **75.321 → 74.775 ms**, although the initial candidate p95 regresses, **74.561 → 76.401 ms**. These small samples do not establish a general tail guarantee. Initial/repeat IG2/IR4 paired gains are 0.038/0.130 ms; IG1/IR2 is selected after the final comparison.

Across the three comparisons, all **136 measured complete float32 PCM streams** and final sampling RNG states match within their comparison. All **52 control hashes** match prior selected output. **240 full-model private-graph cases** preserve every logit, sampled ID, RNG state and all 72 entire KV buffers after poisoning current positions. Equivalence is to the preceding calibrated G32 path, not original upstream BF16 generation.

The final initial-32-step LLM interval falls **60.541 → 60.311 ms**. Unchanged prefill is **6.824 / 6.825 ms**, and first codec decoding **4.533 / 4.531 ms**. The LLM remains the main bottleneck.

## Cloning, synchronization and integration checks

Memcheck, racecheck and synccheck each pass **252 changing-input private-graph cases**, with zero memory errors, race hazards/warnings or synchronization errors. Coverage includes the selected and alternate one-CTA layouts, two-CTA gathered and distributed quantization, three real layers, residual/no-residual and production/debug cubins, zero/spike inputs and cache positions 0/127/1023. This is operator-chain coverage, not whole-service sanitization.

The integrated option generates all **48 original/expanded Chinese and English cloning utterances**. All WAVs, eight reference copies and generation metadata match the preceding down-tile suites. `gateup_compiler_quality_exact_v1.json` records the hashes. Previous identical-audio ASR/speaker diagnostics therefore apply to these files; there is no new ASR run or human-quality claim.

The ordinary CLI, without audio-head buckets, completes five requests at **74.410 ms median**, p95 **75.145 ms**, with exactly matching 36-step diagnostics and saved WAV. Its separate-process median is slightly slower than the preceding 74.299-ms CLI observation; this is not presented as a speedup. Paired measurements above isolate the kernel change more closely.

## HTTP and fresh voice references

`paired_gateup_http_server.py` retains the preceding benchmark server's transport and single GPU worker. Both modes use the selected eight-row down tile and share weights, FP32 codec and reference encoder; the worker switches the gate/up callable together with all context/head/initial-audio graph sets. This temporary loopback server is benchmark-only.

| Workload | Control median | Candidate median | Median paired gain | Faster |
|---|---:|---:|---:|---:|
| Cached voice, 20 pairs | 77.401 ms | 77.102 ms | 0.210 ms | 13/20 |
| Fresh registration plus synthesis, 10 pairs | 111.810 ms | 110.754 ms | 0.329 ms | 6/10 |

Cached p95 improves **78.892 → 78.500 ms**. Fresh p95 regresses **115.606 → 123.279 ms**, including the retained 128.377-ms candidate request. Fresh measurements start before registration HTTP and end at the first complete 3,840-byte PCM chunk of the following synthesis; WAV/base64 preparation precedes timing, and every fresh request re-encodes its reference.

Cached internal engine medians are **73.566 / 73.324 ms**, with **0.216 ms** paired gain. Initial LLM generation is 60.563/60.338 ms; prefill 6.828/6.832 ms and first codec 4.521/4.520 ms remain effectively unchanged.

Fresh internal engine paired gain is **0.160 ms**. Reference-registration medians are **35.001 / 33.729 ms**, with **0.213 ms paired difference despite unchanged encoder code**. Thus the fresh-workflow median difference is not attributed solely to the new kernel or described as an encoder improvement.

All **60 measured HTTP streams** preserve complete PCM and final RNG state. Twenty cached controls match prior HTTP hashes. Invalid variant/voice rejection and cross-variant cancellation recovery pass. The temporary benchmark server stops afterward. `validate_gateup_server.py` separately checks the actual production CLI flag with fresh registration and a complete stream, then terminates its temporary server.

## Remaining work and reproduction

The post-timing selected profile has 12,101 kernel events. Gate/up and down remain the largest projection intervals: medians **18.849 / 25.120 µs**, summed resident intervals **22.691 / 30.118 ms**. These intervals include dependency waits and overlap; they are not critical-path percentages, counter-derived utilization or a hardware lower bound.

Current [Gluon multi-CTA documentation](https://triton-lang.org/main/getting-started/tutorials/gluon/multicta.html) describes automatic cross-CTA layout communication and its synchronization cost. The [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) provides the cluster/resource constraints. These informed the experiments, not a prediction of their measured speed.

A next bounded experiment is CUDA register allocation around the new one-CTA kernels, especially the 128-register threshold and its effect on coexistence with dependent down CTAs. Previous caps tested a different compiler/kernel; the new 134/148-register schedules justify a targeted repeat. A wider cooperative persistent G32 chain is another hypothesis: [CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) requires collective synchronization and supported cooperative launch, so residency and memory ordering must be checked before such a prototype. Neither idea establishes that 50 ms is impossible. All 32 codebooks remain required.

Append `--gateup-compiler` to the complete selected `--down-tile8` command in the preceding report. It is supported by `optimization.server`, `optimization.quality_generate` and `optimization.benchmark_quantization`, and requires installation before graph capture.

Run GPU jobs sequentially from `/workspace/MOSS-TTS`, using new tags:

```bash
# Isolated compiler only; select the configuration subset to export.
/venv/moss-triton38/bin/python -m optimization.gateup_cluster_binary --tag NEW --configs c1 c2 c4 c8 c1_ig1ir2 c2_dist

# Selected runtime for all measurements and serving.
/venv/moss-vllm/bin/python -m optimization.benchmark_gateup_cluster --tag NEW --bundle gateup_cluster_bundle_NEW --layers 36 --rounds 6
/venv/moss-vllm/bin/python -m optimization.benchmark_gateup_cluster_paired --tag NEW --up-bundle optimization/results/gateup_exact_bundle_v1 --configs c1_ig1ir2 --rounds 20
/venv/moss-vllm/bin/python -m optimization.benchmark_gateup_http --tag gateup_NEW --rounds 20 --fresh-rounds 10
/venv/moss-vllm/bin/python -m optimization.validate_gateup_server --tag NEW
```

Primary evidence: `gateup_cluster_screen_v1.json`, `gateup_cluster_paired_final_v1.json`, `paired_http_gateup_v1.json`, `paired_http_gateup_v1_stages_summary.json`, and `gateup_compiler_pass_summary_v1.json`. Source/binary snapshots and individual samples remain under `results/`.
