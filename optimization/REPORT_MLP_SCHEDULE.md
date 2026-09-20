# MLP schedule review and paired HTTP measurements

No new MLP schedule is selected in this pass. The existing **eight-row down tile** remains the best tested path, with all **32 acoustic codebooks**, streaming voice cloning, calibrated G32 decode, BF16 prefill and the FP32 codec. The 50-ms target remains unmet.

A new same-process HTTP comparison supports the preceding tile's median gain: **77.789 → 77.312 ms**, with **0.329 ms median paired improvement** and 15/20 pairs faster. Its p95 regresses, **80.281 → 81.018 ms**, and the 103.690-ms candidate outlier is retained. This is a temporary benchmark server with two graph sets, not a deployed service. It complements rather than replaces the earlier separate-process HTTP regression in [REPORT_DOWN_TILE.md](REPORT_DOWN_TILE.md).

## Rechecking the MLP dependency chain

The preceding pass made verified progress. Its **378-file** archive and every archived working-tree file were revalidated before this pass. The H200 was idle with the original two services running. GPU jobs remained sequential.

`benchmark_mlp_schedule.py` tests **58 configurations** including control and three explicit aliases of the same selected settings:

- Twelve gate/up and down producer-release combinations. Every consumer retains its original dependency wait; zero means the implicit kernel-completion release, not removal of synchronization.
- Twenty-seven down integer layouts across eight/sixteen-row and four/eight-warp tiles, with group and row distribution parameters of one, two and four.
- Eighteen gate/up integer layouts using 32/64 output rows and the same three group/row distribution choices.

The consumer is the selected clustered QKV/head-preparation kernel. The corrected single-layer pilot passes **174** bytewise output/cache comparisons. The complete 36-layer weight ring passes **6,264** comparisons and **58** private CUDA graph checks, including whole KV buffers with poisoned current positions. All actual weight layers and three frozen inputs per layer are used; attention metadata is the nearest saved fixture, and layer 35 wraps synthetically to layer 0. These are chain screens, not actual request trajectories or TTFA.

Every changed setting loses in the full ring. The selected chain measures **36.683 µs**. The closest release variant, `t_u1_d0`, measures **36.774 µs** and loses **0.107 µs** paired in all six rounds. Gate/up `u_r32_ig1ir2` measures **36.788 µs** and loses **0.128 µs**, also in all six rounds. Their small favorable hot-pilot readings do not transfer to all weights.

Fewer registers are not sufficient: `u_r32_ig1ir2` reduces gate/up registers from **162 to 160**, but shared memory grows from **1 to 2 KiB** and latency worsens. Some wider settings spill; `u_r32_ig4ir1` uses 255 registers with 156 reported spills and takes **134.981 µs** per chain. Resource metadata and cubin hashes for every configuration are retained. These observations do not identify actual occupancy or a hardware latency lower bound.

## Complete-model rejection

Both closest alternatives are tested in full requests despite the frozen-ring losses, to check whether their interactions change on real trajectories. They keep separate context/head/initial-audio graph sets and restore eager dispatch when selected.

| Variant | Median TTFA | Median paired gain versus control | Faster pairs |
|---|---:|---:|---:|
| Selected eight-row control | 73.531 ms | — | — |
| Implicit down release | 73.631 ms | −0.123 ms | 1/12 |
| Gate/up IG1/IR2 | 73.731 ms | −0.187 ms | 2/12 |

All **36 measured complete streams** and final RNG states match. All twelve control hashes match preceding selected output. **96** full-model graph checks pass for IDs, logits, sampling state and all 72 entire KV buffers after poisoning. One warmup triplet is excluded. Both variants remain benchmark-only; no serving flag or selected kernel changes.

## Paired HTTP and fresh cloning references

`paired_http_server.py` copies the existing transport from `server.py` SHA-256 `d040811ca58c6a1affec0ebe6323ec73aab86ed43b88200e64e8fb60cbff3084`, adding an explicit benchmark variant field, request identifiers, second graph capture and RNG diagnostics. A single GPU worker holds the original busy lock. It selects the corresponding context/head/initial-audio graphs and down callable inside that worker. Both variants share model weights, the FP32 codec, reference encoder and registered voice.

`benchmark_paired_http.py` runs balanced AB/BA pairs against that one process. This removes the separate-process comparison's large time separation, while still measuring real loopback HTTP through the first full **3,840-byte, 80-ms PCM chunk**. It does not eliminate request-level variation. Generated audio is never cached.

| Workload | Control median | Down-tile median | Median paired gain | Faster pairs |
|---|---:|---:|---:|---:|
| Cached cloned voice, 20 pairs | 77.789 ms | 77.312 ms | 0.329 ms | 15/20 |
| New reference registration plus synthesis, 10 pairs | 110.933 ms | 109.967 ms | 1.204 ms | 7/10 |

All **60 measured complete streams** have identical PCM and final RNG state within each pair. All twenty cached controls also match prior HTTP PCM hashes. The test re-encodes a fresh reference for each fresh-workload request; reference WAV/base64 preparation precedes its timer, which starts before registration HTTP and ends at the first PCM chunk of the following synthesis request. One warmup pair per workload is excluded.

Cached p95 is **80.281 / 81.018 ms**. The candidate's 103.690-ms outlier includes an **80.114-ms engine TTFA**, a 62.114-ms initial LLM interval and 6.117-ms first codec chunk. The HTTP measurement has additional delay beyond those engine stages; these diagnostics do not establish its cause. No outliers are removed.

For cached requests, internal engine medians are **74.069 / 73.551 ms**, with **0.538 ms** median paired engine gain. Initial LLM generation is **61.037 / 60.520 ms**. Unchanged prefill is **6.824 / 6.829 ms** and unchanged first codec **4.522 / 4.524 ms**. This aligns with the prior paired engine improvement, without supporting a tail-latency claim.

For fresh requests, internal engine medians are **74.038 / 73.608 ms**, with **0.471 ms** paired gain. Registration medians are **33.470 / 33.277 ms**, with **0.376 ms** paired difference despite identical encoder code. Thus the 1.204-ms whole-workflow gain is not attributed solely to the down tile or presented as an encoder improvement. Fresh p95 is **126.257 / 114.468 ms**, heavily influenced by the retained 135.719-ms control outlier; ten pairs do not establish a general tail result.

Unknown variant/voice rejection and cross-variant cancellation recovery pass, including **429 → 200** recovery. The temporary server stops. The first attempt correctly failed initialization because the original context manager was still marked installed; its source/log are retained. The corrected setup captures and restores each variant's context manager explicitly. The failed attempt supplies no performance evidence.

## Research and next hardware target

The current [Gluon layout tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html) and [Linear Layouts paper](https://arxiv.org/abs/2505.23819) explain the distinction between register rearrangement and layout conversions requiring thread/warp communication. The measured sweep keeps floating layouts fixed while changing integer mappings, rather than assuming that fewer resources guarantees speed.

The September 11 [Hopper utilization study](https://arxiv.org/abs/2609.12923) distinguishes resource limits, occupancy, grid coverage and useful tensor-core work. Its BF16 vLLM/H100 results do not transfer quantitatively to this H200 DP4A path. Host counter restrictions remain; the measurements here do not claim counter-derived utilization. The September 2 [Leech-lattice serving paper](https://arxiv.org/abs/2609.02652) also distinguishes on-disk bit rate from served memory traffic and reports a quality cost. Its format is not an exact storage transformation of these GPTQ weights; no implementation or MOSS speedup is claimed from reading it. Original synchronization continues to follow [CUDA's PDL contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

The gate/up kernel remains a concrete target: its selected 32-row tile requires 162 registers per thread, and simply changing integer layouts or release points did not help. A next experiment can split one 32-value gate/up output-quantization group across two or four SM90 CTAs, exchange that small output through cluster shared memory, and preserve the exact 32-value activation quantizer. This may reduce per-CTA register pressure without duplicating the full gate/up projection. It is a hypothesis, not a measured gain. The existing isolated Triton 3.8 cluster compiler and explicit legacy normalization/reduction helpers are available for it. All 32 acoustic codebooks remain required.

## Reproduction

Run sequentially in `/workspace/MOSS-TTS` with `/venv/moss-vllm/bin/python`. Use new tags to preserve existing evidence.

```bash
python -m optimization.benchmark_mlp_schedule --tag NEW --layers 36 --rounds 6
python -m optimization.benchmark_mlp_schedule_paired --tag NEW --rounds 12 --configs t_u1_d0 u_r32_ig1ir2
python -m optimization.benchmark_paired_http --tag NEW --rounds 20 --fresh-rounds 10
```

Primary artifacts: `mlp_schedule_ring_v1.json`, `mlp_schedule_paired_v1.json`, `paired_http_down8_v2.json`, and `paired_http_down8_v2_stages_summary.json`. The HTTP harness owns and terminates its loopback server; it does not replace either supervisor service.
