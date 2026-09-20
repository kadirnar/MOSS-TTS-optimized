# Down-projection tiling and cluster placement

The eight-row/four-warp down-projection tile measures **73.773 ms median warm cached-voice TTFA**, compared with **74.233 ms** for the preceding clustered-QKV preset. The median paired gain is **0.447 ms**, with **17/20** pairs faster. The p95 slightly regresses, **74.783 → 74.838 ms**. This is a median improvement, not a tail-latency or 50-ms success. All **32 acoustic codebooks**, streaming voice cloning, calibrated G32 decode, BF16 prefill and the FP32 codec are retained.

TTFA starts at complete-text submission and ends at the first playable PCM chunk. It includes text preparation, prefill, initial LLM generation, codec decoding and CPU PCM availability. Cached-reference timing excludes voice registration and network transit. HTTP and fresh-reference observations are recorded separately. Exactness is against the selected calibrated G32 model, not the original BF16 model.

## What changed

Optional `--down-tile8` installs a per-model down-projection callable before graph capture. It requires the selected `--qkv-cluster` preset. The tile changes from four rows/two warps to eight rows/four warps; weights, scales, activation grouping, FP32 reduction order, BF16 rounding, dependency waits and release points are unchanged. Scale-only register preload and the 1/16 bulk weight hint remain enabled. There are 512 rather than 1,024 CTAs per down projection, with the same total thread count.

Both tiles use **128 registers per thread and zero spills**. Shared memory increases from **1 to 2 KiB per CTA**. The emitted PTX retains eight static global-load instructions before the dependency wait, and the SASS contains its `ACQBULK` wait. These are compiler-resource observations, not measured occupancy or memory-throughput claims. PTX, Gluon IR, cubins, SASS and hashes are in `results/down_tile_audit_v1/`.

`down_tile.py` validates the selected 32-codebook G32/PDL path and installs the callable without changing module-global plans. Serving, quality generation and the ordinary benchmark expose the flag, defaulting off. Existing supervisor services are not replaced.

## Experiments and rejected alternatives

The cluster-placement sweep changes launch attributes on fresh modules loaded from the same qualified SM90 cubins. It tests DEFAULT/SPREAD/LOAD_BALANCING policies and five shared-memory carveout preferences. All **1,728** full-ring comparisons and sixteen private graph checks pass, including complete poisoned-cache bytes. The best full-ring setting, load balancing with 50% carveout, saves only **0.030 µs** paired; its apparent **0.256 µs** hot single-layer gain does not transfer. Zero carveout is consistently much slower.

The CUDA occupancy API reports a potential 124 clusters for default/spread and 132 for load balancing; this is not proof of actual residency, and the kernel launches only 48 clusters. Complete requests reject the 50% carveout setting: **−0.030 ms** paired gain, only 4/12 faster. Load balancing without a carveout gains **0.110 ms** in that first process, but **−0.018 ms** in the subsequent twenty-round comparison. It is therefore unselected. The selected cluster policy remains SPREAD.

The down sweep tests thirty configurations: control; two/four/eight-row tiles with two/four/eight warps and scale-only, weight-only or combined preloads; and two sixteen-row combined-preload tiles. The consumer is the selected clustered QKV/head-preparation kernel. The real 36-layer ring passes **3,240** output/cache comparisons and thirty private graph checks. Its eight-row/four-warp scale-only chain measures **36.734 µs**, versus **37.143 µs** control; median paired gain **0.419 µs**, all six rounds faster. The single-layer pilot also favors it by 0.411 µs. Other schedules lose in the full ring, including every packed-weight preload. More warps reduce registers for some small tiles but do not improve latency.

The ring uses actual projection weights and three frozen inputs per layer with nearest saved attention metadata. Its final layer synthetically wraps to layer 0. It is not an actual request trajectory or TTFA. The initial pilot had a Python comparison-helper error on lists; its failed log/source are retained, and it contributes no successful checks or timings. The corrected pilot passes ninety comparisons.

## Complete requests and validation

Each mode owns separate context/head/initial-audio CUDA graphs, and restores eager dispatch when selected. Requests use balanced rotated/reversed order, identical seeds and a cached cloned voice. One warmup set per process is excluded.

| Run | Control median | Down-tile median | Median paired gain | Faster pairs |
|---|---:|---:|---:|---:|
| Four-mode comparison | 72.603 ms | 71.235 ms | 0.589 ms | 19/20 |
| Final two-mode repeat | 74.233 ms | 73.773 ms | 0.447 ms | 17/20 |

The first comparison also tests placement alone and both changes together. It crosses faster/slower runtime phases, which explains why subtracting its medians differs substantially from its paired median gain. The final comparison is the primary latency result. No cross-process absolute difference is attributed to this tile.

Across the placement screen and both complete-model comparisons, all **156 measured complete streams** preserve PCM and final RNG state; all **52** control hashes match prior selected results. There are **288** full-model graph checks across these processes, covering IDs, logits, RNG state and all 72 KV buffers after poisoning current positions. The final two-mode run contributes 48 of these checks and forty streams.

The final initial-32-step LLM interval is **61.048 → 60.564 ms**. Prefill is **6.830 / 6.828 ms** and first codec chunk **4.527 / 4.528 ms**. Unchanged-stage differences are runtime variation.

Memcheck, racecheck and synccheck each pass **72 cases** with zero memory errors, synchronization errors, or race hazards/warnings. Coverage includes private changed-input graphs of gate/up → down → clustered QKV, production/diagnostic consumers, positions 0/127/1023, three real layers and real/zero/spike inputs. Separate projection cases exercise normal and odd output-row counts. This is operator/chain coverage, not whole-service sanitizer coverage.

## Cloning, HTTP and remaining bottleneck

The integrated option generates all **48** original/expanded Chinese and English cloning utterances. All generated WAVs, eight reference copies and generation metadata are byte-identical to the preceding clustered-QKV suites. The proof is `down_tile_quality_exact_v1.json`. Prior identical-audio normalized Chinese CER **5.37% / 1.18%**, English WER **0% / 0%**, and speaker cosine **0.9143 / 0.9149** therefore apply to these samples. No ASR rerun or new human listening assessment is claimed.

The sequential temporary-server HTTP test **regresses**, from **75.754 to 77.418 ms median**, and **76.958 to 78.497 ms p95**. All twenty complete PCM streams match; invalid-input handling and cancellation recovery pass, including **429 → 200** recovery. The internal engine medians are **71.866 / 73.808 ms**, and initial LLM intervals **59.327 / 60.645 ms**. Unchanged prefill also shifts **6.688 / 6.843 ms**, and unchanged first codec decode **4.195 / 4.539 ms**. These differences are consistent with broader runtime variation, but do not establish its cause or an HTTP speedup. The paired in-process result supports the tile's median gain; this service test remains a regression and the flag stays opt-in.

One fresh-registration-plus-synthesis observation is **114.908 / 115.102 ms**. Registration itself is **34.718 / 35.189 ms**; no reference-encoder improvement is claimed. Both temporary servers stop, port 18084 is closed afterward, and the original supervisor PIDs 54030 / 56179 remain ready with 32 codebooks.

The independent benchmark CLI, without audio-head buckets, completes five requests at **74.299 ms median / 74.948 ms p95**. Its entire 36-step diagnostic dictionary and saved WAV match the preceding selected CLI output. This separate-process run is entry-point validation, not another paired speedup claim. All three CLI help checks expose the new flag.

The post-timing 34-step profile has **12,101 kernel events**, unchanged in count. Gate/up and down remain the largest projection intervals: medians **19.137 / 25.280 µs**, summed resident intervals **23.007 / 30.346 ms**. These include dependency waits and overlap, so they are not additive critical-path shares, utilization or latency lower bounds. Initial LLM generation still dominates the measured request. Further gate/up/down register and reduction layouts are concrete targets; these results do not establish that 50 ms is impossible and do not justify reducing codebooks.

## Reproduction

Use `/venv/moss-vllm/bin/python` and run GPU jobs sequentially from `/workspace/MOSS-TTS`. Each result tag must be new.

```bash
python -m optimization.benchmark_cluster_placement --tag NEW --layers 36 --rounds 8
python -m optimization.benchmark_cluster_placement_paired --tag NEW --rounds 12 --configs p2_c50 p2_cNone
python -m optimization.benchmark_down_preload --tag NEW --layers 36 --rounds 6
python -m optimization.benchmark_down_placement_paired --tag NEW --rounds 20
python -m optimization.benchmark_down_placement_paired --tag NEW_FINAL --rounds 20 --configs down_r8 --profile
compute-sanitizer --tool memcheck --error-exitcode 1 python -m optimization.validate_down_tile --tag NEW_MEM
compute-sanitizer --tool racecheck --error-exitcode 1 python -m optimization.validate_down_tile --tag NEW_RACE
compute-sanitizer --tool synccheck --error-exitcode 1 python -m optimization.validate_down_tile --tag NEW_SYNC
python -m optimization.audit_down_tile --tag NEW
python -m optimization.benchmark_down_tile_http --tag NEW
```

Add `--qkv-cluster --down-tile8` to the complete register-preload serving command in [REPORT_ASYNC_WEIGHTS.md](REPORT_ASYNC_WEIGHTS.md). No new model download, driver change or runtime upgrade is required.

The launch-policy experiment uses the [CUDA driver launch-attribute API](https://docs.nvidia.com/cuda/cuda-driver-api/unionCUlaunchAttributeValue.html). The register/shared-memory tradeoffs follow the [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/). Producer-dependent activation reads remain behind the synchronization required by [NVIDIA's PDL documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html); possible overlap is not assumed to be guaranteed concurrency.

Follow-up: [REPORT_MLP_SCHEDULE.md](REPORT_MLP_SCHEDULE.md) records a later same-process, alternating HTTP comparison and ten fresh-reference pairs. It retains this report's separate-process regression and all new outliers.
