# Native projection scheduling and integer tensor-core experiments

This continuation retains **all 32 codebooks**, voice cloning, BF16 prefill and the FP32 codec. The selected path entering this pass is the scaled-DP4A implementation: **87.67 ms median warm in-process TTFA** in its previously recorded matched comparison. The 50 ms objective remains unmet. Operator timings below are not TTFA measurements.

## Why projections were revisited

The preceding complete-request profile attributes 51.9% of GPU time to projections. The selected Triton kernels still contain layout conversions and shared-memory barriers. This pass tests whether native scheduling, explicit Gluon layouts or integer tensor cores can remove enough of that cost to improve complete streaming requests.

The arithmetic contract is the existing calibrated INT4 model, not the original BF16 checkpoint. Signed nibbles are expanded into signed bytes scaled by sixteen; the integer factor is removed before FP32 conversion. Every group retains its original 32 activation values and scale. The floating reductions reproduce the selected compiler's local multiply/FMA order, warp XOR order and BF16 rounding boundaries. The [PTX integer instruction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a) and [MMA fragment mapping](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832) inform these implementations. Synchronization and resource tradeoffs follow NVIDIA's [kernel programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html).

## Implemented alternatives

- `dp4a_native.cu/.py`: warp-per-row or warp-pair-per-row projections, activation caching, register tiling, optional offline group permutation, and fused gate/up/SiLU/activation quantization. The permutation changes storage order only. Small K4096 projections require no block barrier; down and fused gate/up use a final shared handoff.
- `dp4a_staged.cu/.py`: coalesced integer-group processing followed by shared-memory transfer into the exact floating reduction layout. Three versions test chained dots, four independent dot chains, and staging only integer sums. Original CUDA sources, compilation logs and binaries are retained by source hash.
- `dp4a_layout.py`: scaled-integer dots with explicit Gluon integer and floating layouts. Scales are loaded directly into the floating layout. Four configurations change some results despite the intended layout contract and remain rejected.
- `dp4a_mma.cu/.py`: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` computes diagonal group products. Other matrix outputs are unused. Follow-ups repack unchanged weight codes into the A-register layout and limit loop unrolling. This preserves the grouped activation quantization and floating epilogue.

Native kernels compile to SM90 cubins with the installed CUDA 12.8 toolkit. They use the caller's current CUDA stream. No toolkit/driver change or service replacement is involved.

## Operator results

Each cell is **best exact candidate / contemporaneous selected-Triton reference**, in microseconds; lower is better. The benchmark rotates eight distinct weight/scale pairs, captures 24 calls per graph, and reports the median of nine event timings. Weight packing is performed before timing. Each candidate is compared on three real layers, four recorded inputs per layer and zero/spike inputs, including private-stream execution.

| Experiment | QKV | Output | Fused gate/up | Down |
|---|---:|---:|---:|---:|
| Native warp scheduling + optional permutation | 7.54 / 6.08 | 6.63 / 5.09 | 20.62 / 17.08 | 13.67 / 10.14 |
| Native independent integer chains | 8.26 / 6.12 | 6.53 / 5.13 | 20.56 / 17.16 | 13.12 / 10.17 |
| Shared sums and coefficients | 6.44 / 6.18 | 5.13 / 5.13 | 23.72 / 17.19 | 12.47 / 10.19 |
| Shared staging + independent chains | 6.46 / 6.17 | 5.07 / 5.17 | 27.34 / 17.18 | 12.90 / 10.15 |
| Shared integer sums only | 6.71 / 6.09 | 5.30 / 5.09 | 24.96 / 17.10 | 12.22 / 10.13 |
| Explicit Gluon layouts | 6.09 / 6.07 | 5.07 / 5.09 | 17.58 / 17.11 | 10.13 / 10.10 |
| Integer MMA, original weight layout | 15.80 / 6.10 | 15.30 / 5.17 | 81.01 / 17.21 | 44.69 / 10.21 |
| Integer MMA, repacked A fragments | 12.37 / 6.13 | 7.92 / 5.16 | 46.86 / 17.27 | 23.85 / 10.14 |
| Repacked MMA, limited loop unrolling | 11.86 / 6.15 | 11.32 / 5.15 | 38.47 / 17.26 | 32.00 / 10.19 |

Across these completed sweeps there are **448 configuration measurements and 8,064 operator checks**. Of those checks, 8,054 are exact; 444 configurations pass all 18 checks. The remaining ten failed checks belong to four Gluon configurations. No changed-output configuration is selected. `results/native_projection_experiments.json` lists the raw benchmark artifacts.

The tensor-core packing removes redundant weight fetches and improves its own implementation, but it does not beat DP4A. Limiting unrolling helps gate/up while regressing some other projections. Neither fewer barriers nor fewer dot instructions alone establishes a speedup. The initial MMA binary includes register spills in its larger kernels; the repacked binary removes those spills but still uses up to 252 registers per thread. Build logs retain the per-instantiation details.

## Near-winning output projection

The staged independent-dot output variant uses R4/W8, FP32 scales, 32 registers per thread, no spills and 4,128 dynamic shared bytes. It stages integer sums and scale products, then executes one block barrier and the original floating reduction. `dp4a_output_native.cu/.py` isolates that configuration for further validation.

Twelve rotating/reversing operator rounds measure selected Triton **5.1067 µs**, native staged **4.9993 µs**, and Gluon **5.0900 µs**. The median paired native gain is **2.24%**, versus 0.36% for Gluon. This meets the 2% operator threshold for a full-request trial; it does not establish a TTFA improvement. Evidence: `native_projection_repeat_v1.json`.

The isolated native kernel passes **504 all-layer real/zero/spike comparisons** and eight private-stream graph checks with output row counts 1, 3, 4, 5, 37, 4,095, 4,096 and 4,097. Compute Sanitizer memcheck reports zero errors and racecheck reports zero hazards for those eight graph cases. The full-request comparison is recorded separately below.

Ten same-process complete-request pairs measured **87.481 ms control / 87.444 ms native-output median TTFA**, with p95 89.720 / 88.778 ms. The **median paired gain is only 0.0023 ms**, with five improvements and five regressions. One control request is 2.96 ms slower than its candidate partner; the cause is unknown, and all raw observations are retained. All ten full float32 PCM streams match exactly, remain finite and complete without truncation. This does **not** establish a useful request-level improvement, so native output remains an internal benchmark option and is not added to the serving preset. No new HTTP or bilingual quality run is claimed for this rejected change. Evidence: `native_output_paired_v1.json/.log`.

The selected scaled-DP4A plan remains unchanged. Its prior 87.67 ms matched-comparison median is still the reference improvement result; the 87.48 ms control here is a new observation of the same implementation, not another optimization. Projection scheduling has not removed the approximately 38 ms remaining gap to the target.

## Independent MMA integer verification

`mma_integer.cu/.py` verifies the packed register mapping against independent Torch integer multiplication and summation. It covers every signed INT4 weight and signed INT8 activation in each of 32 positions, all positions together, 65,536 random groups, and a second reversed weight row. All **401,408 dots** and the packing roundtrip match exactly on a private stream inside a CUDA graph. Compute Sanitizer memcheck reports **zero errors**. This validates the integer primitive; it does not qualify the slower MMA projection implementation for serving.

## Retained failed attempts

The first native sweep used exported BF16 scales for output/down, whereas the selected serving plan casts those scales to FP32. Its down stage also failed on the unsupported scale format. That incomplete artifact is annotated and excluded from the table. The subsequent complete sweeps use the correct formats.

The first Gluon sweep failed compilation because a twice-expanded index had an incompatible sliced layout. Its error records remain in `dp4a_layout_kernels_v1.json/.log`. The corrected `v2` sweep supplies the measurements above. Compilation failures and numerical mismatches are not counted as successful optimizations.

## Reproduction

Run GPU commands sequentially in `/workspace/MOSS-TTS`. The native/staged/MMA source files now contain the latest experimental variants; earlier implementations are preserved under the corresponding `results/*_build/` directories. Each result records its native source snapshot and SHA-256. Original benchmark logs and all raw timing samples remain available.

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_dp4a_native --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_dp4a_staged --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_dp4a_layout --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_dp4a_mma --tag reproduce
/venv/moss-vllm/bin/python -m optimization.validate_native_output --tag reproduce
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.mma_integer
/venv/moss-vllm/bin/python -m optimization.benchmark_native_output_paired \
  --pairs 10 --tag reproduce
```

The timing boundary for complete requests includes text processing, prefill, delayed generation and first complete 80 ms CPU PCM chunk; it excludes network transit and fresh reference encoding. Existing running services retain their previous configurations and voice caches.

`native_projection_source_hashes.json` and `native_projection_sources.tar.gz` preserve the resulting source snapshot. The source archive excludes checkpoints, binary builds and measured output artifacts, which remain under `results/`. The active optimization goal remains open; these results do not prove that 50 ms is impossible with 32 codebooks.
