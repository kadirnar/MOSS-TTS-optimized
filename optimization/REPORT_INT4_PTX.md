# INT4 PTX lowering and exact projection experiment

The native CUDA experiment is rejected for serving. Its arithmetic is exact on the tests below, but its fastest projection variants take **four to five-and-a-half times** the selected DP4A kernel time. The inspected H200 machine code expands the INT4 PTX instructions into INT8 matrix instructions and conversion sequences. PTX instruction availability did not imply native INT4 hardware throughput. Every experiment retains all 32 codebooks; no new TTFA result or quality change is claimed for this rejected path.

## Implemented arithmetic

`mma4_projection.cu/.py` issues `mma.sync` with M8×N8×K32 and M16×N8×K32 shapes. It consumes original signed INT4 weight codes and decomposes each unchanged signed INT8 activation exactly:

`x = unsigned_low_nibble(x) + 16 * signed_high_nibble(x)`.

One signed-weight/unsigned-activation matrix instruction computes the low contribution; another signed/signed instruction computes the high contribution. INT32 addition reconstructs the original dot product before any floating conversion. Weights are reordered into contiguous A-register fragments, without expanding them to INT8 in application storage. The checked packing inverse recovers every original byte across all 36 layers.

The projection kernel stages group dots in shared memory and reproduces the selected G32 floating epilogue: local multiply/FMA order, descending warp-XOR reduction, BF16 boundaries and, for the down projection, the final two-warp sum. Gate/up includes SiLU, multiplication and the following grouped activation quantization. The benchmark supplies prequantized inputs and excludes producer normalization from both sides; these are projection-stage measurements.

## Validation and timing

The independent integer audit covers all 16 signed INT4 values and all 256 signed INT8 values at each of 32 positions, all positions together, row permutations, and 65,536 random multirow groups. Both matrix shapes pass **4,816,896 exact dots**, using an independent Torch INT32 product/sum reference and private-stream CUDA graphs. Memcheck repeats that audit with **zero errors**. An initial Python packing attempt failed because a singleton dimension retained a non-unit stride before a dtype view; flattening the register dimension fixes the view. The failed log is retained, and no incorrect GPU output was selected.

Thirty-two projection configurations vary matrix shape, four/eight warps and loop unrolling one/four. The timing ring rotates **36 actual layer weights, scales and input states**. Four timing rounds rotate/reverse option order; each uses nine event timings of three full rings. Four recorded inputs plus zero/spike at every layer give **6,912 checks**, all exact. Twenty-eight additional padded/actual-row cases cover N=1/7/8/15/16/17 for both K lengths and N=32/64 for fused gate/up, with private-stream graphs; memcheck reports **zero errors**.

| Stage | Selected DP4A µs | Best INT4-PTX µs | Best configuration |
|---|---:|---:|---|
| QKV | 5.948 | 26.530 | M16, 8 warps, unroll 1 |
| Attention output | 4.937 | 19.845 | M16, 8 warps, unroll 1 |
| Gate/up + output quantization | 16.895 | 92.237 | M16, 8 warps, unroll 4 |
| Down | 9.941 | 54.015 | M16, 8 warps, unroll 4 |

All failed performance candidates remain in `mma4_projection_v1.json`. No full-model or HTTP test follows because every candidate loses by a large margin. Existing selected serving kernels, codec precision and voice-cloning outputs are not replaced by this implementation.

## Machine-code evidence

The CUDA 12.8 build targets SM90 native cubins. `cuobjdump --dump-sass` shows each integer primitive containing four **INT8** IMMA instructions and conversion/bit-manipulation code for the two INT4 PTX operations. M8 lowers to `IMMA.8816.S8.S8` / `S8.U8`; M16 lowers to `IMMA.16816.S8.S8` / `S8.U8`. Static instruction counts are not dynamic executed counts. Register/spill details and exact native source snapshots are retained in `results/mma4_build/`.

The observation supports rejecting this compiled implementation on this H200; it is not a proof that every alternative tensor-core schedule is slow or that 50 ms is impossible. The prior INT8 MMA experiment and this INT4 PTX experiment have different layouts and costs, and neither should be described as a successful hardware optimization.

Instruction shapes and fragment layouts follow the [NVIDIA PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) and [CUTLASS's SM75 instruction definitions](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm75.h). The [August 2026 ISA/source-level audit](https://arxiv.org/abs/2608.11693) examines a different GPU and integer path, with no performance measurements. Its distinction between advertised formats, ISA support and usable implementation is relevant here; our H200 result comes from the compiled code and measurements above.

Run sequentially in `/workspace/MOSS-TTS`:

```bash
/venv/moss-vllm/bin/python -m optimization.validate_mma4_integer --tag reproduce
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_mma4_integer --tag reproduce_memcheck
/venv/moss-vllm/bin/python -m optimization.benchmark_mma4 --tag reproduce
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_mma4_memory --tag reproduce_memcheck
```

Artifacts: `mma4_integer_v2.json`, `mma4_integer_memcheck_v3.json`, `mma4_projection_v1.json`, `mma4_memory_memcheck_v1.json`, their logs, `mma4_sass_v1.txt`, and `mma4_machine_code_audit_v1.json`. The initial integer-only and later projection-enabled builds have separate source hashes. Source archive inclusion is recorded with the following short-scale pass.
