# Projection resource budgets and parallel warp partitions

The preceding codec pass was progress: it qualified an exact optional codec improvement. This continuation tests the remaining dominant LLM projections while retaining **all 32 codebooks, streaming voice cloning, calibrated G32 weights, BF16 prefill and the FP32 codec**. Neither new experiment improves the selected implementation, so neither is installed in serving. The prior qualified **79.53 ms warm cached-voice TTFA** remains the latest result, not a new measurement from this pass. **50 ms remains unmet.**

## CUDA 13 register and shared-memory experiments

The selected fused gate/up kernel uses 162 registers per thread with four warps and no spills. Its theoretical occupancy is three CTAs per SM. The selected QKV and down kernels use 56 and 128 registers. Under programmatic dependent launch, resource use can affect which neighboring kernels can be resident together, so register limits are tested on dependent chains rather than isolated projections.

CUDA 13 adds opt-in shared-memory register spilling. NVIDIA documents that it requires static shared memory, excludes register reallocation with `setmaxnreg`, and can still fall back to local memory. See the [NVIDIA implementation discussion](https://developer.nvidia.com/blog/?p=105101) and [PTX 9.0 specification](https://docs.nvidia.com/cuda/archive/13.0.0/pdf/ptx_isa_9.0.pdf). It does not guarantee a speedup for a kernel that originally had no spills.

`ptx_resources.py` takes the selected Triton 3.7.1 PTX, retains its instruction body, replaces the dynamic shared declaration with an equal-sized static declaration, optionally adds a register limit and the spilling pragma, and assembles an SM90a cubin. A separate CUDA 13.0.88 assembler is extracted from a pinned, checksum-verified NVIDIA wheel. It matches the host's advertised CUDA major/minor capability; no driver, existing toolkit, Python environment or service is changed. CUDA 13.4 tools already present in another package are not selected. A CUDA Driver API launcher uses the caller's stream and preserves the programmatic dependency attribute. Original/transformed PTX, assembler logs, cubins and resource hashes are retained.

Twenty-nine configurations cover the ordinary control, CUDA 13 reassembly with dynamic/static shared memory, and register caps with local/shared spilling on QKV, gate/up and down. The benchmark rotates all 36 actual weight/input layers through MLP → down → next QKV, validates three saved inputs per layer, and adds private-stream CUDA graph checks. Every captured chain has the expected two programmatic edges. The synthetic ring supplies a preceding MLP residual even at layer-number wrap; it is an operator-chain screen, not a full model trajectory.

All **3,161 comparisons**, including control checks, match exactly. Six rotated/reversed timing rounds give:

| Chain variant | Median µs | Relevant result |
|---|---:|---|
| Selected control | 35.988 | No register spills |
| CUDA 13, dynamic shared | 36.036 | No useful gain |
| CUDA 13, static shared | 36.056 | No useful gain |
| Gate/up capped at 128, local spills | 44.269 | 304 local bytes/thread |
| Gate/up capped at 128, shared spills | 43.082 | 9,216 static shared bytes; 240 local bytes/thread |
| Down capped at 112, local spills | 37.941 | 40 local bytes/thread |
| Down capped at 112, shared spills | 36.132 | 3,584 static shared bytes; no local bytes |
| QKV capped at 48, shared spills | 36.786 | 4,608 static shared bytes; no local bytes |

Gate/up's 128-register cap increases theoretical occupancy from three to four CTAs per SM, but added spill traffic dominates. Shared spilling improves several capped kernels relative to their local-spill counterparts, while **none beats the original spill-free chain**. Down at 112 registers still admits eight CTAs per SM, the same as its 128-register control. The driver occupancy query measures a resource limit, not observed concurrent execution or GPU utilization. These timings are not TTFA.

## Parallel gate/up warp partitions

`dp4a_gateup_warp_specialized.py` implements two Gluon variants. One assigns gate and up projections to independent four-warp partitions. The other assigns half of the 32 output rows to each partition. Each retains the qualified four-warp normalization/reduction order. An explicit shared-memory buffer and mbarrier join the partitions before the final SiLU/product and G32 quantization. Only the default partition writes the residual, avoiding duplicate cross-partition writes. Programmatic waits precede producer-data reads.

The [Gluon warp-specialization tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html) describes the synchronization and register-allocation tradeoffs. The [Triton compiler roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) and [Twill scheduling paper](https://arxiv.org/abs/2512.18134) motivate measuring scheduling/resource combinations; their tensor-core/attention examples do not establish gains for this DP4A GEMV workload. No third-party kernel body is copied.

Sixteen partition/register configurations plus a control pass **1,853 comparisons**, including private-stream graph checks. All outputs, returned residuals and quantized buffers are exact. The best parallel-row chain is **39.657 µs**, versus **36.089 µs** for its contemporaneous control; the best parallel-branch chain is **43.424 µs**. The selected examples use eight total warps, 128/192 reported registers, 2,092/2,124 shared bytes and no spills. Additional parallelism does not offset its resource and communication costs here. No variant is promoted.

The first pilot fails compilation because separate scalar specialization arguments are not retained correctly through the warp-specialized call. Passing them as one constexpr configuration fixes it. The next pilot exposes use of a nonexistent Gluon `trans` alias; using its supported `permute` operation fixes that. Failed logs and both original source snapshots remain. The final 36-layer sweep has no compilation failures or numerical mismatches.

## Validation, scope and next target

Together, the two completed sweeps cover **44 alternatives plus two controls**, with **5,014 exact chain/graph comparisons** (4,796 candidate and 218 control comparisons). They supply evidence to reject these schedules, not a request-level speedup or a proof that 50 ms is impossible. The selected model kernels and streaming path are unchanged; rerunning ASR on unchanged serving code would not qualify these rejected experiments.

The dedicated validator captures six configurations on private streams and replays each graph with three changing inputs from layers 0, 17 and 35. All **54 cases** match every returned residual, projection output and quantized buffer. **Memcheck reports zero errors; racecheck reports zero hazards, errors or warnings.** Both runs complete. This covers representative register-spilling and both warp-partition paths, not every register configuration or an entire serving trajectory.

The next concrete latency target is the host/device synchronization in the initial full-codebook delay schedule. `StreamingTTS._stream` currently reads a text token back to the CPU after every LLM step, then updates the next graph's state from the host. Once `audio_start` has been selected, the first complete frame requires 32 audio steps. A bounded GPU graph for that initial interval may remove host round trips while preserving all codebooks. This is an implementation candidate, not a measured gain; its RNG sequence, early-delay behavior, context boundaries, cancellation and complete streams still require validation before use.

## Reproduction and artifacts

Run GPU jobs sequentially in `/workspace/MOSS-TTS` with `/venv/moss-vllm`; use new result tags. The compiler helper verifies the existing file or extracts only the pinned assembler, without a package installation.

```bash
python3 -m optimization.prepare_ptxas
/venv/moss-vllm/bin/python -m optimization.benchmark_ptx_resources --tag reproduce --layers 36 --rounds 6
/venv/moss-vllm/bin/python -m optimization.benchmark_gateup_warp_specialized --tag reproduce --layers 36 --rounds 6
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_projection_resources --tag reproduce_memcheck
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_projection_resources --tag reproduce_racecheck
```

- [CUDA resource sweep](results/ptx_resources_ring_v1.json) and [warp partition sweep](results/gateup_ws_ring_v1.json), each with six timing rounds, validation records and resource data.
- [Compact resource summary](results/projection_resources_summary_v1.json), including assembler provenance and original/transformed PTX/cubin hashes. Detailed builds are in `results/ptx_resource_build/`.
- `results/gateup_ws_ring_v1_*.ptx/.ttgir` retain the actual final Gluon code generation; initial failed pilots and source snapshots are separate.
- `results/projection_resources_validation_{memcheck,racecheck}_v1.json/.log` contain the completed 54-case sanitizer runs.
- [Static/integration checks](results/projection_resources_static_checks_v1.json) verify all 209 Python source files parse, four new CLI help commands succeed, and every prior runtime source retains its checksum. Both original services remain ready with 32 codebooks; no extra GPU process or temporary endpoint remains.
- [Source archive](results/projection_resources_sources.tar.gz) and [per-file hashes](results/projection_resources_source_hashes.json) cover 264 source/documentation/configuration files, including licenses and prior calibration/packing plans. Compiler binaries, measurement traces, audio and model weights remain separate. [Pass summary](results/projection_resources_pass_summary_v1.json) indexes the evidence and limitations.
