# Initial audio CUDA graph, retaining all 32 codebooks

The initial 32 audio steps now execute in one CUDA graph. Twenty alternating complete voice-cloning request pairs measure **79.505 → 77.910 ms median warm TTFA**, with **1.616 ms median paired gain** and every pair faster. All full PCM streams and final CUDA RNG states match the qualified calibrated G32/attention-PDL/clocked-codec control. **The 50 ms target remains unmet.**

These are H200 NVL batch-one measurements using the selected `/venv/moss-vllm` environment: PyTorch 2.13 + CUDA 13.0, Triton 3.7.1, Transformers 5.14.1 and cuDNN 9.20. Existing supervisor services, drivers and clocks are unchanged. GPU workloads run sequentially. The model, 32-codebook delay schedule and first playable 80-ms PCM chunk are unchanged.

## Implementation and limits

After `audio_start`, the sampler cannot emit `audio_end` until the 33rd audio step. `FirstAudioGraph` captures the preceding 32 complete LLM evaluations and full-shape stochastic samplers. A small Triton `_advance` kernel copies each sampled row into history and updates device position, audio length and delay state. This removes per-step Python dispatch and CPU inspection during that guaranteed initial interval.

The capture executes `_decode` directly, allowing PyTorch to track every RNG operation. It does not capture nested graph replays. Every masked audio channel still participates in the original full-shape sampler, preserving RNG consumption. With the existing audio-head optimization enabled, projection prefixes of 8, 16, 24 and 32 heads skip only outputs already masked by the delay schedule. Every emitted PCM frame still uses all 32 codebooks.

Four graphs cover context capacities 128/256/512/1024. The selected capacity must cover the entire interval, including requests crossing a previous per-step capacity boundary. Ordinary streaming resumes immediately afterward, emitting each subsequent full-codebook codec frame incrementally. Request state stays serialized; output history belongs to the graph and is copied before reuse. No utterance is buffered before yielding the first PCM chunk.

The feature defaults off and is available through `--first-audio-graph` in the benchmark, quality generator and server. It requires warmed custom decode attention. The selected configuration uses calibrated INT4/G32 projections, BF16 prefill and the exact FP32 clocked codec. Four captures add about **128 MiB live / 136 MiB reserved GPU allocation** and **3.13 seconds** of startup capture time in the paired process. These exclude the already existing graphs and model loading.

Individual wall-clock step times are unobservable inside the new graph. `step_ms` therefore contains 32 null entries for that interval, preserving index alignment; `first_audio_graph_ms` reports its actual elapsed wall time and capacity. Consumers must use that aggregate rather than averaging or replacing null entries with invented per-step timings.

## Paired complete-request measurements

One engine alternates the ordinary and graph paths for seeds 7000–7019, excluding one preceding warmup pair. All measured requests complete; profiling starts only after timing. Voice-reference codes are cached, and TTFA runs from complete text processing through the first CPU PCM chunk, excluding voice registration and network transit.

| Measurement | Ordinary initial steps | Initial audio graph |
|---|---:|---:|
| Median TTFA | 79.505 ms | 77.910 ms |
| p95 TTFA | 79.963 ms | 78.968 ms |
| Median preparation | 1.095 ms | 1.218 ms |
| Median prefill | 9.217 ms | 9.216 ms |
| Median initial 32 audio steps | 63.988 ms | 62.345 ms |
| Median first codec frame | 4.519 ms | 4.520 ms |

Median paired gain is **1.616 ms**, with **20/20 pairs faster**. All 40 measured PCM streams and both excluded warmup streams agree within their pairs, as do final RNG states. The twenty control PCM hashes also match the preceding codec-clock pass for the same seeds. Stage medians are separate statistics and need not sum to the TTFA median.

A preceding ten-pair interval-only pilot measures 63.793 → 62.299 ms, median paired gain 1.476 ms. That narrower measurement excludes prefill and codec work and is not reported as TTFA.

## Numerical, boundary and memory checks

The initial pilot compares every text/audio logit, sampled ID, final state and RNG state for three seeds with fresh-audio and immediate-delay starts: six intervals, 192 steps per path. All match exactly.

The complete boundary validator tests 19 starting positions with both delay states, including every captured capacity boundary, crossings and the final valid KV position. Across **38 intervals / 1,216 steps per path**, every text/audio logit, sampled ID, final RNG/state and all **72 entire KV buffers** match bitwise. Future cache slots are poisoned with NaNs, and both paths run on a private CUDA stream. Out-of-capacity intervals fail before state mutation.

Compute Sanitizer **memcheck reports zero errors**. Racecheck filtered to the changed `_advance` kernel reports **zero hazards, errors or warnings**. Each sanitizer runs two complete 32-step full-model intervals at position 145, with ordinary and immediate-delay states. This is changed-kernel memory/race coverage; the separate unsanitized validator supplies full boundary coverage, and filtered racecheck is not a claim about every unchanged library kernel.

The stream-loop regression loads the SHA-verified archived pre-change implementation and compares it with both current graph-disabled and graph-enabled paths on the same engine. Chinese and English cases at budgets 33/34/40/400 produce identical complete PCM, frame counts, truncation flags, prompt/step counts and final RNG state across all three paths. The shortest valid request yields exactly one full-codebook frame. Both current paths reject too-short and oversized budgets, preserve the original owner when an overlapping request is rejected, release ownership on generator close, and recover with identical first PCM. Previously yielded PCM storage stays unchanged.

An independent five-request benchmark through the ordinary CLI, without audio-head buckets, measures **78.607 ms median / 78.951 ms p95**. Its complete 36-step diagnostic dictionary and saved WAV match the preceding codec-clock benchmark. Every request completes. This checks flag integration and full-head capture; its separate-process timing is not another isolated gain estimate.

## Voice-cloning quality

All **48 generated bilingual WAVs**, **eight reference WAVs** and generation metadata match the preceding codec-clock suites byte for byte. All requests complete without truncation or nonfinite audio. ASR is not rerun on identical files. Existing normalized Chinese CER **5.37% / 1.18%**, English WER **0% / 0%**, and speaker cosine **0.9143 / 0.9149** apply to the original and expanded suites, respectively.

This is exact equivalence to the selected calibrated G32 path. It does not claim equivalence to the original BF16 checkpoint or broad human quality acceptance; the four reference voices and reused machine-scored prompts remain limited diagnostics.

## HTTP streaming

Sequential temporary loopback servers measure **83.116 → 81.538 ms median TTFA**, with p95 **83.310 → 82.291 ms**, over twenty complete cached-voice requests per implementation. Every full signed-16-bit PCM stream and frame count matches. Both servers pass malformed-input checks and cancellation/recovery, returning 429 while occupied and then 200. Both temporary servers terminate after their tests.

Internal HTTP-engine medians are 79.450 → 77.749 ms. The initial audio interval falls from 64.040 to 62.366 ms; prefill remains 9.220 / 9.218 ms and first codec decoding 4.514 / 4.520 ms. These separate-process HTTP measurements support integration; the alternating in-process comparison is the better estimate of isolated gain.

One contiguous fresh-reference registration plus synthesis observation measures **122.020 → 120.408 ms**; registration alone takes 35.338 / 35.473 ms. These are single observations, not latency distributions. They include server waveform parsing, reference encoding and two loopback requests, with client WAV/base64 preparation outside the timer. The request-to-audio timer ends only after a complete 3,840-byte / 80-ms PCM chunk arrives.

## Remaining bottleneck

The post-timing profile covers 33 decode steps and two codec chunks in a deliberately truncated 34-step diagnostic request. CUDA graph launch calls fall from 36 to 5, async copy calls from 151 to 27, and kernel events from 14,060 to 13,937. Projection and attention event counts remain 4,752 and 3,564: the change removes scheduling/state overhead while preserving model computations.

The selected trace contains 39.822 ms with projections alone, 9.519 ms with projections and LLM attention, and 6.729 ms with LLM attention alone. Other kernel intervals total 24.167 ms. These resident intervals include PDL dependency waits; they are not utilization or pure arithmetic percentages. The separate traces are not a paired latency experiment, and profiler gaps must not be used to infer a measured TTFA gain.

The initial audio interval still takes about **62.35 ms**, compared with **9.22 ms prefill** and **4.52 ms codec**. Projection work therefore remains the main target for a larger gain. The approximately 27.9-ms remaining gap cannot be closed by this scheduling change alone. These measurements do not establish an absolute 50-ms lower bound, and no codebook reduction is introduced.

## References and reproduction

The [CUDA graphs documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) and [PyTorch CUDA notes](https://docs.pytorch.org/docs/main/notes/cuda.html) describe graph execution and capture constraints. This implementation measures actual full-request behavior and independently verifies RNG state; documentation is not performance evidence for MOSS.

Run GPU jobs sequentially with fresh tags:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_first_audio_graph --tag reproduce_pilot
/venv/moss-vllm/bin/python -m optimization.benchmark_first_audio_paired --tag reproduce --pairs 20
/venv/moss-vllm/bin/python -m optimization.validate_first_audio_graph --tag reproduce_boundary
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_first_audio_graph --tag reproduce_memcheck --short
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --kernel-name kns=_advance --error-exitcode 1 \
  /venv/moss-vllm/bin/python -m optimization.validate_first_audio_graph --tag reproduce_racecheck --short
/venv/moss-vllm/bin/python -m optimization.validate_first_audio_stream --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_first_audio_http --tag reproduce
```

For the selected temporary loopback serving configuration:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant \
  --scaled-dp4a --norm-projection --audio-head-buckets --short-scales \
  --projection-pdl --attention-pdl --codec-clock --first-audio-graph
```

Persistent services use the instance's supervisor conventions. This pass uses temporary local validation servers and stops them afterward.

## Evidence

- [Complete paired requests](results/first_audio_paired_v1.json) and [stage/capture summary](results/first_audio_timing_summary_v1.json).
- [Pilot interval audit](results/first_audio_graph_pilot_v1.json), [boundary validation](results/first_audio_validation_boundaries_v1.json), and the `first_audio_validation_memcheck_v1` / `first_audio_validation_racecheck_v1` JSON/log pairs.
- [Previous-pass PCM equivalence](results/first_audio_previous_pcm_equivalence_v1.json).
- [Archived stream-loop and cancellation regression](results/first_audio_stream_validation_v1.json).
- [Independent CLI regression](results/first_audio_regression_summary_v1.json).
- [All 48 cloning WAVs and eight reference WAVs](results/quality_suite/first_audio_equivalence.json).
- [HTTP comparison](results/http_first_audio_comparison_v1.json) and [HTTP stage summary](results/http_first_audio_stage_summary_v1.json), with individual request, health and server logs retained.
- [Profile summary](results/first_audio_profile_summary_v1.json); full trace and table remain alongside it.
- [Static, CLI and service checks](results/first_audio_static_checks_v1.json).
- [Source archive](results/first_audio_sources.tar.gz), [per-file hashes and round-trip verification](results/first_audio_source_hashes.json), and [pass index](results/first_audio_pass_summary_v1.json). Measurement logs, traces, audio and model weights remain separate from the source archive.
