# Projection wait audit and asynchronous weight staging

All experiments retain **32 acoustic codebooks**, streaming and voice cloning. A sixteen-row attention-output tile that preloads immutable weights into registers before the dependency wait reaches **74.254 ms median warm cached-voice TTFA**, with **0.605 ms median paired gain** in a twenty-round comparison. The 50-ms target is still unmet. None of the diagnostic timestamps below are end-to-end TTFA.

## Measuring waits in the actual model

Earlier profiler kernel intervals include programmatic dependency waits. Added experimental SM90 `%globaltimer` probes before/after those waits and, optionally, at kernel entry and after lane-zero output-store issue. They preserve the selected arithmetic and retain the original unconditional dependency waits. A compact variant reads both wait timestamps in one inline PTX block and stores them after the wait.

Initial seven-configuration pilots show that instrumentation changes register pressure and timing substantially. The compact form does not eliminate that effect. The subsequent 14-configuration, 36-layer screen passes **1,512 bitwise intermediate/output comparisons** and 14 private-graph checks. Sparse gate/up wait-only probes add about **0.003 µs** per synthetic three-projection chain; sparse QKV full probes add **0.111 µs**. Sparse down wait-only probes add **1.112 µs**, and dense full gate/up probes add **4.745 µs**. Probe overhead is measured, not subtracted from measurements to invent an uninstrumented execution time.

The isolated ring initially shows a 1.024-µs median gate/up wait. That dependency is synthetic: its preceding kernel is the previous ring entry's QKV, not the model's actual attention-output projection. Therefore a separate full-model experiment captures the actual first 32 audio steps with independent control and probe graphs.

Eight rotating/reversing rounds measure the uninstrumented first-audio graph at **61.765 ms median**. Full cache comparison once per configuration preserves all **72 KV buffers**, IDs, final text/audio logits, status and RNG state. Every timing round also preserves IDs, final logits, status and RNG. The sampled observations are:

| Probe in actual model | Samples | Median wait µs | p95 wait µs | Maximum wait µs | Median paired graph overhead ms |
|---|---:|---:|---:|---:|---:|
| Gate/up, wait only | 27,648 | 0.000 | 0.032 | 0.032 | 0.059 |
| QKV, full span | 55,296 | 0.000 | 0.032 | 0.064 | 0.109 |
| Down, wait only | 73,728 | 0.352 | 1.280 | 18.112 | 1.382 |

QKV's post-wait lane interval is **6.368 µs median / 7.232 µs p95**. This supports targeting its post-wait work rather than assuming a long dependency stall. The down projection has a long wait tail, but its probe is notably intrusive. No hardware bandwidth/utilization ceiling is inferred from these observations.

The timer has target-specific semantics; NVIDIA describes it as intended for its tooling. Observed values have 32-ns granularity, so a reported zero wait does not prove zero waiting. SASS shows `CS2R` timer reads execute even for unsampled CTAs, while stores are sampled. The final timestamp marks one lane's output-store issue, not completion of all threads or visibility of the entire output. Only one frozen cloned-voice prompt and its initial 32 steps are instrumented. PTX, compiler IR, cubins, SASS and raw timestamp tensors are retained.

## Exact G32 asynchronous copies

`dp4a_async_weights.py` stages immutable packed weights before the existing PDL wait. Activation reads remain after that wait, and shared-memory consumption waits for copy completion. It reuses the selected DP4A arithmetic and FP32 reduction layout without changing quantization values.

- Cooperative `cp.async` copies test shared-memory swizzles, requested cache modifiers, and optional existing prefix hints.
- Hopper TMA copies use tensor descriptors and transaction barriers. Full down-projection tiles include power-of-two padding.
- Compact down tiles use three TMA transactions and add the zero fourth segment after loading, reducing shared-memory allocation.
- Partial tiles stage the first third of actual down weights with TMA and load the remainder normally after the dependency wait.

The 37-configuration down-chain screen passes **3,996 bitwise comparisons** and 37 private-graph checks. All variants are slower over six rotated/reversed all-layer rounds:

| Down-chain schedule | Median µs | Median paired gain µs |
|---|---:|---:|
| Selected prefix-hint control | **35.409** | — |
| Best partial TMA, eight rows/four warps | 36.201 | −0.783 |
| Partial TMA, four rows/two warps | 36.498 | −1.100 |
| Compact TMA, eight rows/four warps | 36.535 | −1.147 |
| Compact TMA, four rows/two warps | 37.286 | −1.877 |
| Full TMA, four rows/two warps | 39.002 | −3.592 |

Representative down resource counts explain a tradeoff, not a complete performance model: control uses 128 registers and 1 KiB shared memory; cooperative full staging uses 122 registers and 32 KiB; unswizzled full TMA uses 117 registers and 32,776 bytes; compact TMA uses 117 registers and 24,584 bytes; partial TMA uses 122 registers and 8,200 bytes. These variants have zero register spills. Fewer registers alone do not predict a faster kernel.

## Attention-output copies and producer overlap

The standalone attention-output screen tests 19 configurations, passes **4,104 bitwise comparisons** against the selected control using four real inputs plus zero/spike per layer, and checks 19 private graphs. All alternatives have slower medians. Control is **5.056 µs**; the closest cooperative copy, 16 rows/four warps, is **5.083 µs**, with a **−0.023-µs** paired gain.

Standalone timing excludes a potential benefit: immutable weights may overlap with the preceding attention producer. A separate seven-configuration screen includes selected norm/QKV hints, QK/RoPE, native attention, reduction/quantization and output projection. All **756 intermediate/output comparisons** and seven four-edge PDL graphs match. Frozen attention fixtures use the nearest captured layer 0/17/35 and do not constitute a full model trajectory.

That five-kernel ring measures **19.549 µs** for control and **18.959 µs** for cooperative swizzle-eight/eight-row staging, with **0.422 µs median paired gain**. The 16-row version gains 0.199 µs. Timing varies between rounds, so these candidates proceed to full streamed-request measurement rather than being selected from the microbenchmark.

## Complete streamed requests

Complete-model tests use independent context/head/first-audio CUDA graph sets and restore eager dispatch together with the selected graph set. BF16 prefill, G32 weights/activations, codec precision and all 32 codebooks are shared. One warmup round is excluded; configurations rotate/reverse with matching seeds.

| Run | Configuration | Median TTFA ms | p95 ms | Median paired gain ms | Faster rounds |
|---|---|---:|---:|---:|---:|
| Twelve rounds | Control | 74.816 | 75.848 | — | — |
| Twelve rounds | Eight-row cooperative copy | **74.438** | 75.608 | **0.330** | 11/12 |
| Twelve rounds | Sixteen-row cooperative copy | 74.629 | 75.811 | 0.192 | 8/12 |
| Twenty-round repeat | Control | 74.767 | 76.140 | — | — |
| Twenty-round repeat | Eight-row cooperative copy | **74.461** | 76.065 | **0.287** | 16/20 |

All **76 measured complete PCM streams** and final RNG states match within their respective rounds; all 32 controls also match the preceding qualified preset's saved hashes. The two runs add **144 private-stream comparisons** between selected-control eager decode and candidate graphs, covering context boundaries through position 1023 and multiple audio-head counts. The repeat includes an **84.110-ms candidate outlier**; p95 alone does not convey this maximum. This is a small repeatable median benefit with host/runtime variability, not uniform latency improvement.

The repeated run's initial 32-step LLM interval falls from **61.737 to 61.456 ms**; prefill is unchanged at about **6.83 ms** and first codec decode about **4.52 ms**. The sixteen-row candidate is not selected over the better eight-row candidate.

### Isolating the tile and finding a better schedule

A second seven-configuration attention-chain screen passes another **756 bitwise comparisons** and seven PDL graph checks. Increasing the tile alone is slower: eight/sixteen-row ordinary projections with scale-only preload regress by **0.602 / 0.586 µs** paired. Preloading both immutable weights and scales into registers before the PDL wait improves the same tiles by **0.465 / 0.496 µs**. Shared-memory asynchronous copy with eight rows gains 0.321 µs in that run. Thus the tile alone does not explain the benefit, and register preloading merits a complete-model comparison.

A twelve-round four-way run has appreciable host/runtime variation: control is 75.796 ms, shared-copy eight-row 75.435 ms, register eight-row 75.261 ms and register sixteen-row 75.088 ms. Median paired gains are **0.361 / 0.493 / 0.821 ms**; the sixteen-row register schedule wins 11/12 pairs. All 48 complete PCM streams and final RNG states match, and 144 additional control-eager/candidate-graph comparisons pass.

A twenty-pair repeat confirms the sixteen-row register schedule:

| Configuration | Median TTFA ms | p95 ms | Median first-32 LLM interval ms |
|---|---:|---:|---:|
| Previous selected control | 74.809 | 75.779 | 61.783 |
| Register preload, sixteen rows/four warps | **74.254** | **75.368** | **61.194** |

Median paired gain is **0.605 ms**, with **19/20** improvements. All forty complete streams and final RNG states match, every control matches prior saved PCM, and 48 additional private-graph comparisons pass. Prefill and codec remain about 6.83 and 4.53 ms. This brings the full pass to **164 measured complete-stream comparisons** and **336 context/head graph comparisons**, excluding warmup requests.

`--output-weight-prefetch` selects this register-preload schedule before capture. `--async-output` retains the slower shared-copy experiment; the flags are mutually exclusive. Neither changes G32 values or the codec, and neither is enabled by default. Integration uses a per-module projection callable, with original graph/eager behavior retained whenever neither flag is set.

## Memory and compiled-code checks

Unfiltered memcheck and racecheck each pass **144 changed-input private-stream cases**: five async down-chain schedules, three compact timing-probe schedules, and four attention-output variants at both normal and odd output-row counts, across three real layers and real/zero/spike inputs. Every output matches selected G32 bits; sampled timestamps are ordered. Memcheck reports **zero errors**; racecheck reports **zero hazards, errors or warnings**. This is operator/chain coverage, not whole-service sanitizer coverage.

Ten compiled kernels are archived with PTX, Gluon IR, cubins, SASS and SHA-256 hashes. Copies precede the original dependency wait; async completion precedes shared-memory consumption. Audited cooperative copies lower to `cp.async.cg` even when the source requests `.ca`; requested cache options do not necessarily produce distinct instructions. TMA variants retain transaction-barrier operations. Source configuration labels must not be interpreted as proof of different machine code.

The later register-preload validator passes another **36 cases under each sanitizer**, covering both eight/sixteen-row tiles, changing graph inputs and odd output-row counts. Again there are zero memory errors or race hazards/warnings. Its dedicated twelve-kernel audit includes the final register schedule: **96 registers, 2 KiB shared memory, zero spills**, with immutable weight/scale loads before `ACQBULK` and activation loads after it. The eight-row register alternative uses 48 registers; the shared-copy eight-row alternative uses 40 registers and 16 KiB shared memory.

## Cloning quality and remaining bottleneck

Both the shared-copy candidate and the final register-preload candidate generate **48 bilingual cloning utterances each**, using the same four reference assets, prompts and seeds as the previous qualified G32 suites. All 96 generated WAVs and sixteen reference copies match their corresponding prior files byte for byte, as do frame counts and generation metadata. All utterances finish. The selected register schedule therefore inherits the previous identical-audio diagnostic results: original/expanded Chinese CER **5.37% / 1.18%**, English WER **0% / 0%**, and speaker cosine **0.9143 / 0.9149**. No ASR rerun or new human-listening claim is made.

Post-timing profiling of the final register schedule retains 13,289 kernel events, including 4,752 projection and 3,564 attention events. Projection-only resident intervals total **38.207 ms**, projection/attention overlap **10.607 ms**, and attention-only intervals **6.743 ms**. These include waits and are not pure compute or utilization percentages. Prefill still uses 375 launches. The initial LLM interval remains the dominant measured stage at **61.19 ms**; 50-ms end-to-end TTFA is not achieved and no codebook reduction is justified.

A concrete next experiment is an exact QKV projection/attention-preparation schedule: the sampled QKV post-wait interval is about 6.37 µs and its actual producer wait is short. Any proposed fusion must retain normalization rounding, KV writes and the CUDA dependency contract, then pass all-layer checks before full-model timing. Neither the sampled timer nor the current results prove a hardware latency lower bound.

## HTTP and entry-point validation

The initial shared-copy HTTP run passes all output, input-error and cancellation checks but **regresses** from 78.939 to 79.390 ms median. CPU CLI import/help checks overlapped part of that test, so it is retained as integration evidence with a confound noted; it is not a speedup claim.

The final register-preload HTTP comparison runs sequential temporary servers without concurrent qualification/import jobs. Median TTFA is **80.201 → 79.348 ms**, p95 **81.149 → 81.100 ms**. All twenty complete PCM streams match and malformed-input/cancellation/recovery checks pass, including the expected 429 then 200 recovery sequence. Internal engine medians are **75.951 → 75.193 ms** and initial-LLM intervals **61.858 → 61.261 ms**. Host/runtime variation contributes to the HTTP difference; the paired in-process experiment is the cleaner estimate of the kernel gain. This HTTP median is also higher than the earlier, separate prefill-pointwise run's 78.438 ms; no cross-session HTTP improvement is claimed.

One fresh-reference registration-plus-synthesis observation is **123.389 → 119.961 ms**. Registration alone varies **39.834 → 37.620 ms**, although this change does not modify the reference encoder. These single observations do not establish a reference-encoding speedup. Both temporary servers stop, and port 18084 is free afterward. Both original supervisor services remain ready with 32 codebooks.

The ordinary benchmark CLI without audio-head buckets completes five requests at **72.768 ms median / 74.061 ms p95**. Its entire 36-step diagnostic dictionary and saved WAV match the preceding selected preset. This separate entry-point run is not substituted for the paired **74.254-ms** result or presented as another isolated gain. Six updated entry-point help checks and all 275 Python files parse successfully.

## Reproduction and sources

Run GPU jobs sequentially in `/workspace/MOSS-TTS`, using `/venv/moss-vllm/bin/python`. Each command requires a new result tag and refuses to overwrite its JSON evidence.

```bash
python -m optimization.benchmark_projection_timestamps --tag NEW --layers 36 --rounds 6 --compact --targeted
python -m optimization.profile_projection_waits --tag NEW --rounds 8
python -m optimization.benchmark_async_weights --tag NEW --layers 36 --rounds 6
python -m optimization.benchmark_async_output --tag NEW --rounds 6
python -m optimization.benchmark_async_attention --tag NEW --rounds 6
python -m optimization.benchmark_async_output_paired --tag NEW --rounds 12
python -m optimization.benchmark_async_attention --tag NEW_ABLATION --rounds 6 --ablation
python -m optimization.benchmark_async_output_paired --tag NEW_REGISTER --rounds 20 --configs register_r16 --profile
compute-sanitizer --tool memcheck --error-exitcode 1 python -m optimization.validate_async_weights --tag NEW_MEM
compute-sanitizer --tool racecheck --error-exitcode 1 python -m optimization.validate_async_weights --tag NEW_RACE
compute-sanitizer --tool memcheck --error-exitcode 1 python -m optimization.validate_output_preload --tag NEW_REGISTER_MEM
compute-sanitizer --tool racecheck --error-exitcode 1 python -m optimization.validate_output_preload --tag NEW_REGISTER_RACE
python -m optimization.audit_async_weights --tag NEW
python -m optimization.benchmark_async_output_http --tag NEW --strategy register
```

Selected local serving command (the benchmark's temporary server is stopped afterward):

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant \
  --scaled-dp4a --norm-projection --audio-head-buckets --short-scales \
  --projection-pdl --attention-pdl --codec-clock --first-audio-graph \
  --bulk-prefetch --prefill-qkv --prefill-pointwise --output-weight-prefetch
```

The shared-copy experiment substitutes `--async-output` for the last flag. Persistent service setup follows the instance's supervisor guide; neither existing supervisor service is replaced by these tests.

Implementation references: [NVIDIA PTX globaltimer semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi), [Triton Gluon asynchronous copy tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html), and [Gluon TMA tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html). The installed Triton 3.7.1 API was inspected directly; newer tutorial API spellings are not blindly copied.
