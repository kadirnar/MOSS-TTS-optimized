# Measured improvements

MOSS-TTS-v1.5 8B Delay, one H200 NVL, batch one, streaming voice cloning, **all 32 acoustic codebooks**. The latest qualified optional preset measures **71.86 ms engine TTFA**, **75.02 ms loopback HTTP TTFA** with a registered voice, and **110.67 ms fresh voice registration plus synthesis**. **50 ms has not been reached.**

Engine TTFA includes complete-text processing, prefill, all required delay steps, FP32 codec decoding and the first 80-ms PCM chunk on CPU. Cached-voice measurements exclude reference encoding; fresh-reference and HTTP results have separate boundaries. Historical milestones are not one paired comparison. Per-change paired gains cannot be added: configurations and runtime phases differ.

| Improvement | Measured result | Evidence / qualification |
|---|---|---|
| Static KV caches, decode CUDA graphs, fused QKV and custom Triton GEMV/attention | LLM step approximately 21.9 → 4.784 ms; initial complete pipeline 848.52 → 175.22 ms TTFA | [Initial report](REPORT.md); BF16 model differs numerically from upstream |
| FP32 streaming codec graphs, projected LFQ tables, fused RoPE/cache/attention, multi-tensor reset | Approximately 46 → 4.83 ms per PCM chunk | [Initial report](REPORT.md); 140-frame comparison, 115.95 dB SNR |
| CUDA graph buckets for cloned-voice reference encoding | 69.07 → 20.89 ms encoder latency | [Initial report](REPORT.md); identical tokens for eight tested durations |
| Residual/RMSNorm and gate/up/SiLU fusion | BF16 TTFA 175.68 → 170.96 ms | [Review](REPORT_REVIEW.md); 16 diagnostic WAVs byte-identical to preceding custom BF16 |
| FP8 backbone plus fused decode kernels | 118.14 ms engine / 122.17 ms HTTP TTFA | [Review](REPORT_REVIEW.md); experimental, Chinese quality diagnostic regresses |
| GPTQ INT4/G32 calibration, grouped activation DP4A, normalization fusion and prefill buckets | 106.43 ms engine TTFA milestone | [Calibration](REPORT_CALIBRATED.md); quantized path, not upstream BF16 equivalence |
| Interleaved INT4 packing and bounded 128/256/512-token decode graphs | 106.43 → 98.69 ms historical engine medians | [Packing](REPORT_PACKED.md); matching-layout path preserves diagnostic WAVs |
| Direct packed activation loads in projection kernels | 94.04 ms engine TTFA milestone | [Epilogue](REPORT_EPILOGUE.md); matching arithmetic retained |
| Attention reduction fused with G32 activation quantization | 92.35 ms engine TTFA milestone | [Epilogue](REPORT_EPILOGUE.md); exact recorded outputs |
| Fixed-order vectorized native CUDA attention | **1.71 ms paired TTFA gain** | [Native attention](REPORT_NATIVE_ATTENTION.md) |
| Gate/up projection, SiLU and output quantizer fusion | **1.51 ms paired gain** | [Gate/up quantizer](REPORT_GATEUP_QUANT.md) |
| Scaled signed-integer DP4A unpacking | **1.23 ms paired gain** | [Scaled DP4A](REPORT_SCALED_DP4A.md); weight precision unchanged |
| Fuse normalization/activation quantization into consuming projections | **1.77 ms paired gain** | [Normalization fusion](REPORT_NORM_PROJECTION.md) |
| Skip output heads masked by the existing delay schedule | **0.509 ms paired gain** | [Audio heads](REPORT_AUDIO_HEADS.md); every emitted frame still has 32 codebooks |
| Store exactly representable projection scales in BF16 | **0.44–0.52 ms paired gains** | [Short scales](REPORT_SHORT_SCALES.md); not additional scale quantization |
| CUDA programmatic dependent launch for projections | **0.64–0.76 ms paired gains** | [Projection overlap](REPORT_PROJECTION_PDL.md); producer waits retained |
| Extend dependency overlap through attention | **1.86–4.00 ms paired gains** across runtime phases | [Attention overlap](REPORT_ATTENTION_PDL.md); phase-dependent, not additive |
| Shared codec clocks and first-frame attention specialization | **0.301 ms paired TTFA gain**; first codec frame 4.769 → 4.517 ms | [Codec clocks](REPORT_CODEC_CLOCK.md); exact waveform/cache checks |
| Capture all first 32 audio steps in one CUDA graph | **1.616 ms paired gain** | [Initial audio graph](REPORT_FIRST_AUDIO_GRAPH.md); full sampling/RNG preserved |
| Hopper asynchronous bulk L2 weight hints | **0.446–0.483 ms paired gains** | [Bulk prefetch](REPORT_BULK_PREFETCH.md) |
| BF16 prefill QKV fusion and prefetch address specialization | **1.787 ms** fusion and **0.188 ms** address paired gains | [Prefill QKV](REPORT_PREFILL_QKV.md); separate comparisons |
| BF16 prefill SiLU/product and residual/normalization fusion | **0.644 ms paired gain** | [Prefill pointwise](REPORT_PREFILL_POINTWISE.md) |
| Preload attention-output weights into registers | **0.605 ms paired gain** | [Weight staging](REPORT_ASYNC_WEIGHTS.md) |
| Clustered QKV projection/head normalization/RoPE/cache writes | Approximately **0.11 ms paired gain** | [QKV clusters](REPORT_QKV_CLUSTER.md); eight-CTA Hopper kernel |
| Eight-row, four-warp down-projection tile | **0.447 ms paired gain** | [Down tile](REPORT_DOWN_TILE.md); p95 slightly regresses |
| Isolated Triton 3.8 exact gate/up compilation | **0.238 ms final paired gain** | [Gate/up compiler](REPORT_GATEUP_COMPILER.md); selected host stays on Triton 3.7.1 |
| Load historical K/V before the retained attention dependency wait | **1.685 ms paired engine gain**, reaching **71.856 ms**; **1.919 ms paired HTTP gain**, reaching **75.022 ms** | [Historical KV](REPORT_ATTENTION_HISTORY.md); 48 cloning WAVs unchanged |
| Streaming API and voice lifecycle | Incremental 24-kHz mono PCM, registration, validation, busy admission and cancellation recovery | [API and commands](README.md); batch-one ownership, no generated-audio cache |

The latest preset preserves all 48 bilingual cloning WAVs relative to its preceding calibrated G32 implementation. This does not establish equivalence to upstream BF16 or broad human quality acceptance. New options default off; the two original supervisor services remain on their previously deployed configurations.

## Trials not selected as improvements

| Trial | Result |
|---|---|
| SGLang-Omni 0.1.6 / SGLang 0.5.19 | 423.39 ms full streaming HTTP median; [different workload boundary](REPORT_REVIEW.md) |
| Adapted vLLM-Omni / vLLM 0.28 | 367.54 ms full streaming HTTP median; same boundary qualification |
| CUDA/C, TileLang and CuTe DSL elementwise alternatives | Implemented and checked; selected Triton SiLU was fastest in that screen, [1.461 µs](REPORT.md) |
| G64/G128 quantization, newer global runtime, several tensor-core and speculative alternatives | Speed or quality regressions; retained as documented experiments |
| Latest cooperative MLP fusion | 32 runnable schedules all slower; best 48.489 µs versus 44.519 µs for the selected chain. 3,564 exact comparisons, 264 graph checks; seven schedules pass 315 cases per sanitizer. [Recorded summary](results/cooperative_mlp_pass_summary_v1.json) |

Generated rejected quantization checkpoints and bytecode caches were removed during cleanup. Source, measured JSON, quality evidence, selected G32 checkpoints and required runtime bundles remain. See `results/cleanup_20260920.json` for exact removals and regeneration commands.
