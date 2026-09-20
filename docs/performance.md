# Performance and improvements

All measurements use MOSS-TTS v1.5 **8B**, batch one, streaming voice cloning,
**32 acoustic codebooks**, and 24 kHz mono output. The first playable chunk
contains 1,920 samples (80 ms).

| Historical qualified workload on H200 NVL | Median TTFA |
|---|---:|
| Warm engine, previously encoded voice | 71.86 ms |
| Warm loopback HTTP, previously encoded voice | 75.02 ms |
| Fresh voice registration and HTTP synthesis | 110.67 ms |

**The 50 ms target has not been reached.** These are historical measurements
of the selected calibrated G32 path, not new benchmarks of the library wrapper.
[Retained timing samples](benchmark.json) preserve their measurement boundaries.
The G32 path changes weight precision; equivalence checks are against its
preceding calibrated implementation, not upstream BF16.

The library migration was checked on GPU: BF16 and G32 both synthesize complete
streams; G32 reproduces the previous 76-frame PCM hash exactly. Cancellation,
busy rejection, non-default CUDA streams and independent chunk storage also
pass. See [verification](verification.json) for the recorded migration checks.

## Improvement history

Historical milestones use different configurations and are not a single paired
comparison. Individual gains cannot be added. The library retains the BF16 and
selected G32 inference paths; rejected backends and benchmark scripts were
removed from the working tree. Earlier research records remain in Git history
at commit `34878cf`.

| Improvement | Measured result | Evidence / qualification |
|---|---|---|
| Static KV caches, decode CUDA graphs, fused QKV and custom Triton GEMV/attention | LLM step approximately 21.9 → 4.784 ms; initial complete pipeline 848.52 → 175.22 ms TTFA | Initial report; BF16 model differs numerically from upstream |
| FP32 streaming codec graphs, projected LFQ tables, fused RoPE/cache/attention, multi-tensor reset | Approximately 46 → 4.83 ms per PCM chunk | Initial report; 140-frame comparison, 115.95 dB SNR |
| CUDA graph buckets for cloned-voice reference encoding | 69.07 → 20.89 ms encoder latency | Initial report; identical tokens for eight tested durations |
| Residual/RMSNorm and gate/up/SiLU fusion | BF16 TTFA 175.68 → 170.96 ms | Review; 16 diagnostic WAVs byte-identical to preceding custom BF16 |
| FP8 backbone plus fused decode kernels | 118.14 ms engine / 122.17 ms HTTP TTFA | Review; experimental, Chinese quality diagnostic regresses |
| GPTQ INT4/G32 calibration, grouped activation DP4A, normalization fusion and prefill buckets | 106.43 ms engine TTFA milestone | Calibration; quantized path, not upstream BF16 equivalence |
| Interleaved INT4 packing and bounded 128/256/512-token decode graphs | 106.43 → 98.69 ms historical engine medians | Packing; matching-layout path preserves diagnostic WAVs |
| Direct packed activation loads in projection kernels | 94.04 ms engine TTFA milestone | Epilogue; matching arithmetic retained |
| Attention reduction fused with G32 activation quantization | 92.35 ms engine TTFA milestone | Epilogue; exact recorded outputs |
| Fixed-order vectorized native CUDA attention | **1.71 ms paired TTFA gain** | Native attention |
| Gate/up projection, SiLU and output quantizer fusion | **1.51 ms paired gain** | Gate/up quantizer |
| Scaled signed-integer DP4A unpacking | **1.23 ms paired gain** | Scaled DP4A; weight precision unchanged |
| Fuse normalization/activation quantization into consuming projections | **1.77 ms paired gain** | Normalization fusion |
| Skip output heads masked by the existing delay schedule | **0.509 ms paired gain** | Audio heads; every emitted frame still has 32 codebooks |
| Store exactly representable projection scales in BF16 | **0.44–0.52 ms paired gains** | Short scales; not additional scale quantization |
| CUDA programmatic dependent launch for projections | **0.64–0.76 ms paired gains** | Projection overlap; producer waits retained |
| Extend dependency overlap through attention | **1.86–4.00 ms paired gains** across runtime phases | Attention overlap; phase-dependent, not additive |
| Shared codec clocks and first-frame attention specialization | **0.301 ms paired TTFA gain**; first codec frame 4.769 → 4.517 ms | Codec clocks; exact waveform/cache checks |
| Capture all first 32 audio steps in one CUDA graph | **1.616 ms paired gain** | Initial audio graph; full sampling/RNG preserved |
| Hopper asynchronous bulk L2 weight hints | **0.446–0.483 ms paired gains** | Bulk prefetch |
| BF16 prefill QKV fusion and prefetch address specialization | **1.787 ms** fusion and **0.188 ms** address paired gains | Prefill QKV; separate comparisons |
| BF16 prefill SiLU/product and residual/normalization fusion | **0.644 ms paired gain** | Prefill pointwise |
| Preload attention-output weights into registers | **0.605 ms paired gain** | Weight staging |
| Clustered QKV projection/head normalization/RoPE/cache writes | Approximately **0.11 ms paired gain** | QKV clusters; eight-CTA Hopper kernel |
| Eight-row, four-warp down-projection tile | **0.447 ms paired gain** | Down tile; p95 slightly regresses |
| Isolated Triton 3.8 exact gate/up compilation | **0.238 ms final paired gain** | Gate/up compiler; selected host stays on Triton 3.7.1 |
| Load historical K/V before the retained attention dependency wait | **1.685 ms paired engine gain**, reaching **71.856 ms**; **1.919 ms paired HTTP gain**, reaching **75.022 ms** | Historical KV; 48 cloning WAVs unchanged |
| Streaming API and voice lifecycle | Incremental 24-kHz mono PCM, registration, validation, busy admission and cancellation recovery | API and commands; batch-one ownership, no generated-audio cache |

## Other evaluated backends

SGLang-Omni measured 423.39 ms full streaming HTTP median; the adapted
vLLM-Omni path measured 367.54 ms. Their workload boundaries differ from the
71.86 ms engine result. FlashInfer, Marlin, alternative quantization groups,
TileLang, CuTe DSL and additional native CUDA schedules were investigated;
none replaces the selected runtime here. These research dependencies are not
required to install the library.

## Practical limits

- Full text input with streaming audio output. Incremental text input is not implemented.
- One active request per model. Close an abandoned generator before reusing it.
- Default capacity: 1,024 prompt plus generated positions. G32 bundles require this capacity.
- Reference encoding is included only in fresh-voice measurements.
- G32 requires Hopper SM90, Triton 3.7.1 and a compatible CUDA toolkit with nvcc.
- BF16 weights remain resident for prefill even with G32 decode. This is not a 4-bit-only memory footprint.
- Small bilingual diagnostics do not establish broad perceptual quality or speaker similarity.
- GPU clocks, compiler versions, text length, reference length and load change latency.
