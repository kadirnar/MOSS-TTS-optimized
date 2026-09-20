# Current full-codebook optimization review

All results retain **32 codebooks**, including the first playable PCM chunk, and the original MOSS-TTS-v1.5 checkpoint. The **50 ms target is not met**. No claim of impossibility, fewer-codebook fallback, trained draft model or multi-GPU acceleration is made.

## Measurement boundary

One H200 NVL, batch one, warm models/kernels, complete text supplied at request start. First audio means a complete 1,920-sample, 80 ms chunk at 24 kHz, not a response header or an incomplete codebook frame. The in-process benchmark includes text processing, BF16 prefill, autoregressive delay steps, FP32 codec decoding and CPU float32 PCM transfer. Its voice reference is already encoded. HTTP measurements additionally include local request handling and s16le serialization. Fresh-reference measurements are reported separately. These boundaries are not interchangeable.

The supplied Chinese reference is 3.112 seconds. Every measured request in the primary model and HTTP comparisons finishes generation; the workload sweep explicitly stops after the first chunk. Initial model loading, compilation, graph capture and a warmup request are excluded. Raw samples, generated WAVs and profiles remain in `results/`.

## Current results

| Configuration | In-process median TTFA | p95 | Notes |
|---|---:|---:|---|
| Historical upstream BF16 | 848.52 ms | See original report | Original implementation |
| Custom BF16 before new fusions, cache regression check | 175.68 ms | See raw JSON | Original attention block 128 |
| BF16 with both exact component fusions | **170.96 ms** | 172.23 ms | Current default service configuration |
| Same BF16 fusion configuration, newer runtime | 166.35 ms | 166.80 ms | Not promoted; changed numerical/quality results |
| BF16, both fusions and attention block 32 | 165.87 ms | 166.26 ms | Optional; changed quality diagnostic |
| Current-runtime FP8 before these fusions | 131.34 ms | See raw JSON | BF16 activations/prefill, FP32 codec |
| FP8 + attention block 32 | 127.00 ms | See raw JSON | All 32 codebooks |
| FP8 + residual/RMSNorm fusion | 123.41 ms | 124.02 ms | All 32 codebooks |
| FP8 + both fusions | **118.14 ms** | 118.36 ms | Experimental; five complete measured requests |

The default BF16 service's separate 20-request HTTP measurement is **175.45 ms median / 178.23 ms p95**. Input validation, reference registration, cancellation and recovery passed. This HTTP result does not show a speedup over the historical 174.19 ms HTTP run; the in-process comparison supports the fusion improvement. Do not mix those measurements into a claimed HTTP speedup.

The experimental FP8 service measured **122.17 ms median / 125.27 ms p95** across 20 complete loopback HTTP requests. Voice registration, input validation and cancellation/recovery passed there too. Both services are running locally; neither is publicly exposed. The FP8 service remains explicitly separate from the default BF16 service.

The FP8 workload sweep measured 118.05–118.62 ms median across the four cached-reference Chinese/English prompt cases. Including fresh reference encoding, from a decoded CPU waveform to first PCM, measured **138.79 ms median / 139.23 ms p95** with the optimized FP32 encoder (20.52 ms median encoding), versus 168.65 ms with the original encoder. This excludes file parsing and network transfer. Evidence: `workloads_fp8_latest_fused.json`.

Core evidence: `all32_none_main_exact_fusion_fr_fg.json`, `all32_fp8_all_latest_gateup_b32w4_fr_fg.json`, `http_bf16_exact_fused.json`, `http_fp8_latest.json`. Full profiles use corresponding `_profile.txt` names. Historical results are preserved rather than overwritten.

## What changed

- Fused residual addition and RMSNorm while preserving BF16 rounding after residual addition and normalization. Exact component checks passed with graph replay and a non-default CUDA stream.
- Fused gate/up matrix-vector products with SiLU and multiplication, again preserving intermediate BF16 rounding. Four-warps-per-block kernels matched the previous components exactly in the recorded tests. FP8 isolated gate/up/SiLU improved from 32.09 to 28.06 microseconds with eight distinct matrices exceeding L2 cache.
- Tuned attention tiles across 40 configurations/contexts. Short-context block 32 improved latency but changed reduction order, so the default retains block 128 after quality evaluation.
- Fixed prefill KV updates to use explicit positions with newer Transformers StaticCache implementations. Original-runtime numerical regression metrics match the historical custom path.
- Added current-runtime and full serving-engine measurements rather than relying only on library kernel microbenchmarks.

## Voice cloning quality checks

Four supplied reference recordings, four prompts per voice: 16 utterances per configuration, Chinese and English. Every output was finite, nonempty and untruncated. ASR uses faster-whisper-small, CUDA FP16, beam five. Speaker vectors use pinned `microsoft/wavlm-base-plus-sv`, revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`. Chinese normalization converts traditional characters and numbers; English uses Whisper's English normalizer. Raw transcripts and raw metrics are retained.

| Configuration | Mean Chinese CER | Mean English WER | Mean speaker cosine |
|---|---:|---:|---:|
| Previous custom BF16 | 4.46% | 0.78% | 0.9104 |
| BF16, block 128, both fusions | Same saved audio | Same saved audio | Same saved audio |
| BF16, block 32, both fusions | 6.95% | 0% | 0.9163 |
| FP8, block 32, both fusions | 7.59% | 0% | 0.9090 |
| BF16, block 128, both fusions, newer runtime | 8.51% | 0% | 0.9072 |

All 16 WAV files from the selected BF16 fusion configuration are **byte-identical** to the previous custom BF16 suite. SHA-256 pairs and the complete comparison are in `quality_suite/exact_fusion_audio_equivalence.json`. This establishes equivalence on these samples, not to upstream generation or every possible request.

All configurations selected the intended voice among the four references under the speaker-vector metric. This small diagnostic suite and uncalibrated cosine measure do not establish corpus-level intelligibility or human-perceived voice quality. FP8 is kept experimental because its Chinese ASR error increased by 3.13 percentage points. No audio quality acceptance is inferred from preserving all codebooks alone.

The newer Torch/Triton runtime also changes BF16 numerical results: 166.35 ms TTFA, but 8.51% normalized Chinese CER on the same diagnostic suite. It is not promoted to the default service. Its separate `quality_suite/evaluation_bf16_latest_normalized.json` preserves the prior evaluations; this result also cautions against attributing all sampling differences to weight quantization alone.

## Serving-library and hardware experiments

| Trial | Result | Decision |
|---|---|---|
| SGLang-Omni 0.1.6 / SGLang 0.5.19 | Full streaming HTTP 423.39 ms median / 464.95 ms p95 | Slower for this batch-one workload |
| vLLM-Omni / vLLM 0.28 matched pair | Full HTTP 367.54 ms / 370.04 ms with custom v1 codec and incremental delay adapters | Hybrid benchmark, slower |
| Mirage persistent kernel, original BF16 backbone | 5.625 ms backbone only; logit RMS 0.0070 | Slower than custom full decode |
| Native Marlin W4/G128 and W4/G32 | 133.92 and 136.14 ms TTFA; low active-codebook agreement, one G128 truncation | Not selected |
| GemLite with correct FP32 accumulation | All 12 projection checks pass; slower kernels | Not selected |
| ExLlama native CUDA INT4 and PyTorch tinygemm | All 16 projection checks pass; no clear advantage over FP8 on the main shapes | Not promoted to model |
| Split-K Triton FP8, FP32 partial reduction | 162 shape/configuration checks pass; best down projection 19.45 us versus 17.20 us baseline | Not integrated |
| Fused Q/K norm, RoPE, cache update and attention | 18 output/KV checks pass; at position 144, block 32: 15.20 us versus 15.15 us baseline under cold-L2 timing | Not integrated |
| Codec TF32 policy | 4.899 → 4.548 ms, 60.39 dB SNR against IEEE FP32 | Precision tradeoff left disabled |

SGLang/vLLM full HTTP requests include uploading the reference data URI each time; the engines may internally cache reference tokens. Their numbers are not identical-boundary comparisons against the custom in-process cached-reference run. They are five warm complete requests after one warmup.

The upstream vLLM-Omni v1 codec could not initialize its streaming state pool. The explicitly gated adapter uses our FP32 codec. Its original delay adapter buffered the whole utterance; an incremental adapter now emits only completed 32-codebook frames and passed 18 frame-order/padding/drain cases. The prior buffered result (1,448 ms) is retained separately. An initial accidental SSE-as-PCM trial is marked `.invalid.*` and excluded.

Mirage launches private CUDA streams; validation calls its wait API before reading output. The first unjoined asynchronous trial is marked `.invalid_async.*` and excluded. Correct timing includes the wait. The source commit is `1f3338f9ae084559726fb1fc898ae077ff7d171a`.

Older CUDA/C, CuTe DSL, TileLang, native CUTLASS FP8, SGLang/vLLM operator, FlashInfer and Triton trials remain documented in the earlier reports. No library is claimed faster merely because it was installed.

## Remaining bottleneck

In the latest FP8 first-chunk profile, weight GEMV and fused gate/up/SiLU together consume **62.7% of GPU time**. Attention and its reduction consume about 7.7%; fused residual/RMSNorm about 4.7%. The delay pattern still requires successive backbone evaluations before the first complete 32-codebook frame. The unsuccessful split-K trial demonstrates that adding blocks and a reduction is not automatically beneficial.

Reaching 50 ms still requires a substantial additional improvement in serial backbone execution, without silently changing codebooks, streaming semantics or voice quality. Quantization, a trained speculative draft and multiple GPUs remain distinct possibilities to investigate; current measurements do not prove an absolute lower bound.

## Runtime and reproduction

- Default service: `/venv/main`, Torch 2.9.1+cu128, Triton 3.5.1, Transformers 5.0.0. `moss_tts_optimized`, local `127.0.0.1:18080`; BF16, attention block 128, both fusions, FP32 codec.
- Experimental FP8: `/venv/moss-vllm`, Torch 2.13.0+cu130, Triton 3.7.1; attention block 32, both fusions, FP32 codec. `moss_tts_fp8_experimental`, local port 18083, autostart disabled.
- Full SGLang and vLLM-Omni use their isolated environments. vLLM 0.29 was tested for native kernels but is CLI-incompatible with Omni 0.28; the full engine uses the matched 0.28 pair.
- Latest-package inventory, fixed HF model revisions, raw environment records, source hashes and exact reproduction commands are saved alongside the results and in `README.md`. Engine dependency pins take precedence over the newest standalone Torch version.

Run GPU benchmarks sequentially without concurrent inference. The two local custom services have independent process-local voice caches. Registered aliases must be recreated after restarting a service.
