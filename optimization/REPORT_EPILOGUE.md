# Attention quantizer fusion and explicit layout experiments

The 50 ms target remains open. All work retains **32 codebooks**, voice cloning, streaming PCM, BF16 prefill and the FP32 codec. The existing supervisor services remain unchanged. This pass builds on the direct-load calibrated path in `REPORT_PACKED.md`.

## Scale-layout experiments

The direct-load kernel still converted integer group sums into a different register layout before scaling. Two approaches were measured: limit scale-load contiguity to 1/2/4, or load scales inside inline PTX in the integer sum's layout. The 192 row/warp/mode combinations passed tolerance checks against the preceding plan on actual projection shapes. Each candidate was additionally checked on three layers, four recorded activation rows, zero and spike inputs, using a private CUDA stream.

No candidate both passed all 18 exact comparisons for its projection and exceeded the predeclared 2% operator-gain threshold. The resulting plan therefore retains the previous operators. The fastest numerical variants saved only small amounts and changed some outputs; they were not integrated. `dp4a_scale_kernels.json` retains all timings and mismatch counts. An initial inline-assembly pointer/float broadcast compilation error was fixed by passing integer addresses; the failed log is retained separately.

## Explicit Gluon layouts

Following the [Gluon layout documentation](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html) and the March 2026 revision of the [Linear Layouts paper](https://arxiv.org/abs/2505.23819), `dp4a_gluon.py` explicitly distributes group sums and scales among registers, lanes and warps. It tests 144 configurations across the four real projection shapes. This uses Gluon in the installed Triton 3.7.1 environment; it is not a new serving engine.

| Projection | Previous operator | Fastest exact Gluon operator | Shared bytes |
|---|---:|---:|---:|
| QKV | 8.08 µs | 11.26 µs | 0 |
| Attention output | 7.01 µs | 8.93 µs | 0 |
| Gate/up | 19.48 µs | 31.60 µs | 0 |
| MLP down | 12.03 µs | 32.61 µs | 32 |

The chosen exact candidates eliminated most shared-memory communication but were slower. Their register counts were 68/66/96/180 per thread, without spills. The fastest nonmatching candidates were close to the existing operators and did not justify another numerical change. No Gluon operator was selected. This demonstrates why barrier count alone is insufficient for selecting a kernel; hardware-counter attribution remains unavailable on this instance.

Evidence: `dp4a_gluon_kernels.json`, `gluon_*.ptx`, `gluon_*.ttgir`, and the unchanged projection entries in `dp4a_gluon_exact_plan.json`. An initial sliced-layout construction error was corrected and its failed log retained.

## Attention output and activation quantization

The profile also contained one separate G32 activation-quantization launch after attention in every layer. `attention_quant.py` performs that operation in the split-attention reduction kernel, after the same BF16 output rounding. It preserves grouped maximum, rounded division, reciprocal and integer-rounding semantics. The following output projection consumes the quantized activation directly.

The operator sweep covers capacities 128/256/512/1024, six positions per capacity and four warp counts: 96 comparisons of BF16 output, INT8 codes and FP32 scales. Only **four warps** passed every exact check; it is the sole integrated configuration. Other warp counts can change floating reductions and remain experiments. At capacity 256, the cold-L2 graph timing fell from 7.79 to 6.27 µs for reduction plus quantization. These are isolated operator timings, not full-request savings.

Sixteen complete-backbone checks used separately captured baseline and fused graphs on a private stream. All text and audio logits matched exactly at early positions, capacity boundaries, position 1023 and returns to earlier positions, with a fully initialized 1023-token causal prefix. Evidence: `attention_quant_graph_validation.json`.

The complete warm, batch-one cloned-voice benchmark measured **92.35 ms median / 92.54 ms p95 TTFA**, five complete requests. Its 36-step teacher-forced diagnostics and saved WAV match the preceding direct-load path exactly. That path's preceding ten-request run measured 94.04 ms; these independent runs are retained separately and do not establish a latency guarantee. Processing, prefill, generation, FP32 codec and CPU PCM copy are included; HTTP, network transit and fresh reference encoding are excluded. The first playable chunk remains 80 ms of PCM decoded from all 32 codebooks.

All **16 bilingual generated WAVs and four reference WAVs** are byte-identical to the preceding selected path, with matching text, seed, voice, frame and termination metadata. The existing small-suite diagnostics therefore apply to the identical artifacts: Chinese CER 5.37%, English WER 0%, speaker cosine 0.9143. This is not a new ASR run or equality with the original BF16 model. Evidence: `quality_suite/attention_quant_audio_equivalence.json`.

The new profile reduces separate quantizer launches from 2,376 to 1,188 during the profiled request. The fused reduction consumes 2.335 ms total versus the previous reduction plus quantization's greater combined cost. Projections still consume 50.7% of GPU time, normalization/quantization 7.3%, and the main attention stage 7.0%.

## Streaming HTTP comparison

Fresh sequential temporary servers compared the attention-fused candidate with the preceding direct-load control on the same current source, runtime, registered voice, text and seeds. Each row contains 20 complete requests after a warmup, with all 32 codebooks:

| Configuration | HTTP median / p95 | Engine median | Initial decode step |
|---|---:|---:|---:|
| Direct-load control | 97.97 / 100.02 ms | 94.09 ms | 2.445 ms |
| Attention quantizer fused | 96.10 / 98.45 ms | 92.28 ms | 2.391 ms |

Preparation, prefill and first-codec-frame costs were similar, approximately 1.15, 9.21 and 4.77 ms. The paired HTTP median improvement is **1.86 ms**. All **20 complete PCM stream hashes match** between the configurations, and no request truncated. Invalid-input and cancellation/recovery checks passed; cancellation recovered through 429 then 200. The benchmark now records a SHA-256 hash after receiving each complete stream, in addition to timing and frame counts.

One contiguous fresh-reference registration-to-PCM observation took 136.74 ms fused and 135.18 ms control. These single observations do not demonstrate a fresh-reference improvement or define latency distributions. The earlier direct-load HTTP run was faster in different timing conditions; its historical 91.11 ms number is not used as the control for this comparison. Both temporary servers were stopped after testing. The original BF16 and FP8 supervisor endpoints remain unchanged.

Evidence: `http_attention_quant.json`, `http_attention_control.json`, `http_attention_comparison.json`, `http_attention_quant_health.json` and both JSONL stage records.

## Latest runtime trial

The current [Torch 2.14 release](https://pypi.org/project/torch/2.14.0/) requires [Triton 3.8](https://pypi.org/project/triton/3.8.0/). An isolated `/venv/moss-kernels-latest` environment was created with Torch 2.14.0+cu130, Triton 3.8.0, torchvision 0.29.0 and the resolved CUDA dependencies, including cuDNN 9.24. It imports the unchanged Transformers/audio support packages from the existing runtime through a `.pth` file. The original environments and services were not upgraded. This is a custom-kernel runtime test, not a vLLM engine upgrade.

With the same attention-fused configuration, five complete requests measured **89.71 ms median / 90.02 ms p95 TTFA**. The main attention kernel's profiled time fell from 6.325 to 4.050 ms across 1,188 calls, while the projection times remained similar. Teacher-forced agreement with the upstream model in that runtime changed: relative RMS 0.0121033, active top-1 0.679558 and KL 0.0281765. These are diagnostics against a newly executed reference, not equality with the prior runtime.

All 16 generated WAVs changed. A fresh matched CUDA-FP16 Whisper/WavLM evaluation produced normalized Chinese CER **8.51%**, English WER **0%**, speaker cosine **0.9153**, intended reference top-1 in 16/16 cases and no truncation. Chinese CER is higher than the preceding calibrated path's 5.37%, so this faster runtime is not selected as the quality-preserving option. The small diagnostic corpus remains insufficient for broad acceptance. An accidentally started CPU-INT8 ASR run was interrupted and retained as a partial artifact; only the completed CUDA-FP16 run supplies this comparison.

The fused attention operator was separately rechecked on the new compiler: four warps again passed every exact comparison across 96 configurations; its resource record was 36 registers/thread, no spills and 2048 shared bytes. The runtime, requirements, installation log, full-model profile and new quality scores are recorded in `latest_kernel_runtime.json`, `latest_kernel_runtime_requirements.json`, `latest_kernel_install.log`, `all32_gptq_dp4a_attention_quant_torch214*.json`, and `quality_suite/evaluation_attention_quant_torch214_normalized.json`.

A second isolated environment, `/venv/moss-triton38`, changes only Triton to 3.8 while retaining Torch 2.13 and cuDNN 9.20. It intentionally overrides Torch's declared Triton pin for this custom-kernel experiment; no vLLM or Inductor compatibility is claimed. Five requests measured **89.34 ms median / 89.51 ms p95**. Its recorded teacher-forced diagnostics and saved WAV exactly match the Torch 2.14 trial. This isolates a compiler-related numerical change on that fixture, rather than attributing it solely to the Torch/cuDNN upgrade. The compiler-only variant did not receive a separate 16-sample quality evaluation and is not selected for serving.

The default BF16 regression after the code changes completed five requests at **171.34 ms** and reproduced the preceding validation and saved WAV exactly. The next numerical investigation should isolate which Triton 3.8 operator changes the model outputs while retaining its attention speed improvement.

## Reproduction

Run GPU work sequentially. The measured configuration uses `/venv/moss-vllm` with Torch 2.13 and Triton 3.7.1:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_dp4a_scale
/venv/moss-vllm/bin/python -m optimization.benchmark_dp4a_gluon
/venv/moss-vllm/bin/python -m optimization.benchmark_attention_quant
/venv/moss-vllm/bin/python -m optimization.validate_attention_quant_graphs
/venv/moss-vllm/bin/python -m optimization.benchmark_quantization \
  --mode none --calibration optimization/results/gptq_v1_g32_d10 \
  --calibrated-backend dp4a --tag attention_quant_v1 \
  --attention-block 32 --fused-residual --dp4a-reciprocal \
  --grouped-activation --fused-dp4a --dp4a-gateup --norm-layout-limit 8 \
  --prefill-buckets 128 160 256 512 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant
```

`quality_generate` and `server` accept the same optional `--attention-quant` flag. It requires grouped G32 DP4A and the custom attention backend, and must be installed before graph capture. The default BF16 and FP8 configurations do not enable it.

The temporary serving test used:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant
```

Register a reference using `/v1/voices` and stream `/v1/audio/speech` as documented in the README. This foreground development invocation binds only to loopback; a persistent instance service should use supervisor. `epilogue_source_hashes.json` and `epilogue_sources.tar.gz` preserve the source snapshot for this pass.
