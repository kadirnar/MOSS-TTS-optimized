# MOSS-TTS Optimized

Streaming optimization of **MOSS-TTS-v1.5 8B Delay** with voice cloning and **all 32 acoustic codebooks**, including the first playable audio chunk.

This is prepared as an independent repository for `kadirnar/moss-tts-optimized`, with a fresh Git history. It derives from OpenMOSS source; attribution and Apache-2.0 licenses are retained.

Repository documentation, comments, and interface messages use English. Multilingual synthesis examples, audio-aligned reference transcripts, normalization rules, and recorded benchmark data retain their original language so they remain valid and reproducible.

| Latest qualified measurement | Median |
|---|---:|
| Warm engine TTFA, registered cloned voice | **71.86 ms** |
| Loopback HTTP TTFA, registered cloned voice | **75.02 ms** |
| Fresh voice registration plus synthesis to first PCM | **110.67 ms** |

Measured on one H200 NVL, batch one. First PCM is a complete 80-ms chunk at 24 kHz. **The 50-ms target has not been reached.** The latest path uses calibrated INT4/G32 decode, BF16 prefill and an FP32 codec; it is not numerically equivalent to upstream BF16. Options default off, and published benchmark results do not imply that the original running services were upgraded.

- [Complete improvement table](optimization/IMPROVEMENTS.md)
- [Installation, benchmark commands and streaming API](optimization/README.md)
- [Latest selected optimization and qualification](optimization/REPORT_ATTENTION_HISTORY.md)
- [Latest rejected cooperative CUDA experiment](optimization/REPORT_COOPERATIVE_MLP.md)
- [Adapted upstream documentation](UPSTREAM_README.md)

## Implementation

The optimization package contains Triton/Gluon kernels, native CUDA/PTX implementations, calibrated DP4A projections, static KV caches, CUDA graph scheduling, prefill fusions, dependency overlap and streaming codec state. It includes the benchmarking and validation sources for SGLang, vLLM, TileLang, CuTe DSL and other investigated backends, with measured results and rejected trials distinguished from selected improvements.

The server provides voice registration and incremental PCM synthesis, input validation, single-request GPU admission, cancellation and recovery. See the optimization README for request formats and the exact optional presets.

## Runtime and generated artifacts

Use the environment specified by the individual report. The baseline BF16 requirements are in `optimization/requirements-runtime.txt`. The latest qualified preset uses Python 3.12, Torch 2.13.0+cu130, Triton 3.7.1, Transformers 5.14.1 and an H200/SM90 GPU. Isolated Triton 3.8 cubins are supplied for the selected QKV and gate/up kernels; this is not an instruction to upgrade the complete runtime to Triton 3.8.

Original model weights, calibration tensors, generated audio and GPU traces are external generated artifacts, excluded from Git. Download the fixed model revisions in `optimization/common.py`. For the calibrated preset, collect calibration inputs with `optimization.collect_calibration`, then export the selected weights with:

```bash
python -m optimization.calibrate_gptq --tag gptq_v1_g32_d10 --group 32 --damping 0.1
```

Read the report and calibration scripts before running GPU benchmarks; run GPU jobs sequentially. The raw JSON measurements and profile summaries included here document prior runs. Some evidence checkers require the original generated artifacts in addition to this source checkout. Rejected quantization checkpoint files were removed locally; their configurations, measurements and regeneration commands remain in `optimization/results/cleanup_20260920.json`.

## Provenance

- [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS), source commit `934d6826b084c46a0d033402174d5f8ac4ed2519`.
- [OpenMOSS/MOSS-Audio-Tokenizer](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer), source commit `56776e867cb38446fa4bc00d0aceccab5001b008`, vendored as ordinary source files under `moss_audio_tokenizer/`.
- TTS checkpoint revision: `cdd3b911b1585e3f2dbc7775ef10f9926f58850a`.
- Codec checkpoint revision: `3cd226ba2947efa357ef453bcad111b6eafba782`.

`SOURCE_PROVENANCE.json` records the original copied file hashes, packaging scope, and subsequent English-language edits and removals under `local_changes`. Upstream authorship and licenses remain applicable; see [LICENSE](LICENSE) and [tokenizer license](moss_audio_tokenizer/LICENSE).
