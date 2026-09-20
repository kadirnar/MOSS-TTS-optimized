# Skip masked future heads, retain every codec codebook

After normalization/consumer fusion, an optional output-head graph change measures **85.27 → 84.84 ms median in-process TTFA**, with **0.509 ms median paired gain** across ten complete streams. Eight pairs improve and two regress; p95 changes **85.77 → 85.89 ms**. All PCM is identical. This is a modest, noisier gain than the preceding normalization fusion, and the **50 ms target remains unmet**.

Every playable frame still contains **all 32 codebooks**. The original delay schedule initially masks future heads: only one is active at the first audio step, then two, and so on. This experiment computes a prefix of 8, 16, 24 or 32 heads that includes every active channel. It skips only matrix rows whose outputs the existing schedule would discard. It keeps the full 32×1,024 sampling tensor and random-draw shape, filling omitted logits with zeros before the existing mask replaces their sampled IDs with padding. Full-head computation resumes when all channels are active. The codec, delay dependencies and required serial steps are unchanged.

## Arithmetic audit and implementation

`benchmark_audio_head_prefix.py` loads the original BF16 head weights directly and tests all 32 prefix sizes on three captured real final hidden states, eight random inputs, zero and spike inputs: **416 comparisons**. Changing cuBLAS matrix dimensions changes arithmetic for prefixes **1, 2, 3, 4, 7 and 9**: 456 differing active logits and two differing sampled tokens in aggregate. Those sizes are rejected. The selected **8, 16, 24, 32** prefixes match every active logit and sampled token in the audit.

Private-stream graphs rotate across 16 copies of the head weights so even one-head prefixes exceed H200 L2 capacity. Nine rounds reverse timing order. Full projection takes **69.67 µs**; selected prefixes 8/16/24 take **23.51 / 39.64 / 60.62 µs**, including zero padding/copy. These are isolated operator timings, not TTFA.

`DecodeAudioHeadBuckets` captures the Cartesian product of head prefixes and attention capacities 128/256/512/1,024. It reuses the four existing full-head graphs and adds twelve prefix graphs. CPU scheduling chooses a capacity strictly greater than the actual position and a prefix covering the current active-channel count. `FastLLM._decode` uses the prefix only while those graphs are captured; default execution still computes all heads. Diagnostic logits for skipped inactive heads are zero, while active logits and generated IDs are preserved on the tested cases.

## Full-model and streaming checks

- **1,120 private-stream full-model graph cases** cross 16 positions, 14 audio-length boundaries and five delay states. All text logits, active audio logits and complete sampled IDs match the full-head graphs. Cases include the 1,024 fallback and returns to earlier positions.
- Ten alternating paired requests use identical references/text/seeds and reverse order every pair. All complete float32 PCM streams match; every control hash also matches the preceding norm-fusion benchmark. Paired gains are 0.616, 0.840, 1.013, 0.522, 0.484, 0.496, −1.261, 0.338, 0.533 and −0.026 ms. The larger regression remains unexplained.
- All **16 bilingual generated WAVs and four reference WAVs** match the previous calibrated path byte for byte, including completion/voice/text/seed metadata. Existing small-suite CER 5.37%, English WER 0% and speaker cosine 0.9143 apply to the identical files. This does not establish original BF16 equality or broad quality acceptance.
- Twenty complete HTTP PCM hashes/frame counts match. All requests finish without truncation; invalid-input, cancellation, HTTP 429 and recovery checks pass. Both temporary servers are stopped and the original services retain their PIDs/caches.

Evidence: `audio_head_prefix_v1.json`, `audio_head_graph_validation_v1.json`, `audio_head_paired_v1.json`, `audio_head_prior_control_equivalence.json`, `quality_suite/audio_head_audio_equivalence.json`, and `http_audio_head_comparison_v1.json`.

## HTTP variability

Sequential temporary servers measure **91.01 → 88.72 ms HTTP median**, p95 **93.85 → 93.33 ms**. One fresh-reference registration-plus-synthesis observation measures **131.64 / 127.32 ms**. Registration alone measures 36.30 / 34.77 ms. The first complete PCM chunk is 3,840 bytes of 24 kHz mono signed 16-bit audio.

The candidate includes a **139.64 ms** HTTP outlier. Late in that same run, unchanged prefill falls from about 9.21 to 8.83 ms and codec decode from about 4.77 to 4.39 ms. The cause is unknown; the final 83.97 ms observation is not a selected median or evidence of a larger kernel gain. Median engine TTFA is 87.05 → 84.87 ms, while preparation is 1.548 → 1.136 ms and seeding 0.327 → 0.235 ms. The entire 2.29 ms HTTP difference cannot be attributed to the head optimization. The alternating same-process **0.509 ms** paired gain is the narrower estimate.

Stage evidence and all individual samples remain in `http_audio_head_stages_comparison_v1.json`. Timing and correctness checks run sequentially without concurrent test inference. These are warm, batch-one, complete-text, cached-voice measurements; fresh reference costs are separate and initial model/graph startup is excluded.

## Reproduction

A separate profile and memory pass preserves PCM in its two diagnostic pairs. The extra twelve graphs require **384.8 MiB additional live tensor allocation / 408 MiB reserved**, and **0.58 seconds** of additional warmup/capture after the base engine is warm. These measurements exclude model loading and earlier compilation; they are not total cold startup cost.

The updated profile still places fused normalization/projections and the remaining projections at **50.170 ms / 59.49%** of GPU time. Full-width cuBLAS calls in the principal audio-head kernel fall from 35 to 11 as 24 decode calls use prefix shapes. Total profiled GPU time changes 84.886 → 84.333 ms in these separate profiles; profiling is excluded from reported TTFA. The main remaining bottleneck is still backbone projection arithmetic/weight traffic. Evidence: `audio_head_paired_profile_v1.json` and `audio_head_profile_profile_v1.txt`.

Use the selected Torch 2.13 / Triton 3.7.1 environment, with fresh tags and one GPU job at a time:

```bash
/venv/moss-vllm/bin/python -m optimization.benchmark_audio_head_prefix --tag reproduce
/venv/moss-vllm/bin/python -m optimization.validate_audio_head_graphs --tag reproduce
/venv/moss-vllm/bin/python -m optimization.benchmark_audio_head_paired --tag reproduce --pairs 10
/venv/moss-vllm/bin/python -m optimization.benchmark_audio_head_http --tag reproduce
```

Add `--audio-head-buckets` to the preceding norm-projection `quality_generate` command. An optional development server is:

```bash
/venv/moss-vllm/bin/python -m optimization.server --port 18084 \
  --calibration optimization/results/gptq_v1_g32_d10 \
  --packing-plan optimization/results/dp4a_direct_exact_plan.json \
  --decode-buckets --attention-quant --native-attention --gateup-quant \
  --scaled-dp4a --norm-projection --audio-head-buckets
```

This is a foreground loopback development endpoint; use supervisor for a persistent service. The new flag is opt-in, requires norm-projection plus context buckets, and adds graph startup/memory costs. It does not change voice registration or streaming API usage. Omit it to retain the preceding fully qualified norm-fusion path. Exactness evidence is scoped to this runtime, GPU, model and test cases.

The code/docs are included in `norm_projection_sources.tar.gz` and its SHA-256 manifest together with the preceding fusion. Model weights and measured artifacts remain separate.
