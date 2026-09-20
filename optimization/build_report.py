"""Generate the report directly from measured JSON artifacts."""
import json
import statistics
from .common import ROOT,RESULTS


def main():
    def read(name):return json.loads((RESULTS/(name+'.json')).read_text())
    upstream=read('upstream_ttfa');native=read('streaming_optimized');http=read('http')
    fp8=read('streaming_fp8_experimental');work=read('workloads');validation=read('validation')
    encoder=read('encoder');backends=read('attention_backends');libraries=read('library_trials')
    env=read('environment')
    selected=native['ttfa']['median_ms']
    speedup=upstream['ttfa']['median_ms']/selected
    rtf=statistics.median(r['total_ms']/(r['frames']*80) for r in http['runs'])
    rows=[]
    for name,metric in [('Upstream loop + earliest-frame emission',upstream['ttfa']),
        ('Optimized BF16 LLM / FP32 codec',native['ttfa']),('Optimized loopback HTTP',http['ttfa']),
        ('Uncached reference, optimized encoder',work['uncached_reference_optimized_encoder']['ttfa']),
        ('Experimental FP8 MLP, all 32 codebooks',fp8['ttfa'])]:
        rows.append(f"| {name} | {metric['median_ms']:.2f} | {metric['p95_ms']:.2f} | {len(metric['samples_ms'])} |")
    table='\n'.join(rows)
    backend_table='\n'.join(f"| {name} | {v['full_llm_decode']['median_ms']:.3f} | {v['logits_max_abs_difference_from_triton']:.3f} |" for name,v in backends.items())
    library_table='\n'.join(f"| {name} | {libraries[name]['us']:.3f} | {libraries[name]['max_abs_error']:.5f} |" for name in ['torch_silu','triton_silu','sglang_silu','vllm_silu','cuda_c_silu','tilelang_silu','cute_dsl_silu'])
    codec_ms=statistics.median(sum([r['codec_ms'] for r in native['runs']],[]))
    report=f'''# Measured optimization report — 2026-09-19

**50 ms TTFA was not achieved.** The requested checkpoint now streams with voice cloning on one H200 NVL. The all-32-codebook BF16/FP32 path measures **{selected:.1f} ms median TTFA**, a **{speedup:.2f}× improvement** over the measured upstream loop. The running local service measures **{http['ttfa']['median_ms']:.1f} ms median / {http['ttfa']['p95_ms']:.1f} ms p95** over 20 complete HTTP requests.

The subsequent all-32-codebook quantization, Triton tile, and FlashInfer experiments are documented separately in [REPORT_ALL32.md](REPORT_ALL32.md). The service still uses this BF16/FP32 path.

## Measurement contract

Batch one; warm weights, kernels and graphs; complete text input; first actual 1,920-sample (80 ms) mono PCM chunk available on CPU. Includes text processing, LLM prefill, all sequential generation steps, sampling, codec decoding and transfer. HTTP results also include loopback transport. Cached-reference numbers exclude reference encoding, which is reported separately. No generated audio is cached. Model loading, compilation/capture, WAN delay and perceptual speech onset are not included.

The main workload is the repository's 3.112-second, 48 kHz mono Chinese reference, resampled to 24 kHz, and the Chinese sentence in `results/fixture.pt`. Four Chinese/English prompt cases were also timed. Repeated prompt timing does not bypass prefill or reuse generated speech.

| Path | Median TTFA ms | p95 ms | Measured requests |
| --- | ---: | ---: | ---: |
{table}

The upstream implementation has no incremental output API. Its baseline here uses an observation hook around its **unmodified** generation loop to decode at the earliest moment all 32 codebooks exist, then stops before the next unnecessary LLM forward. This is a streaming baseline, not time to the entire utterance.

Service median real-time factor is **{rtf:.3f}**, approximately **{1/rtf:.2f}× real time** including startup delay for each utterance. Reference registration in the final HTTP test took {http['voice_registration']['reference_encode_ms']:.2f} ms for encoding after WAV parsing. The uncached-reference test above starts with a decoded CPU waveform and includes resampling, normalization, encoding, and synthesis to first PCM.

## Improvements and re-profiling

1. The baseline measured about 21.9 ms per LLM forward and 46 ms per streaming codec chunk. The first profile showed extensive launch overhead, small elementwise operations, matrix reads and repeated codec projections.
2. Static KV storage and CUDA graphs reduced LLM decode to about 11.7 ms. Fused QKV/gate-up projections and Triton RMSNorm reduced it further to about 10 ms. Fusing only heads with eager static attention regressed performance and was not selected.
3. Triton Q/K normalization, RoPE/cache writes and split decode attention reduced the LLM step to about 5.6 ms. Subsequent Triton GEMV kernels for MLP, QKV and output projections brought the final full LLM decode measurement to **{backends['triton']['full_llm_decode']['median_ms']:.3f} ms**. The text head computes only the two permitted tokens while inside audio generation; full-vocabulary projection remains for the prefix.
4. Codec graph replay reduced about 46 ms to 15.5 ms, bitwise equal in the initial graph-only comparison. Cached projected LFQ tables eliminated repeated codebook projections. Fused interleaved RoPE, ring-cache writes and causal attention reduced final per-chunk decoding plus CPU transfer to approximately **{codec_ms:.2f} ms**.
5. Multi-tensor reset eliminated hundreds of small counter-reset operations. Prefill graphs and trimming unused KV capacity enabled causal FlashAttention for prefill. The service retains complete state across emitted chunks and resets it between requests.
6. Uncached reference encoding now uses FP32 graph buckets. In its paired benchmark the encoder fell from **{encoder['upstream']['median_ms']:.2f} to {encoder['cuda_graph']['median_ms']:.2f} ms**. It produced identical reference tokens for eight tested durations from 0.24 to 14.5 seconds.
7. The final profiler attributes roughly **68% of GPU time to GEMV**, followed by decode attention at roughly **8%**. Weight movement and the 32 sequential LLM steps are the remaining primary latency costs. Profiles are in `results/final_profile.txt` and `results/final_trace.json`.

## Library and language trials

The following are **actual complete-LLM decode measurements with different attention kernels**. The scheduler, sampler, embeddings and codec remain this implementation. They are not full SGLang-Omni/vLLM-Omni server benchmarks. vLLM timing includes conversion into its paged-cache layout; this is an adapter comparison, not a claim that its native engine is slower.

| Attention implementation | Full LLM step ms | Max logit difference from Triton |
| --- | ---: | ---: |
{backend_table}

The SGLang and custom Triton results are close; the deployment keeps the custom kernel to avoid serving-stack dependencies. The SGLang code is from 0.5.7, with only platform detection isolated; vLLM 0.12.0 CUDA operators loaded and executed on the installed Torch build. Their complete dependency requirements differ from this runtime, so full serving-engine comparisons remain unfinished. Current Omni repositories were inspected, and the SGLang-Omni dependency resolver was exercised, but no full-engine TTFA is claimed.

For the actual MLP SiLU×up shape (12,288 BF16 values), graph-amortized GPU microbenchmarks produced:

| Implementation | Microseconds | Max absolute error from eager Torch |
| --- | ---: | ---: |
{library_table}

Native CUDA/C and multiple Triton GEMV layouts were also measured. For the 24,576×4,096 projection, the selected Triton row kernel measured {libraries['triton_gemv_r1_24576x4096']['us']:.2f} µs versus cuBLAS {libraries['cublas_gemv_24576x4096']['us']:.2f} µs and native CUDA/C {libraries['cuda_c_gemv_24576x4096']['us']:.2f} µs. These are shape-specific microbenchmarks, not end-to-end TTFA. TileLang and CuTe DSL kernels compiled, ran, and passed the tested elementwise comparison. The integrated path uses Triton and cuBLAS; native CUDA/C, TileLang, and CuTe trials are retained for reproducibility.

## Correctness and quality evidence

- The 140-frame decoder comparison crosses the ten-second ring boundary and exercises a non-default CUDA stream. SNR against the original decoder was **{validation['codec']['snr_db']:.2f} dB**, max absolute waveform error **{validation['codec']['max_abs_error']:.3g}**, and the first-frame error after reset was zero.
- Reference encoding produced exactly the same tokens in all eight duration cases. These are slices/repetitions of one supplied reference, not eight independent voices.
- Over 36 teacher-forced LLM steps, maximum relative RMS logit error was **{100*validation['llm_teacher_forced']['max_relative_rms_error']:.3f}%**; active-codebook top-1 agreement averaged **{100*validation['llm_teacher_forced']['active_codebook_mean_top1_agreement']:.2f}%** and mean active-codebook KL divergence was **{validation['llm_teacher_forced']['active_codebook_mean_kl_divergence']:.5f}**. Floating-point operation ordering and sampling precision differ. Generation is **not bitwise identical** to the upstream implementation.
- One Chinese utterance from each of BF16 and experimental FP8 was transcribed with Whisper-small. Both retained the sentence in this smoke check, with a homophonic transcription substitution. This is not corpus-level WER, multilingual quality certification, or a speaker-similarity evaluation.
- HTTP tests cover voice registration, invalid voice/length/context requests, full streaming responses, cancellation, busy admission and subsequent recovery. The service serializes GPU work and copies PCM before the next graph replay.

Mixed-precision codec decoding was tried, but the small additional speed benefit and approximately 34 dB waveform SNR did not justify selecting it. The deployed codec stays FP32. FP8 MLP weights are benchmark-only and require broader quality validation before deployment.

## Why the 50 ms target remains unresolved

This checkpoint's delay pattern needs 32 sequential LLM decode steps after the initial audio-start prediction to assemble a complete first frame. The user explicitly requires retaining all 32 codebooks, including the first PCM chunk.

A bandwidth estimate explains the remaining gap for conventional BF16 autoregression: backbone linear weights contain about 6.95 billion parameters, or 13.9 GB in BF16. Reading them for 32 sequential steps moves about 445 GB. At the H200's advertised [4.8 TB/s peak bandwidth](https://www.nvidia.com/en-us/data-center/h200/), that alone is roughly **93 ms**, before prefill, attention, sampling, codec and transfers. This is an idealized estimate for this execution strategy, not a proof against speculative decoding, different precision, multi-GPU execution, or architectural changes.

The original objective is therefore **partially achieved**: substantial measured acceleration, streaming and voice cloning are implemented; **50 ms TTFA, full serving-engine comparisons, and broad quality evaluation remain outstanding**. A speculative/draft model, more aggressive validated quantization, or more hardware would be separate next experiments. Every experiment must retain all 32 codebooks.

## Artifacts and provenance

- [Run instructions and API](README.md)
- [Raw HTTP measurements](results/http.json), [upstream TTFA](results/upstream_ttfa.json), [workloads](results/workloads.json), [experimental FP8](results/streaming_fp8_experimental.json)
- [Numerical validation](results/validation.json), [reference encoding validation](results/encoder_validation.json), [ASR smoke](results/asr_smoke.json)
- [Attention backend comparison](results/attention_backends.json), [library microbenchmarks](results/library_trials.json)
- [BF16 audio sample](results/streaming_optimized.wav), [FP8 sample](results/streaming_fp8_experimental.wav)
- [Environment and revisions](results/environment.json), [runtime requirements](requirements-runtime.txt)

Repository commit: `{env['source_commit']}`. TTS revision: `{env['tts_revision']}`. Codec revision: `{env['codec_revision']}`. Hardware: one NVIDIA H200 NVL, CUDA 12.8 toolkit, Torch {env['versions']['torch']}, Triton {env['versions']['triton']}. Upstream [model](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) and [source](https://github.com/OpenMOSS/MOSS-TTS) are unchanged; changes are isolated in `optimization/` plus the supervisor service files.
'''
    (ROOT/'optimization/REPORT.md').write_text(report)
    summary={'target_ms':50,'target_achieved':False,'all_codebooks':32,
        'upstream_ttfa_median_ms':upstream['ttfa']['median_ms'],'optimized_ttfa_median_ms':selected,
        'speedup':speedup,'http':http['ttfa'],'experimental_fp8':fp8['ttfa'],
        'uncached_reference':work['uncached_reference_optimized_encoder']['ttfa'],
        'realtime_factor':rtf,'service':'http://127.0.0.1:18080'}
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
