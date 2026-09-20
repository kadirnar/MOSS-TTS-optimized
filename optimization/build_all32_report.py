"""Generate the follow-up report without replacing first-pass measurements."""
import hashlib
import json
from .common import ROOT, RESULTS


def main():
    cases={}
    labels={
        'none':'BF16 regression',
        'fp8_all':'FP8 all backbone projections',
        'int8_all':'INT8 all backbone projections',
        'fp8_all_tiled':'FP8, multiple output rows per CTA',
        'int8_all_tiled':'INT8, multiple output rows per CTA',
        'int4_all':'INT4 groups of 128, first unpacking kernel',
        'int4_all_tiled':'INT4 groups of 128, paired unpacking',
    }
    latency=[]
    quality=[]
    for name,label in labels.items():
        path=RESULTS/f'all32_{name}.json'
        if not path.exists():
            continue
        d=json.loads(path.read_text())
        assert d['codebooks']==32
        cases[name]=d
        v=d['validation']
        latency.append(f"| {label} | {d['ttfa']['median_ms']:.2f} | {d['ttfa']['p95_ms']:.2f} | {d['decode']['median_ms']:.3f} |")
        quality.append(f"| {label} | {100*v['max_relative_rms_logit_error']:.3f}% | {100*v['active_codebook_mean_top1_agreement']:.2f}% | {v['active_codebook_mean_kl_divergence']:.5f} |")
    flash=json.loads((RESULTS/'flashinfer.json').read_text())
    attention=[]
    for name,d in flash.items():
        if 'full_llm_decode' in d:
            attention.append(f"| {name} | {d['full_llm_decode']['median_ms']:.3f} | {d['max_logit_difference_from_triton']:.4f} |")
        else:
            attention.append(f"| {name} | failed | {d.get('error','Unknown')} |")
    asr=json.loads((RESULTS/'all32_asr_smoke.json').read_text())
    transcripts=[]
    for name,d in asr['samples'].items():
        transcripts.append(f"| {name} | {100*d['normalized_character_error_rate']:.2f}% | {d['transcript']} |")
    kernel=json.loads((RESULTS/'weight_kernel_validation.json').read_text())
    best=min(cases,key=lambda name:cases[name]['ttfa']['median_ms'])
    report=f'''# All-32-codebook follow-up — 2026-09-19

**All 32 codebooks are mandatory, including the first PCM chunk. The 50 ms target remains unmet.** Streaming voice cloning is retained. No reduced-codebook option is implemented or planned under this requirement.

The fastest result in this pass is **{cases[best]['ttfa']['median_ms']:.2f} ms median TTFA** ({labels[best]}), and is **experimental, not deployed**. The running service continues to use BF16 LLM weights and the FP32 codec; its earlier measured HTTP median was 174.19 ms. See [the first-pass report](REPORT.md) for the original baseline, codec work, voice registration, and service tests.

## Timing

Warm batch one on the same H200 NVL, complete Chinese text and a cached 3.112-second voice reference. The timer includes text processing, BF16 prefill, all 32-codebook generation steps, sampling, FP32 codec decoding and first 1,920-sample PCM transfer to CPU. Network and reference encoding are excluded from these cached-reference measurements. Each TTFA row contains five measured requests after one discarded warmup; complete utterances were generated with a 400-step limit. Check each raw run's `truncated` flag. LLM component timing has 20 samples. GPU experiments ran sequentially with the deployed service idle.

| Variant | Median TTFA ms | p95 ms | Full LLM step ms |
| --- | ---: | ---: | ---: |
{chr(10).join(latency)}

FP8/INT8 use symmetric per-output-row scaling. FP8 is E4M3FN; INT8 rounds to nearest and clamps to ±127. INT4 uses groups of 128 input weights and symmetric ±7 levels, without AWQ/GPTQ calibration. Activations, prefill, embeddings, and output heads remain BF16; codec and reference encoder remain FP32. Original BF16 weights are retained for prefill, so these experiments reduce decode weight traffic, not total resident model memory.

## Iteration decisions

1. The first-pass profile attributed about 68% of GPU time to backbone GEMV. Extending 8-bit weights from only the MLP to attention QKV and output projections reduced measured TTFA further. The FP8 profile still attributes about 59% of GPU time to GEMV and 10% to decode attention.
2. A 108-case Triton sweep varied rows per CTA and warps at the four real backbone projection shapes and three weight dtypes. Each graph rotated eight distinct matrices to exceed L2 capacity. Multiple output rows improved isolated 8-bit kernel measurements, but **regressed end-to-end TTFA**. The original single-row kernel remains selected for the 8-bit experiments.
3. FlashInfer SIMT and tensor-core attention both passed dynamic cache-length and non-default CUDA-stream checks. Complete-LLM measurements below did not establish an improvement over the current Triton path. Neither is selected for deployment. These are attention adapters inside this custom LLM, not full serving-engine comparisons. All three attention rows use greedy decoding; the TTFA and quantization component runs use the default stochastic sampler.
4. The initial INT4 implementation spent about 72% of GPU time in its unpack-and-GEMV kernel and also increased logit error substantially. A second kernel loads each packed byte once and processes both nibbles together, but was slower still: about 91% of its GPU time went to this kernel. The 12,288-input variant used 62 registers versus 32 for the first implementation; neither reported register spills. Both timings and error measurements are retained above. Both INT4 implementations are rejected for deployment.

| Attention backend | Full BF16 LLM step ms | Max absolute logit difference from Triton |
| --- | ---: | ---: |
{chr(10).join(attention)}

## Numerical and audio checks

The same 36 upstream teacher-forced decode steps were used for each row. These are diagnostic comparisons on one cloned-voice prompt, not a quality acceptance test.

| Variant | Maximum relative RMS logit error | Mean active-codebook top-1 agreement | Mean active-codebook KL |
| --- | ---: | ---: | ---: |
{chr(10).join(quality)}

All {len(kernel)} quantized-kernel cases passed comparison against independently materialized FP32 weights at the four backbone shapes, CUDA graph replay on a non-default stream, zero-input replay, and exact BF16-prefill fallback. Maximum output RMS error was {100*max(d['relative_rms_vs_fp32'] for d in kernel):.3f}%, including BF16 output rounding. This validates kernel arithmetic, not quantization quality.

FlashInfer was checked at positions 0, 63, 127, 128, 255, 511, 767, 1023, and then 17 using the same captured graph. The BF16 regression reproduced the first-pass 0.772% maximum relative RMS logit error and 92.18% active-codebook top-1 agreement.

The following ASR smoke test uses faster-whisper-small on CPU INT8. The reference sentence is “{asr['reference']}”. Character error ignores punctuation and whitespace. Several outputs have the same homophonic “流逝”/“流式” substitution as the BF16 baseline. One sentence, one reference voice, and ASR alone cannot establish speaker identity, prosody, or broad language quality.

| Audio sample | Normalized character error | ASR transcript |
| --- | ---: | --- |
{chr(10).join(transcripts)}

No quantized path is enabled in the service. Broader speaker-similarity, intelligibility, and listening evaluation is still required. Full SGLang-Omni/vLLM-Omni serving comparisons and 50 ms TTFA remain outstanding.

## Reproduction and artifacts

See [README.md](README.md) for commands. Raw per-request timings, individual decode samples, error metrics and profiles are in `results/all32_*.json` and `results/all32_*_profile.txt`; generated PCM audio is saved as matching WAV files on this instance.

- [FP8 all-projection run](results/all32_fp8_all.json), [INT8 run](results/all32_int8_all.json), [BF16 regression](results/all32_none.json)
- [INT4 first kernel](results/all32_int4_all.json), [INT4 paired unpacking](results/all32_int4_all_tiled.json)
- [Triton tile sweep](results/weight_read_tiles.json), [kernel validation](results/weight_kernel_validation.json)
- [FlashInfer results](results/flashinfer.json), [ASR smoke](results/all32_asr_smoke.json)
- [Current source hashes](results/all32_source_hashes.json); first-pass hashes remain in `results/source_hashes.json`
'''
    (ROOT/'optimization/REPORT_ALL32.md').write_text(report)
    summary={'target_ms':50,'target_achieved':False,'required_codebooks':32,'voice_cloning':True,
        'fastest_experiment':best,'fastest_experimental_ttfa':cases[best]['ttfa'],
        'deployed_precision':'BF16 LLM / FP32 codec','deployed_changed':False,
        'variants':{name:{'ttfa':d['ttfa'],'decode':d['decode']} for name,d in cases.items()}}
    (RESULTS/'all32_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'optimization').rglob('*'))
        if p.is_file() and p.suffix in ('.py','.cu')}
    (RESULTS/'all32_source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='variants'},indent=2))


if __name__=='__main__':main()
