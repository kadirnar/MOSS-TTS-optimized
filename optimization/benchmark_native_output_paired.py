"""Alternate Triton/native output-projection graphs in one model/process.

Both managers share weights/KV storage; every request overwrites its causal
prefix and resets the codec. Order reverses each pair to reduce ordering bias.
"""
import argparse
import hashlib
import json
import statistics
import torch
from .common import RESULTS,load_models,stats
from .streaming import StreamingTTS
from .calibrated_backend import install_calibrated,enable_grouped_activation
from .dp4a_fusions import enable_fusions
from .dp4a_gateup import enable_gateup
from .dp4a_packing import install_packing
from .attention_quant import enable_attention_quant
from .decode_buckets import DecodeContextBuckets


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',type=int,default=10);p.add_argument('--tag',required=True);args=p.parse_args()
    if args.pairs<2 or not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Two or more pairs and a safe tag required')
    if (RESULTS/f'native_output_paired_{args.tag}.json').exists():raise FileExistsError('Preserve previous results')
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True);fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10',backend='dp4a')
    enable_grouped_activation(fast);enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json');enable_attention_quant(fast)
    from .attention_native import enable_native_attention
    from .dp4a_gateup_quant import enable_gateup_quant
    from .dp4a_scaled import enable_scaled
    enable_native_attention(fast);enable_gateup_quant(fast);enable_scaled(fast)
    engine.warmup((128,160,256,512))
    control=DecodeContextBuckets(fast);control.warmup()
    from .dp4a_output_native import library as output_library
    output_library()
    for layer in model.language_model.layers:layer.self_attn._native_output=True
    fast.warmup()  # Capture native full-capacity fallback independently.
    native=DecodeContextBuckets(fast);native.warmup()
    label='native_output'
    managers={'control':control,label:native}
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    rows=[]
    for pair in range(-1,args.pairs):
        order=('control',label) if pair%2==0 else (label,'control')
        records={}
        for name in order:
            fast.step=managers[name].step
            torch.manual_seed(7000+pair)
            chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400))
            pcm=torch.cat([c.pcm for c in chunks]);assert torch.isfinite(pcm).all()
            r=engine.last_metrics.copy();assert not r['truncated']
            r['pcm_float32_sha256']=hashlib.sha256(pcm.numpy().tobytes()).hexdigest()
            records[name]=r
        assert records['control']['pcm_float32_sha256']==records[label]['pcm_float32_sha256'],pair
        row={'pair':pair,'seed':7000+pair,'order':order,'records':records,'gain_ms':records['control']['ttfa_ms']-records[label]['ttfa_ms']}
        print({'pair':pair,'control_ms':records['control']['ttfa_ms'],label+'_ms':records[label]['ttfa_ms'],'gain_ms':row['gain_ms'],'pcm_exact':True},flush=True)
        if pair>=0:rows.append(row)
    result={'method':'One model/process, separately captured complete decode graphs and 128/256/512 buckets; reversed order every pair, same seed/reference/text per pair. One warmup pair excluded. Full streaming generation, all 32 codebooks, BF16 prefill, FP32 codec, cached cloned voice; network excluded.',
        'codebooks':32,'rows':rows,'all_full_float32_pcm_exact':True,'control_ttfa':stats([r['records']['control']['ttfa_ms'] for r in rows]),label+'_ttfa':stats([r['records'][label]['ttfa_ms'] for r in rows]),'median_paired_gain_ms':statistics.median(r['gain_ms'] for r in rows),'torch':torch.__version__,'change':'Only the attention-output projection changes from selected Triton scaled INT4 to native CUDA R4/W8.'}
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    (RESULTS/f'native_output_paired_{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    engine.codec.close()


if __name__=='__main__':main()
