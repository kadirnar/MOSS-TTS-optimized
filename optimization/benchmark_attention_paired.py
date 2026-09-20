"""Alternate old/native attention graphs inside one process and one model.

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
from .attention_native import library
from .decode_buckets import DecodeContextBuckets


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',type=int,default=10);p.add_argument('--tag',required=True);p.add_argument('--gateup-quant',action='store_true');p.add_argument('--wide-qkv',action='store_true');p.add_argument('--scaled-dp4a',action='store_true');args=p.parse_args()
    if sum((args.gateup_quant,args.wide_qkv,args.scaled_dp4a))>1:raise ValueError('Choose one isolated change to compare')
    if args.pairs<2 or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('At least two pairs and a safe tag required')
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True);fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10',backend='dp4a')
    enable_grouped_activation(fast);enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json');enable_attention_quant(fast)
    if args.gateup_quant or args.wide_qkv or args.scaled_dp4a:
        from .attention_native import enable_native_attention
        enable_native_attention(fast)
    if args.wide_qkv or args.scaled_dp4a:
        from .dp4a_gateup_quant import enable_gateup_quant
        enable_gateup_quant(fast)
    engine.warmup((128,160,256,512))
    control=DecodeContextBuckets(fast);control.warmup()
    library()
    for layer in model.language_model.layers:
        if args.scaled_dp4a:
            from .dp4a_scaled import SELECTED
            layer.self_attn._scaled_dp4a={name:dict(SELECTED[name]) for name in ('qkv','out')}
            layer.mlp._scaled_dp4a={name:dict(SELECTED[name]) for name in ('up','down')}
        elif args.wide_qkv:layer.self_attn._dp4a_packing['qkv']=dict(layer.self_attn._dp4a_packing['qkv'],rows=8,warps=8)
        elif args.gateup_quant:layer.mlp._fused_gateup_quant=True
        else:layer.self_attn._native_attention=True
    fast.warmup()  # Capture native full-capacity fallback independently.
    native=DecodeContextBuckets(fast);native.warmup()
    label='scaled' if args.scaled_dp4a else ('wide' if args.wide_qkv else ('fused' if args.gateup_quant else 'native'))
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
        'codebooks':32,'rows':rows,'all_full_float32_pcm_exact':True,'control_ttfa':stats([r['records']['control']['ttfa_ms'] for r in rows]),label+'_ttfa':stats([r['records'][label]['ttfa_ms'] for r in rows]),'median_paired_gain_ms':statistics.median(r['gain_ms'] for r in rows),'torch':torch.__version__,'gateup_quant':args.gateup_quant}
    if args.gateup_quant:result['comparison']='Both configurations use native attention; candidate additionally fuses gate/up/SiLU and grouped activation quantization.'
    if args.wide_qkv:result['comparison']='Both configurations use native attention and gate/up quantizer fusion; candidate changes only QKV tile R4/W4 to R8/W8.'
    if args.scaled_dp4a:result['comparison']='Both configurations use native attention and fused gate/up quantization; candidate adds scaled-integer INT4 unpacking in all four projection families.'
    result['scaled_dp4a']=args.scaled_dp4a
    result['wide_qkv']=args.wide_qkv
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    (RESULTS/f'attention_paired_{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    engine.codec.close()


if __name__=='__main__':main()
