"""Real full-backbone comparison of existing and fused quantizer graphs."""
import json
import argparse
import torch
from .common import RESULTS,load_models
from .streaming import StreamingTTS
from .calibrated_backend import install_calibrated,enable_grouped_activation
from .dp4a_fusions import enable_fusions
from .dp4a_gateup import enable_gateup
from .dp4a_packing import install_packing
from .decode_buckets import DecodeContextBuckets


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native-attention',action='store_true');parser.add_argument('--gateup-quant',action='store_true');parser.add_argument('--scaled-dp4a',action='store_true');parser.add_argument('--tag',default='');args=parser.parse_args()
    if sum((args.native_attention,args.gateup_quant,args.scaled_dp4a))>1:raise ValueError('Compare one change at a time')
    if args.tag and not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True)
    fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10',backend='dp4a')
    enable_grouped_activation(fast);enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json')
    if args.native_attention or args.gateup_quant or args.scaled_dp4a:
        from .attention_quant import enable_attention_quant
        from .attention_native import library
        enable_attention_quant(fast);library()
    if args.gateup_quant or args.scaled_dp4a:
        from .attention_native import enable_native_attention
        enable_native_attention(fast)
    if args.scaled_dp4a:
        from .dp4a_gateup_quant import enable_gateup_quant
        enable_gateup_quant(fast)
    engine.warmup((128,160,256,512))
    baseline=DecodeContextBuckets(fast);baseline.warmup()
    # Existing captured graphs keep their original operations. Capture separate
    # candidate graphs with the new operator; do not mutate graph internals.
    for layer in model.language_model.layers:
        if args.scaled_dp4a:
            from .dp4a_scaled import SELECTED
            layer.self_attn._scaled_dp4a={name:dict(SELECTED[name]) for name in ('qkv','out')}
            layer.mlp._scaled_dp4a={name:dict(SELECTED[name]) for name in ('up','down')}
        elif args.gateup_quant:layer.mlp._fused_gateup_quant=True
        elif args.native_attention:layer.self_attn._native_attention=True
        else:layer.self_attn._quantize_attention_output=True
    candidate={}
    for cap in (128,256,512,1024):
        for layer in model.language_model.layers:layer.self_attn._decode_capacity=cap
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):fast._decode()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=fast._decode()
        candidate[cap]=(graph,out)
    for layer in model.language_model.layers:layer.self_attn._decode_capacity=None
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    ids=fixture['inputs']['input_ids'].cuda()
    fast.prefill(ids.repeat(1,8,1)[:,:1023])
    token=fixture['upstream_ids'][10:11][None].cuda()
    rows=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pos in (0,1,31,32,126,127,128,254,255,256,510,511,512,1023,144,127):
            cap=min(c for c in candidate if pos<c)
            graph,out=baseline.graphs[cap]
            fast.graph=graph;fast.next_ids,fast.text_logits,fast.audio_logits=out
            fast.step(token,pos,11,-1)
            expected=fast.audio_logits.clone();expected_text=fast.text_logits.clone()
            graph,out=candidate[cap]
            fast.graph=graph;fast.next_ids,fast.text_logits,fast.audio_logits=out
            fast.step(token,pos,11,-1)
            actual=fast.audio_logits.clone();actual_text=fast.text_logits.clone()
            assert torch.equal(actual,expected) and torch.equal(actual_text,expected_text),pos
            rows.append({'position':pos,'capacity':cap,'audio_logits_exact':True,'text_logits_exact':True})
    torch.cuda.current_stream().wait_stream(stream)
    result={'private_stream':True,'cases':rows,'codebooks':32,'scope':'All text and audio logits with separate baseline/fused graphs; initialized 1023-token causal prefix, capacity boundaries, full fallback and return to earlier positions.'}
    result['native_attention']=args.native_attention
    result['gateup_quant']=args.gateup_quant
    result['tested_change']='gateup_quant' if args.gateup_quant else ('native_attention' if args.native_attention else 'attention_quant')
    result['baseline_native_attention']=args.gateup_quant or args.scaled_dp4a
    result['candidate_native_attention']=args.gateup_quant or args.native_attention or args.scaled_dp4a
    name='attention_native_graph_validation.json' if args.native_attention else 'attention_quant_graph_validation.json'
    if args.gateup_quant:name='gateup_quant_graph_validation.json'
    result['scaled_dp4a']=args.scaled_dp4a
    if args.scaled_dp4a:
        name='dp4a_scaled_graph_validation.json'
        result['tested_change']='scaled_dp4a'
        result['baseline_gateup_quant']=result['candidate_gateup_quant']=True
    if args.tag:name=name.removesuffix('.json')+'_'+args.tag+'.json'
    (RESULTS/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    engine.codec.close()


if __name__=='__main__':main()
