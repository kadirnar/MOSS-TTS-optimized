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
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    if (RESULTS/f'audio_head_graph_validation_{args.tag}.json').exists():raise FileExistsError('Preserve results')
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True)
    fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10',backend='dp4a')
    enable_grouped_activation(fast);enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json')
    from .attention_quant import enable_attention_quant
    from .attention_native import enable_native_attention
    from .dp4a_gateup_quant import enable_gateup_quant
    from .dp4a_scaled import enable_scaled
    enable_attention_quant(fast);enable_native_attention(fast);enable_gateup_quant(fast);enable_scaled(fast)
    from .dp4a_norm_projection import enable_norm_projection
    enable_norm_projection(fast)
    engine.warmup((128,160,256,512))
    baseline=DecodeContextBuckets(fast);baseline.warmup()
    from .audio_head_buckets import DecodeAudioHeadBuckets
    candidate=DecodeAudioHeadBuckets(fast,baseline);candidate.warmup()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    ids=fixture['inputs']['input_ids'].cuda()
    fast.prefill(ids.repeat(1,8,1)[:,:1023])
    token=fixture['upstream_ids'][10:11][None].cuda()
    rows=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pos in (0,1,31,32,126,127,128,254,255,256,510,511,512,1023,144,127):
            for length in (0,1,7,8,9,15,16,17,23,24,25,31,32,33):
                for delay in (-1,0,17,31,32):
                    torch.cuda.manual_seed(3927)
                    expected_ids=baseline.step(token,pos,length,delay).clone()
                    expected=fast.audio_logits.clone();expected_text=fast.text_logits.clone()
                    torch.cuda.manual_seed(3927)
                    actual_ids=candidate.step(token,pos,length,delay).clone()
                    actual=fast.audio_logits.clone();actual_text=fast.text_logits.clone()
                    active=(fast.channels<length)&((delay<0)|(fast.channels>=delay))
                    exact=torch.equal(actual[active],expected[active]) and torch.equal(actual_text,expected_text) and torch.equal(actual_ids,expected_ids)
                    assert exact,(pos,length,delay)
                    rows.append({'position':pos,'audio_length':length,'delay_length':delay,'active_logits_exact':True,'text_logits_exact':True,'sampled_ids_exact':True})
    torch.cuda.current_stream().wait_stream(stream)
    result={'private_stream':True,'cases':rows,'codebooks':32,'scope':'All text and active audio logits plus sampled IDs with separate full/prefix head graphs; initialized 1023-token causal prefix, capacity boundaries, full fallback and return to earlier positions.'}
    result['tested_change']='audio_head_prefixes'
    name=f'audio_head_graph_validation_{args.tag}.json'
    (RESULTS/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    engine.codec.close()


if __name__=='__main__':main()
