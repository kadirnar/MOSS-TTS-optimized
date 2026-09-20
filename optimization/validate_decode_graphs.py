"""Complete-backbone checks while switching decode graph context capacities."""
import json
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
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True)
    fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10',backend='dp4a')
    enable_grouped_activation(fast);enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_packing_plan.json')
    engine.warmup((128,160,256,512))
    buckets=DecodeContextBuckets(fast);buckets.warmup();buckets.install()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    ids=fixture['inputs']['input_ids'].cuda()
    # Populate a long causal prefix so boundary checks use initialized history.
    long_ids=ids.repeat(1,7,1)[:,:1000]
    fast.prefill(long_ids)
    token=fixture['upstream_ids'][10:11][None].cuda()
    rows=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pos in (126,127,128,254,255,256,510,511,512,1023,144,127):
            graph,out=buckets.graphs[fast.max_length]
            fast.graph=graph;fast.next_ids,fast.text_logits,fast.audio_logits=out
            buckets.original_step(token,pos,11,-1)
            expected=fast.audio_logits.clone()
            fast.step(token,pos,11,-1)
            actual=fast.audio_logits.clone()
            relative=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
            assert torch.isfinite(actual).all() and relative<.003,(pos,relative)
            rows.append({'position':pos,'capacity':min(c for c in buckets.graphs if pos<c),
                         'logit_relative_rms':relative,'exact':torch.equal(actual,expected)})
    torch.cuda.current_stream().wait_stream(stream)
    for bad in (-1,1024):
        try:fast.step(token,bad,11,-1)
        except ValueError:pass
        else:raise AssertionError('Out-of-range position accepted')
    result={'private_stream':True,'cases':rows,'invalid_positions_rejected':True,'dispatch':buckets.stats(),
            'scope':'Boundary transitions, return to smaller buckets, and full-capacity fallback with a long initialized causal prefix. Same packed calibrated backbone.'}
    (RESULTS/'decode_graph_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    engine.codec.close()


if __name__=='__main__':main()
