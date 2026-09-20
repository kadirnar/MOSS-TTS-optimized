"""Audit fused normalization and quantization on all real decode activations."""
import argparse
import json
import torch
from .common import RESULTS,load_models
from .streaming import StreamingTTS
from .calibrated_backend import install_calibrated,enable_grouped_activation
from .kernels import rmsnorm,add_rmsnorm
from .validate_dp4a_fusions import quant
from . import dp4a_fusions


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--layout-limit',type=int,choices=(0,8),default=0)
    args=parser.parse_args()
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True)
    install_calibrated(engine.llm,RESULTS/'gptq_v1_g32_d10',backend='dp4a')
    enable_grouped_activation(engine.llm)
    dp4a_fusions.enable_fusions(engine.llm,layout_limit=args.layout_limit)
    counts=torch.zeros(72,4,device='cuda',dtype=torch.int64)
    max_errors=torch.zeros(72,device='cuda')
    call=0
    original=dp4a_fusions.norm_quant
    def checked(x,residual,weight,eps,group=0,layout_limit=0):
        nonlocal call
        index=call%72;call+=1
        summed,y,pair=original(x,residual,weight,eps,group,layout_limit)
        if residual is None:ref_sum=x;ref_y=rmsnorm(x,weight,eps)
        else:ref_sum,ref_y=add_rmsnorm(x,residual,weight,eps)
        ref_pair=quant(ref_y,group)
        for column,(a,b) in enumerate(zip((summed,y,*pair),(ref_sum,ref_y,*ref_pair))):
            counts[index,column].add_(torch.count_nonzero(a!=b))
        max_errors[index].copy_(torch.maximum(max_errors[index],(y.float()-ref_y.float()).abs().max()))
        return summed,y,pair
    dp4a_fusions.norm_quant=checked
    engine.warmup()
    counts.zero_();max_errors.zero_()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    records=[]
    for seed in (1235,1236):
        torch.manual_seed(seed)
        chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400))
        records.append({'seed':seed,'frames':len(chunks),'truncated':engine.last_metrics['truncated']})
    result={'scope':'Two complete sampled cloned-voice utterances. Fused and unfused producers evaluated on identical activations inside each graph replay. Instrumented timing is not a performance measurement.',
            'codebooks':32,'group':32,'layout_limit':args.layout_limit,'columns':['residual','normalized_bf16','int8_codes','fp32_scales'],
            'per_norm_counts':counts.cpu().tolist(),'max_bf16_abs_errors':max_errors.cpu().tolist(),'runs':records}
    result['total_counts']=counts.sum(0).cpu().tolist()
    name='dp4a_real_fusion_validation'+(f'_layout{args.layout_limit}' if args.layout_limit else '')
    (RESULTS/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('TOTAL',result['total_counts'],'MAX',max(result['max_bf16_abs_errors']),flush=True)
    engine.codec.close()


if __name__=='__main__':main()
