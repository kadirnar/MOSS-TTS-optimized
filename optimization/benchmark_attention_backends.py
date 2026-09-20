import gc
import torch
from .common import RESULTS,load_models,timed,stats,save_json
from .llm import FastLLM


@torch.inference_mode()
def main():
    model,codec,processor=load_models()
    del codec,processor
    gc.collect();torch.cuda.empty_cache()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    ids=fixture['inputs']['input_ids'].cuda()
    nxt=fixture['upstream_ids'][0:1][None].cuda()
    result={};reference=None
    for name in ['triton','sglang','vllm']:
        try:
            engine=FastLLM(model,greedy=True,attention_backend=name)
            engine.warmup();engine.prefill(ids)
            engine.step(nxt,ids.shape[1],1,-1)
            logits=engine.audio_logits.clone()
            if reference is None:reference=logits
            times=[]
            for _ in range(30):
                _,ms=timed(lambda:engine.step(nxt,ids.shape[1],1,-1));times.append(ms)
            result[name]={'full_llm_decode':stats(times),'logits_max_abs_difference_from_triton':(logits-reference).abs().max().item(),
                'scope':'Actual attention kernels integrated into the complete MOSS LLM; custom scheduler/sampler. Not full serving engine.'}
            del engine
            gc.collect();torch.cuda.empty_cache()
        except Exception as e:result[name]={'error':type(e).__name__+': '+str(e)[:2000]}
        save_json('attention_backends.json',result)

if __name__=='__main__':main()
