"""FlashInfer attention correctness and full-LLM decode, not server throughput."""
import gc
import json
import torch
from .common import RESULTS,load_models,timed,stats
from .attention_backends import FlashInferDecodeBackend
from .llm import FastLLM


@torch.inference_mode()
def validate(tensor_cores):
    backend=FlashInferDecodeBackend(1024,tensor_cores)
    q=torch.randn(32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,8,1024,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    pos=torch.zeros(1,device='cuda',dtype=torch.long)
    for _ in range(3):backend(q,k,v,pos)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):backend(q,k,v,pos)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=backend(q,k,v,pos)
    errors=[]
    for p in (0,63,127,128,255,511,767,1023,17):
        with torch.cuda.stream(stream):
            pos.fill_(p)
            graph.replay()
            ref=torch.nn.functional.scaled_dot_product_attention(q.view(1,32,1,128),k[:,:,:p+1],v[:,:,:p+1],enable_gqa=True).reshape(1,1,4096)
            err=(actual-ref).abs().max().item()
            rel=((actual.float()-ref.float()).square().mean().sqrt()/ref.float().square().mean().sqrt()).item()
        errors.append({'position':p,'max_abs':err,'relative_rms':rel})
        assert err<0.04 and rel<0.01, errors[-1]
    torch.cuda.current_stream().wait_stream(stream)
    return {'positions':errors,'graph_replay':True,'non_default_stream':True}


@torch.inference_mode()
def main():
    result={}
    names=['triton']
    for name,tc in [('flashinfer',False),('flashinfer_tc',True)]:
        try:
            result[name]={'attention_validation':validate(tc)}
            names.append(name)
        except Exception as e:
            result[name]={'error':type(e).__name__+': '+str(e)[:2000]}
        print(name,result[name],flush=True)
        (RESULTS/'flashinfer.json').write_text(json.dumps(result,indent=2)+'\n')
    model,codec,processor=load_models()
    del codec,processor
    gc.collect();torch.cuda.empty_cache()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    ids=fixture['inputs']['input_ids'].cuda()
    nxt=fixture['upstream_ids'][0:1][None].cuda()
    reference=None
    for name in names:
        try:
            fast=FastLLM(model,greedy=True,attention_backend=name)
            fast.warmup();fast.prefill(ids)
            fast.step(nxt,ids.shape[1],1,-1)
            actual=fast.audio_logits.clone()
            if reference is None:reference=actual
            samples=[]
            for i in range(33):
                _,ms=timed(lambda:fast.step(nxt,ids.shape[1],1,-1))
                if i>=3:samples.append(ms)
            result.setdefault(name,{}).update({'full_llm_decode':stats(samples),
                'max_logit_difference_from_triton':(actual-reference).abs().max().item(),
                'relative_rms_logits':((actual.float()-reference.float()).square().mean().sqrt()/reference.float().square().mean().sqrt()).item(),
                'scope':'Full BF16 LLM, custom scheduler and sampler; no serving-engine comparison'})
            del fast
            gc.collect();torch.cuda.empty_cache()
        except Exception as e:
            result.setdefault(name,{})['error']=type(e).__name__+': '+str(e)[:2000]
        print(name,result[name],flush=True)
        (RESULTS/'flashinfer.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
