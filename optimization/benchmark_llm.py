import gc
import torch
from .common import RESULTS, load_models, timed, stats, save_json
from .llm import FastLLM
from .kernels import embedding_sum


@torch.inference_mode()
def main():
    model, codec, processor=load_models()
    del codec, processor
    gc.collect(); torch.cuda.empty_cache()
    fixture=torch.load(RESULTS / "fixture.pt",weights_only=True)
    ids=fixture['inputs']['input_ids'].cuda()
    tokens=fixture['upstream_ids'].cuda()
    original=model(input_ids=ids,use_cache=True)
    original_text=original.logits[0][:,-1].clone()
    # Teacher-forced positions: compare logits without sampling divergence.
    teacher=[]
    cache=original.past_key_values
    for i in range(4):
        out=model(input_ids=tokens[i:i+1][None],past_key_values=cache,use_cache=True)
        cache=out.past_key_values
        teacher.append(torch.cat([a[:,-1,:1024] for a in out.logits[1:]],0).clone())
    original_embeddings=model._compute_input_embeddings(ids).clone()
    measurements={}
    for name,fused,graph in [('fused_heads_eager',False,False),('cuda_graph',False,True),('triton_fused_graph',True,True)]:
        fast=FastLLM(model,fused=fused,graph=graph,greedy=True)
        fast.warmup()
        embs=embedding_sum(ids,model.get_input_embeddings().weight,fast.audio_embeddings)
        errors=[]
        prefill,prefill_ms=timed(lambda:fast.prefill(ids))
        for i in range(4):
            fast.step(tokens[i:i+1][None],ids.shape[1]+i,i+1,-1)
            logits=fast.audio_logits if graph else fast._decode()[2]
            errors.append((logits-teacher[i]).abs().max().item())
        for _ in range(3): fast.step(tokens[10:11][None],ids.shape[1]+10,11,-1)
        times=[]
        for _ in range(20):
            _,ms=timed(lambda:fast.step(tokens[10:11][None],ids.shape[1]+10,11,-1))
            times.append(ms)
        measurements[name]={"decode":stats(times),"prefill_ms":prefill_ms,
            "embedding_bitwise_equal":torch.equal(embs,original_embeddings),
            "prefill_logits_max_abs_error":(prefill[0]-original_text).abs().max().item(),
            "teacher_forced_audio_logits_max_abs_error":errors}
        print(name,measurements[name],flush=True)
        save_json('llm_iterations.json',measurements)
        if fused:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                fast.step(tokens[10:11][None],ids.shape[1]+10,11,-1)
            prof.export_chrome_trace(str(RESULTS/'llm_optimized_trace.json'))
            (RESULTS/'llm_optimized_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))
        del fast
        gc.collect(); torch.cuda.empty_cache()

if __name__=='__main__':main()
