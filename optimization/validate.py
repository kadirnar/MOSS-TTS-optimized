"""Numerical, ring-wrap, reset and non-default CUDA stream checks."""
from contextlib import ExitStack
import gc
import torch
from .common import RESULTS,load_models,save_json
from .codec import StreamingCodec
from .llm import FastLLM


@torch.inference_mode()
def main():
    model,codec,processor=load_models()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    codes=fixture['codes'].cuda().repeat(1,1,3)[...,:140].contiguous()
    # 140 frames crosses the codec's ten-second (125-frame) ring capacity.
    original=StreamingCodec(codec,graph=True,cached_codebooks=False,triton_attention=False)
    original.warmup()
    reference=[]
    for i in range(codes.shape[-1]):reference.append(original.decode(codes[...,i:i+1]))
    reference=torch.cat(reference,-1)
    original.close();del original
    opt=StreamingCodec(codec,graph=True,cached_codebooks=True,triton_attention=True)
    opt.warmup()
    actual=[]
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        opt.reset()
        for i in range(codes.shape[-1]):actual.append(opt.decode(codes[...,i:i+1]))
        actual=torch.cat(actual,-1)
    torch.cuda.current_stream().wait_stream(stream)
    diff=actual-reference
    codec_metrics={'frames':codes.shape[-1], 'ring_wrap_tested':True,
        'non_default_stream_tested':True, 'max_abs_error':diff.abs().max().item(),
        'snr_db':(10*torch.log10(reference.square().sum()/diff.square().sum().clamp_min(1e-30))).item(),
        'finite':bool(torch.isfinite(actual).all())}
    opt.reset()
    reset=opt.decode(codes[...,:1])
    codec_metrics['reset_first_frame_error']=(reset-actual[...,:1920]).abs().max().item()
    assert codec_metrics['finite']
    assert codec_metrics['snr_db']>60,codec_metrics
    assert codec_metrics['reset_first_frame_error']<1e-6,codec_metrics
    opt.close()
    ids=fixture['inputs']['input_ids'].cuda()
    tokens=fixture['upstream_ids'].cuda()
    upstream=model(input_ids=ids,use_cache=True)
    cache=upstream.past_key_values
    logits=[]
    for i in range(36):
        out=model(input_ids=tokens[i:i+1][None],past_key_values=cache,use_cache=True)
        cache=out.past_key_values
        logits.append(torch.cat([a[:,-1,:1024] for a in out.logits[1:]],0).clone())
    fast=FastLLM(model,greedy=True)
    fast.warmup();fast.capture_prefill()
    fast.prefill(ids)
    rms=[]; maxima=[]; choices=[]; active_choices=[]; active_kl=[]
    for i,ref in enumerate(logits):
        fast.step(tokens[i:i+1][None],ids.shape[1]+i,i+1,-1)
        value=fast.audio_logits
        rms.append(((value.float()-ref.float()).square().mean().sqrt()/ref.float().square().mean().sqrt()).item())
        maxima.append((value-ref).abs().max().item())
        choices.append((value.argmax(-1)==ref.argmax(-1)).float().mean().item())
        active=min(i+1,32)
        active_choices.append((value[:active].argmax(-1)==ref[:active].argmax(-1)).float().mean().item())
        p=(ref[:active].float()/1.7).softmax(-1)
        active_kl.append((p*((ref[:active].float()/1.7).log_softmax(-1)-(value[:active].float()/1.7).log_softmax(-1))).sum(-1).mean().item())
    assert max(rms)<0.02,(rms,maxima)
    save_json('validation.json',{'codec':codec_metrics,'llm_teacher_forced':{'steps':36,
        'max_relative_rms_error':max(rms),'max_absolute_logit_error':max(maxima),
        'mean_top1_agreement':sum(choices)/len(choices),
        'active_codebook_mean_top1_agreement':sum(active_choices)/len(active_choices),
        'active_codebook_mean_kl_divergence':sum(active_kl)/len(active_kl),
        'note':'Floating point operation ordering changes logits; this is numerical validation, not a corpus-level WER or speaker-similarity evaluation.'}})

if __name__=='__main__':main()
