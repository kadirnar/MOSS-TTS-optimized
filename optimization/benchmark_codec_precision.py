"""Measure TF32/high matmul policy without changing FP32 codec storage."""
import gc
import json
import torch
from .common import RESULTS,load_models,timed,stats
from .codec import StreamingCodec


@torch.inference_mode()
def main():
    model,codec,processor=load_models()
    del model,processor
    gc.collect();torch.cuda.empty_cache()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    codes=fixture['codes'].cuda().repeat(1,1,4)[...,:140].contiguous()
    original_policy=torch.get_float32_matmul_precision()
    result={'codebooks':32,'storage_dtype':'float32','torch':torch.__version__,'cases':{}}
    reference=None
    for policy in ('highest','high','medium'):
        torch.set_float32_matmul_precision(policy)
        engine=StreamingCodec(codec)
        engine.warmup()
        waves=[];times=[]
        for i in range(codes.shape[-1]):
            wave,ms=timed(lambda:engine.decode(codes[...,i:i+1]))
            waves.append(wave);times.append(ms)
        wave=torch.cat(waves,-1)
        if reference is None:reference=wave.clone()
        difference=wave-reference
        snr=(10*torch.log10(reference.square().sum()/difference.square().sum().clamp_min(1e-30))).item()
        engine.reset()
        first=engine.decode(codes[...,:1])
        case={'decode':stats(times),'waveform_snr_db':snr,'max_abs_error':difference.abs().max().item(),
            'ring_wrap_frames':140,'reset_max_abs_error':(first-wave[...,:1920]).abs().max().item(),
            'finite':bool(torch.isfinite(wave).all())}
        result['cases'][policy]=case
        print(policy,case['decode']['median_ms'],snr,flush=True)
        (RESULTS/'codec_precision.json').write_text(json.dumps(result,indent=2)+'\n')
        engine.close()
        del engine,waves,wave
        gc.collect()
    torch.set_float32_matmul_precision(original_policy)


if __name__=='__main__':main()
