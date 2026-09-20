"""Sequential FP32 codec variants, cache wraps, reset, and first-frame timing."""
import argparse
import gc
import hashlib
import json

import torch
from moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerModel
from .common import RESULTS,stats,timed
from .codec import StreamingCodec
from .codec_clock import ClockedStreamingCodec


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--frames',type=int,default=140);p.add_argument('--runs',type=int,default=3)
    p.add_argument('--stage-clocks',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.frames<140 or a.runs<2:raise ValueError('Safe tag, 140 wrap frames and two runs required')
    path=RESULTS/f'codec_clock_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);torch.manual_seed(927)
    codec=MossAudioTokenizerModel.from_pretrained('/workspace/models/moss-codec',dtype=torch.float32,device_map='cuda').eval()
    saved=torch.load(RESULTS/'fixture.pt',weights_only=True)['codes'].cuda()
    codes=saved.repeat(1,1,(a.frames+saved.shape[-1]-1)//saved.shape[-1])[...,:a.frames].contiguous()
    probes=[codes[...,:1],torch.zeros_like(codes[...,:1]),torch.full_like(codes[...,:1],1023),torch.randint(0,1024,(32,1,1),device='cuda')]
    result={'codebooks':32,'torch':torch.__version__,'codec_dtype':'float32','frames':a.frames,'runs':a.runs,'cases':{},
            'method':'One unchanged FP32 codec; wrappers close before next variant. Frozen full-codebook fixture repeated through all attention-cache wraps. '
                     'Three reset streams and four changed first-frame code vectors. CUDA synchronized wall timing; standalone codec, not TTFA. '
                     'First-frame graph reused only immediately after reset. Original reference encoder and quantizer retained.'}
    reference=None;reference_probes=None
    variants=(('control',None),('clock',{'first_frame':False,'stage_clocks':False,'compact_first':True}),('first',{'first_frame':True,'stage_clocks':False,'compact_first':True}),('kv_first',{'first_frame':True,'kv_only':True,'stage_clocks':False,'compact_first':True}))
    if a.stage_clocks:
        variants=(('control',None),('stage_clock',{'stage_clocks':True,'first_frame':False}),
                  ('stage_first',{'stage_clocks':True,'first_frame':True,'compact_first':False}),
                  ('stage_first_compact',{'stage_clocks':True,'first_frame':True,'compact_first':True}),
                  ('stage_kv_first',{'stage_clocks':True,'first_frame':True,'kv_only':True,'compact_first':False}))
    for name,options in variants:
        decoder=StreamingCodec(codec) if options is None else ClockedStreamingCodec(codec,**options)
        decoder.warmup();times=[];first_times=[];waves=[];initial=[]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(a.runs):
                decoder.reset();chunks=[]
                for i in range(a.frames):
                    wave,ms=timed(lambda:decoder.decode(codes[...,i:i+1]));chunks.append(wave)
                    times.append(ms)
                    if i==0:first_times.append(ms)
                waves.append(torch.cat(chunks,-1))
            for probe in probes:
                decoder.reset();initial.append(decoder.decode(probe))
        torch.cuda.current_stream().wait_stream(stream)
        if reference is None:reference=waves[0].clone();reference_probes=[x.clone() for x in initial]
        diff=waves[0]-reference
        case={'decode':stats(times),'first_frame':stats(first_times),'wave_mismatches':int((waves[0]!=reference).sum()),
              'wave_snr_db':float(10*torch.log10(reference.square().sum()/diff.square().sum().clamp_min(1e-30))),
              'wave_max_abs_error':float(diff.abs().max()),'finite':bool(torch.isfinite(waves[0]).all()),
              'reset_exact':all(torch.equal(x,waves[0]) for x in waves[1:]),
              'first_probe_mismatches':[int((x-y).abs().ne(0).sum()) for x,y in zip(initial,reference_probes,strict=True)],
              'pcm_float32_sha256':hashlib.sha256(waves[0].flatten().cpu().numpy().tobytes()).hexdigest()}
        result['cases'][name]=case;path.write_text(json.dumps(result,indent=2)+'\n')
        print(name,{k:v if not isinstance(v,dict) else v['median_ms'] for k,v in case.items()},flush=True)
        decoder.close();del decoder,waves,initial;gc.collect();torch.cuda.empty_cache()
    # Initialize CUPTI only after every timed variant: its driver hooks may
    # otherwise perturb subsequent graph timings even after profiling exits.
    for name,options in variants:
        decoder=StreamingCodec(codec) if options is None else ClockedStreamingCodec(codec,**options)
        decoder.warmup();decoder.reset()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            decoder.decode(codes[...,:1]);decoder.decode(codes[...,1:2]);torch.cuda.synchronize()
        prof.export_chrome_trace(str(RESULTS/f'codec_clock_{a.tag}_{name}_trace.json'))
        decoder.close();del decoder;gc.collect();torch.cuda.empty_cache()
    print('SAVED',path,flush=True)


if __name__=='__main__':main()
