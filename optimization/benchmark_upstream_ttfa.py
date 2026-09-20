"""Measure earliest playable PCM using the unmodified upstream generation loop.

A forward pre-hook observes the previously sampled row. It decodes as soon as
all 32 delayed codebooks exist and stops before the next unused LLM forward.
"""
import time
import torch
from .common import RESULTS,load_models,save_json,stats
from .codec import StreamingCodec


class FirstAudioReady(Exception):pass


@torch.inference_mode()
def main():
    model,codec,processor=load_models()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    decoder=StreamingCodec(codec,graph=False,cached_codebooks=False,triton_attention=False)
    decoder.warmup()
    measurements=[]
    for run in range(6):
        torch.manual_seed(1234+run)
        history=[]
        first_row=None
        decoder.reset()
        torch.cuda.synchronize()
        start=time.perf_counter()
        inputs=processor([[processor.build_user_message(text=fixture['text'],reference=[fixture['reference']],language='Chinese')]],mode='generation').to('cuda')
        measured={}
        def hook(module,args,kwargs):
            nonlocal first_row
            ids=kwargs['input_ids']
            if ids.shape[1]!=1:return
            history.append(ids[0,0].clone())
            if int(ids[0,0,0])==model.config.audio_start_token_id:first_row=len(history)
            if first_row is not None and len(history)>=first_row+32:
                rows=torch.stack(history)
                c=torch.arange(32,device='cuda')
                codes=rows[first_row+c,c+1].reshape(32,1,1)
                assert bool((codes<1024).all())
                before=time.perf_counter()
                pcm=decoder.decode(codes).float().cpu()
                measured.update(ttfa_ms=(time.perf_counter()-start)*1000,
                    codec_ms=(time.perf_counter()-before)*1000,observed_rows=len(history),pcm_samples=pcm.numel())
                raise FirstAudioReady
        handle=model.register_forward_pre_hook(hook,with_kwargs=True)
        try:model.generate(**inputs,max_new_tokens=70)
        except FirstAudioReady:pass
        finally:handle.remove()
        if not measured:raise RuntimeError('No full frame generated')
        print('RUN',run,measured,flush=True)
        if run:measurements.append(measured)
    save_json('upstream_ttfa.json',{'ttfa':stats([m['ttfa_ms'] for m in measurements]),'runs':measurements,
        'definition':'Warm cached-reference request-to-first-1920-PCM-samples. Original upstream sampling and forwards; pre-hook adds incremental emission and stops before an unused next forward.'})
    decoder.close()

if __name__=='__main__':main()
