"""Exact codec cache/state validation, private streams and rotary-table limits."""
import argparse
import gc
import hashlib
import json

import torch
from moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerModel
from .common import RESULTS
from .codec import StreamingCodec
from .codec_clock import ClockedStreamingCodec


def attention_states(codec):
    rows=[];tokens=1
    for i,module in enumerate(codec.decoder):
        if hasattr(module,'transformer'):
            for j,layer in enumerate(module.transformer.layers):rows.append((f'{i}/{j}',tokens,layer.self_attn._streaming_state))
        elif hasattr(module,'patch_size'):tokens*=module.patch_size
    return rows


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--frames',type=int,default=140);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.frames not in (4,140):raise ValueError('Safe tag and four sanitizer or 140 wrap frames required')
    path=RESULTS/f'codec_clock_validation_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);torch.manual_seed(1977)
    codec=MossAudioTokenizerModel.from_pretrained('/workspace/models/moss-codec',dtype=torch.float32,device_map='cuda').eval()
    saved=torch.load(RESULTS/'fixture.pt',weights_only=True)['codes'].cuda()
    codes=saved.repeat(1,1,(a.frames+saved.shape[-1]-1)//saved.shape[-1])[...,:a.frames].contiguous()
    probes=[torch.zeros_like(codes[...,:1]),torch.full_like(codes[...,:1],1023),torch.randint(0,1024,(32,1,1),device='cuda')]
    variants={};reference=None;reference_cache=None
    for name in ('control','candidate'):
        decoder=StreamingCodec(codec) if name=='control' else ClockedStreamingCodec(codec)
        decoder.warmup();states=attention_states(codec)
        if a.frames==140:assert all(a.frames*t>s.kv_cache.capacity for _,t,s in states)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());outputs=[]
        with torch.cuda.stream(stream):
            decoder.reset()
            for _,_,state in states:state.kv_cache.cache.fill_(float('nan'))
            for index in range(a.frames):outputs.append(decoder.decode(codes[...,index:index+1]))
            # Entire KV stores are compared as bytes, including identical NaN
            # poison in slots not visited by the short sanitizer workload.
            cache=[state.kv_cache.cache.cpu().clone() for _,_,state in states]
            for probe in probes:
                decoder.reset()
                outputs.extend((decoder.decode(probe),decoder.decode(codes[...,:1])))
            # Both paths start from identical synthetic history near the last
            # allowed RoPE position, then reject the next frame before launch.
            decoder.reset();decoder.frames=4095
            for _,tokens,state in states:
                state.kv_cache.cache.fill_(.125);state.offset.fill_(4095*tokens)
            if name=='candidate':
                for tokens,offset in decoder.clock_by_t.items():offset.fill_(4095*tokens)
            outputs.append(decoder.decode(codes[...,:1]))
            try:decoder.decode(codes[...,:1])
            except ValueError:limit_rejected=True
            else:raise AssertionError('Rotary table limit not enforced')
            wave=torch.cat(outputs,-1);assert bool(torch.isfinite(wave).all())
            if reference is None:reference=wave.clone();reference_cache=cache
            mismatches=int((wave!=reference).sum())
            cache_mismatches=[int((x.view(torch.int32)!=y.view(torch.int32)).sum()) for x,y in zip(cache,reference_cache,strict=True)]
            assert mismatches==0 and not any(cache_mismatches),(name,mismatches,sum(cache_mismatches))
            if name=='candidate':
                assert all(int(offset)==4096*t for t,offset in decoder.clock_by_t.items())
        torch.cuda.current_stream().wait_stream(stream)
        variants[name]={'frames_compared':a.frames+7,'wave_mismatches':mismatches,'cache_mismatches':cache_mismatches,
                        'rotary_limit_rejected':limit_rejected,'finite':True,'private_stream':True,
                        'capacities':[{ 'layer':label,'tokens_per_frame':t,'capacity':state.kv_cache.capacity} for label,t,state in states],
                        'wave_sha256':hashlib.sha256(wave.flatten().cpu().numpy().tobytes()).hexdigest()}
        print('PASS',name,a.frames+7,'frames',len(states),'caches',flush=True)
        decoder.close();del decoder,outputs,states;gc.collect();torch.cuda.empty_cache()
    result={'codebooks':32,'variants':variants,'all_exact':True,'wraparound_tested':a.frames==140,
            'scope':'Frozen real codes, all-zero/all-1023/random reset probes, poisoned unused cache slots, all 68 entire KV buffers, '
                    'first and steady graphs on a private stream, last valid rotary frame and pre-launch capacity rejection. '
                    'Four-frame sanitizer mode does not cover wraparound; separate 140-frame run does.'}
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
