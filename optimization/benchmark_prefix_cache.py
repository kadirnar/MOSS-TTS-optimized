"""Voice-prefix reuse: suffix/cross-request correctness and full TTFA."""
import argparse
import json
import torch
import soundfile as sf
from .common import RESULTS,load_models,stats,timed
from .streaming import StreamingTTS
from .prefix_cache import PrefixPrefillCache


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('none','fp8_all'),default='fp8_all')
    parser.add_argument('--prefix-length',type=int,default=96)
    parser.add_argument('--fp8-prefill',action='store_true')
    args=parser.parse_args()
    model,codec,processor=load_models()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    engine=StreamingTTS(model,codec,processor,weight_quantization=args.mode,
        attention_block=32 if args.mode=='fp8_all' else 128,fused_residual=True,fused_gateup=True,fp8_prefill=args.fp8_prefill)
    engine.warmup()
    fast=engine.llm
    cache=PrefixPrefillCache(fast,processor.tokenizer,prefix_length=args.prefix_length,max_entries=2)
    cache.warmup()
    cases=[('original',fixture['text'],fixture['reference'],'Chinese'),
           ('different_text','你好，今天阳光明媚，欢迎来这里做客。',fixture['reference'],'Chinese'),
           ('english','The library opens at nine tomorrow morning. Please bring your card.',fixture['reference'],'English'),
           ('long','今天阳光明媚，我们准备出门散步。'*20,fixture['reference'],'Chinese'),
           ('changed_voice','你好，欢迎使用语音服务。',fixture['reference'].roll(1,0),'Chinese'),
           ('no_voice','你好，欢迎使用语音服务。',None,'Chinese')]
    result={'mode':args.mode,'codebooks':32,'codec_dtype':'float32','torch':torch.__version__,'fp8_prefill':args.fp8_prefill,'checks':[]}
    suffix='_pf8' if args.fp8_prefill else ''
    stream=torch.cuda.Stream()
    for name,text,reference,language in cases:
        inputs=processor([[processor.build_user_message(text=text,reference=None if reference is None else [reference],language=language)]],mode='generation').to('cuda')
        ids=inputs.input_ids
        expected=tuple(v.clone() for v in fast.prefill(ids))
        # Populate on miss; poison request-local KV to require correct restore.
        cache.prefill(ids)
        for layer in fast.cache.layers:layer.keys.zero_();layer.values.zero_()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):actual=cache.prefill(ids)
        torch.cuda.current_stream().wait_stream(stream)
        errors=[((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt()).item() for a,b in zip(actual,expected)]
        assert max(errors)<.05,(name,errors)
        assert all(torch.isfinite(v).all() for v in actual)
        result['checks'].append({'case':name,'prompt_tokens':ids.shape[1],'prefill_relative_rms':errors,'cache':cache.stats()})
        print(result['checks'][-1],flush=True)
    ids=fixture['inputs']['input_ids'].cuda()
    tokens=fixture['upstream_ids'].cuda()
    def logits(prefill):
        prefill(ids)
        rows=[]
        for i in range(36):
            fast.step(tokens[i:i+1][None],ids.shape[1]+i,i+1,-1)
            rows.append(fast.audio_logits.clone())
        return rows
    expected=logits(fast.prefill)
    cache.prefill(ids)
    actual=logits(cache.prefill)
    errors=[];agreements=[]
    for i,(a,b) in enumerate(zip(actual,expected)):
        errors.append(((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt()).item())
        active=min(i+1,32)
        agreements.append((a[:active].argmax(-1)==b[:active].argmax(-1)).float().mean().item())
    result['teacher_forced']={'relative_rms_max':max(errors),'active_top1_mean':sum(agreements)/len(agreements),
        'reference':'Same custom runtime/weights with full prefill; 36 forced steps; diagnostic only'}
    for kind in ('full','prefix'):
        if kind=='prefix':cache.install()
        runs=[]
        for i in range(6):
            torch.manual_seed(1234+i)
            chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400))
            assert chunks and all(c.pcm.numel()==1920 and torch.isfinite(c.pcm).all() for c in chunks)
            if i:runs.append(engine.last_metrics.copy())
            if i==1:sf.write(RESULTS/f'prefix_{args.mode}_p{args.prefix_length}_{kind}{suffix}.wav',torch.cat([c.pcm for c in chunks]).numpy(),24000)
            print(kind,i,engine.last_metrics['ttfa_ms'],engine.last_metrics['truncated'],flush=True)
        result[kind]={'ttfa':stats([r['ttfa_ms'] for r in runs]),'prefill':stats([r['prefill_ms'] for r in runs]),'runs':runs}
    result['cache']=cache.stats()
    result['definition']='Warm complete-text request to first 80 ms CPU PCM chunk, encoded voice reference cached. Prefix contains conditioning only. No audio or user text cache. Six varied prompt/reference cases and 36 teacher-forced steps checked.'
    (RESULTS/f'prefix_cache_{args.mode}_p{args.prefix_length}{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
    print({kind:result[kind]['ttfa']['median_ms'] for kind in ('full','prefix')},result['teacher_forced'],flush=True)
    engine.codec.close()


if __name__=='__main__':main()
