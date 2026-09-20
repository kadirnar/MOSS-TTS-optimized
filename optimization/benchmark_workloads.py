import argparse
import time
import torch
import soundfile as sf
from .common import ROOT,RESULTS,load_models,stats,save_json
from .streaming import StreamingTTS
from .reference_encoder import ReferenceEncoder


def first_chunk(engine,text,reference,language):
    gen=engine.stream(text,reference,language,max_new_tokens=400)
    try:return next(gen).elapsed_ms
    finally:gen.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='workloads')
    parser.add_argument('--mode',choices=('none','fp8_all'),default='none')
    parser.add_argument('--attention-block',type=int,default=128)
    parser.add_argument('--fused-residual',action='store_true')
    parser.add_argument('--fused-gateup',action='store_true')
    args=parser.parse_args()
    if not all(c.isalnum() or c=='_' for c in args.name):raise ValueError('Unsafe name')
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,weight_quantization=args.mode,
        attention_block=args.attention_block,fused_residual=args.fused_residual,fused_gateup=args.fused_gateup)
    engine.warmup()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    cases=[('zh_short','你好，欢迎使用语音服务。','Chinese'),
           ('zh_baseline',fixture['text'],'Chinese'),
           ('en_short','Hello, this is a streaming voice cloning test.','English'),
           ('en_long','The library opens at nine in the morning. Please bring your card, and ask the librarian if you need help finding a book. We hope you enjoy your visit.','English')]
    result={'configuration':{**vars(args),'torch':torch.__version__,'codebooks':32,'codec_dtype':'float32'}}
    for name,text,language in cases:
        first_chunk(engine,text,fixture['reference'],language)
        values=[]
        for i in range(5):
            torch.manual_seed(400+i)
            values.append(first_chunk(engine,text,fixture['reference'],language))
        result[name]={'ttfa':stats(values),'text':text,'reference':'cached 3.112-second Chinese voice reference'}
        save_json(args.name+'.json',result)
    wave,sr=sf.read(ROOT/'assets/audio/reference_zh.wav',dtype='float32')
    wave=torch.from_numpy(wave).reshape(1,-1)
    uncached=[];encoding=[]
    for i in range(4):
        torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():reference=processor.encode_audios_from_wav([wave],sr)[0]
        encoded=(time.perf_counter()-start)*1000
        first_chunk(engine,fixture['text'],reference,'Chinese')
        total=(time.perf_counter()-start)*1000
        if i:uncached.append(total);encoding.append(encoded)
    result['uncached_reference']={'ttfa':stats(uncached),'reference_encode':stats(encoding),
        'definition':'Timer starts with decoded CPU reference waveform; includes reference encoding and full synthesis-to-first-PCM. Excludes network and file parsing.'}
    encoder=ReferenceEncoder(processor)
    encoder.warmup()
    uncached=[];encoding=[]
    for i in range(4):
        torch.cuda.synchronize()
        start=time.perf_counter()
        reference=encoder.encode(wave,sr)
        encoded=(time.perf_counter()-start)*1000
        first_chunk(engine,fixture['text'],reference,'Chinese')
        total=(time.perf_counter()-start)*1000
        if i:uncached.append(total);encoding.append(encoded)
    result['uncached_reference_optimized_encoder']={'ttfa':stats(uncached),'reference_encode':stats(encoding),
        'definition':'Same uncached-reference timing, with the validated FP32 CUDA graph encoder.'}
    save_json(args.name+'.json',result)
    engine.codec.close()

if __name__=='__main__':main()
