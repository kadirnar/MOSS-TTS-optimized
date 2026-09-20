import argparse
import torch
import soundfile as sf
from .common import RESULTS, load_models, save_json, stats
from .streaming import StreamingTTS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=int,default=5)
    parser.add_argument('--no-graph',action='store_true')
    parser.add_argument('--no-fused',action='store_true')
    parser.add_argument('--max-new-tokens',type=int,default=80)
    parser.add_argument('--fp8-mlp',action='store_true')
    args=parser.parse_args()
    model,codec,processor=load_models()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    engine=StreamingTTS(model,codec,processor,graph=not args.no_graph,fused=not args.no_fused,fp8_mlp=args.fp8_mlp)
    engine.warmup()
    name='streaming_eager' if args.no_graph else 'streaming_optimized'
    if args.fp8_mlp:name='streaming_fp8_experimental'
    runs=[]
    for i in range(args.runs+1):
        torch.manual_seed(1234+i)
        chunks=list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=args.max_new_tokens))
        if not chunks:raise RuntimeError('No audio produced')
        metrics=engine.last_metrics.copy()
        print('RUN',i,metrics['ttfa_ms'],metrics['frames'],flush=True)
        if i>0:runs.append(metrics)
        if i==1:
            wave=torch.cat([c.pcm for c in chunks]).numpy()
            sf.write(RESULTS/(name+'.wav'),wave,24000)
        if runs:save_json(name+'.json',{'ttfa':stats([m['ttfa_ms'] for m in runs]),'runs':runs,
            'workload':'Batch 1, 3.112 s cached voice reference, Chinese, all 32 codebooks, 24 kHz mono, 80 ms PCM chunks. Warmup excluded. Request timer includes processor, prefill, sampling, codec and CPU PCM transfer; excludes network.'})

if __name__=='__main__':main()
