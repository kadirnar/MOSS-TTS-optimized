"""Isolate request seed costs with one selected model and complete PCM checks."""
import argparse
import hashlib
import json
import statistics
import time
import torch
from .common import RESULTS,load_models,stats
from .streaming import StreamingTTS
from .calibrated_backend import install_calibrated,enable_grouped_activation
from .dp4a_fusions import enable_fusions
from .dp4a_gateup import enable_gateup
from .dp4a_packing import install_packing
from .attention_quant import enable_attention_quant
from .attention_native import enable_native_attention
from .dp4a_gateup_quant import enable_gateup_quant
from .dp4a_scaled import enable_scaled
from .decode_buckets import DecodeContextBuckets


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v1');args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe nonempty tag required')
    outfile=RESULTS/f'request_seed_{args.tag}.json'
    if outfile.exists():raise FileExistsError('Choose a fresh tag')
    model,codec,processor=load_models();engine=StreamingTTS(model,codec,processor,attention_block=32,fused_residual=True);llm=engine.llm
    install_calibrated(llm,RESULTS/'gptq_v1_g32_d10',backend='dp4a');enable_grouped_activation(llm);enable_fusions(llm,layout_limit=8);enable_gateup(llm)
    install_packing(llm,RESULTS/'dp4a_direct_exact_plan.json');enable_attention_quant(llm);enable_native_attention(llm);enable_gateup_quant(llm);enable_scaled(llm)
    engine.warmup((128,160,256,512));manager=DecodeContextBuckets(llm);manager.warmup();manager.install()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    modes={'all':torch.manual_seed,'cuda':torch.cuda.manual_seed};samples={k:[] for k in modes};records=[]
    for i in range(200):
        for name in (('all','cuda') if i%2==0 else ('cuda','all')):
            begin=time.perf_counter();modes[name](9000+i);samples[name].append((time.perf_counter()-begin)*1000)
    for pair in range(-1,10):
        records_by_mode={};reference_hash=None
        for name in (('all','cuda') if pair%2==0 else ('cuda','all')):
            begin=time.perf_counter();modes[name](9000+pair);seed_ms=(time.perf_counter()-begin)*1000
            chunks=[];first_ms=None
            for chunk in engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400):
                if first_ms is None:first_ms=(time.perf_counter()-begin)*1000
                chunks.append(chunk.pcm)
            pcm=torch.cat(chunks);metrics=engine.last_metrics.copy();assert not metrics['truncated'] and torch.isfinite(pcm).all()
            digest=hashlib.sha256(pcm.numpy().tobytes()).hexdigest()
            if reference_hash is None:reference_hash=digest
            assert digest==reference_hash,(pair,name)
            records_by_mode[name]={'request_through_first_pcm_ms':first_ms,'seed_ms':seed_ms,'pcm_float32_sha256':digest,'engine':metrics}
        row={'pair':pair,'records':records_by_mode,'gain_ms':records_by_mode['all']['request_through_first_pcm_ms']-records_by_mode['cuda']['request_through_first_pcm_ms']}
        if pair>=0:records.append(row)
        print({'pair':pair,'all_ms':records_by_mode['all']['request_through_first_pcm_ms'],'cuda_ms':records_by_mode['cuda']['request_through_first_pcm_ms'],'gain_ms':row['gain_ms'],'pcm_exact':True},flush=True)
    result={'method':'One selected scaled-DP4A model and shared graphs, identical text/voice/seed, reversed request order, one warmup pair excluded. Request clock includes seed call through first complete CPU float32 PCM chunk. 200 standalone seed calls per method after model warmup.','codebooks':32,'all_full_pcm_exact':True,'seed_micro_ms':{k:stats(v) for k,v in samples.items()},'summary':{k:stats([r['records'][k]['request_through_first_pcm_ms'] for r in records]) for k in modes},'median_paired_gain_ms':statistics.median(r['gain_ms'] for r in records),'records':records}
    outfile.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('records','seed_micro_ms')},flush=True);engine.codec.close()


if __name__=='__main__':main()
