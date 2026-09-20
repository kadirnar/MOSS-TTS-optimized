"""Lossless allocation comparison for the FP32 codec and BF16 audio heads."""
import argparse
import gc
import json
import statistics
import time

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerModel
from .common import RESULTS,CODEC_REVISION,TTS_REVISION,stats
from .compressed_alloc import clone,library
from .codec import StreamingCodec
from .tune_weight_reads import measure


@torch.inference_mode()
def codec_benchmark(rounds):
    path=snapshot_download('OpenMOSS-Team/MOSS-Audio-Tokenizer',revision=CODEC_REVISION,local_files_only=True)
    codec=MossAudioTokenizerModel.from_pretrained(path,dtype=torch.float32,device_map='cuda').eval()
    decoder=StreamingCodec(codec);swaps=[];allocations=[]
    roots=(codec.decoder,codec.quantizer.output_proj)
    for root_index,root in enumerate(roots):
        for module_name,module in root.named_modules():
            for name,parameter in module.named_parameters(recurse=False):
                alternatives={'torch':parameter}
                for variant in ('vmm_plain','compressed'):
                    value,meta=clone(parameter,compressed=variant=='compressed')
                    assert torch.equal(value,parameter)
                    alternatives[variant]=torch.nn.Parameter(value,requires_grad=False)
                    allocations.append({'name':f'{root_index}.{module_name}.{name}','variant':variant,**meta})
                swaps.append((module,name,alternatives))
    alternatives={'torch':codec.quantizer._projected_codebooks}
    for variant in ('vmm_plain','compressed'):
        value,meta=clone(alternatives['torch'],compressed=variant=='compressed')
        assert torch.equal(value,alternatives['torch']);alternatives[variant]=value
        allocations.append({'name':'projected_codebooks','variant':variant,**meta})
    swaps.append((codec.quantizer,'_projected_codebooks',alternatives))
    graphs={}
    def select(name):
        for module,key,values in swaps:setattr(module,key,values[name])
        if name in graphs:decoder.graph,decoder.audio=graphs[name]
    for name in alternatives:
        select(name);decoder.warmup();graphs[name]=(decoder.graph,decoder.audio)
    codes=torch.load(RESULTS/'fixture.pt',weights_only=True)['codes'].cuda()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    waves={}
    with torch.cuda.stream(stream):
        for name in graphs:
            select(name);decoder.reset()
            waves[name]=torch.cat([decoder.decode(codes[...,i:i+1]) for i in range(codes.shape[-1])],-1)
    torch.cuda.current_stream().wait_stream(stream)
    assert all(torch.equal(v,waves['torch']) for v in waves.values())
    rows=[]
    for repeat in range(rounds):
        order=list(graphs);order=order[repeat%3:]+order[:repeat%3]
        if repeat%2:order.reverse()
        records={}
        for name in order:
            select(name);decoder.reset();torch.cuda.synchronize()
            begin=time.perf_counter();first=decoder.decode(codes[...,:1]).float().flatten().cpu()
            first_ms=(time.perf_counter()-begin)*1000
            assert torch.equal(first,waves['torch'][...,:1920].float().flatten().cpu())
            # Graph GPU timing retains consecutive causal frames and resets for
            # each group; it excludes reset and CPU output-copy time.
            gpu=[]
            for _ in range(3):
                decoder.reset();decoder.codes.copy_(codes[...,:1])
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):decoder.graph.replay()
                end.record();end.synchronize();gpu.append(start.elapsed_time(end)/20)
            records[name]={'first_pcm_ms':first_ms,'graph_ms':statistics.median(gpu)}
        rows.append({'round':repeat,'order':order,'records':records});print('CODEC',rows[-1],flush=True)
    result={'rows':rows,'allocations':allocations,'private_stream_pcm_exact':True,'frames_checked':codes.shape[-1],
            'variants':{name:{'first_pcm':stats([r['records'][name]['first_pcm_ms'] for r in rows]),
                              'graph':stats([r['records'][name]['graph_ms'] for r in rows])} for name in graphs},
            'method':'Shared FP32 codec and stream state, separate graphs for unchanged Torch allocations, uncompressed VMM and compressed VMM decoder parameters/output projection/codebook table. First-frame CPU PCM copy included; no LLM traffic. All 32 codebooks. Rounded VMM sizes reported separately.'}
    select('torch');decoder.close()
    return result


@torch.inference_mode()
def heads_benchmark(rounds):
    path=snapshot_download('OpenMOSS-Team/MOSS-TTS-v1.5',revision=TTS_REVISION,local_files_only=True)
    from pathlib import Path
    mapping=json.loads((Path(path)/'model.safetensors.index.json').read_text())['weight_map']
    parts=[]
    for i in range(1,33):
        key=f'lm_heads.{i}.weight'
        with safe_open(str(Path(path)/mapping[key]),framework='pt',device='cuda') as handle:
            parts.append(handle.get_tensor(key)[:1024])
    w=torch.cat(parts).contiguous();del parts
    variants={'torch':w};allocations=[]
    for name in ('vmm_plain','compressed'):
        variants[name],meta=clone(w,compressed=name=='compressed');assert torch.equal(variants[name],w)
        allocations.append({'variant':name,**meta})
    torch.manual_seed(92)
    inputs=[torch.randn(1,4096,device='cuda',dtype=torch.bfloat16) for _ in range(12)]
    checks=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for count in (8,16,24,32):
            for index,x in enumerate(inputs):
                expected=F.linear(x,w[:count*1024])
                for name in ('vmm_plain','compressed'):
                    actual=F.linear(x,variants[name][:count*1024]);assert torch.equal(actual,expected)
                    checks.append({'heads':count,'input':index,'variant':name,'exact':True})
    torch.cuda.current_stream().wait_stream(stream)
    rows=[]
    # This stage has one persistent 256 MiB head matrix. Prefix 8 fits differently
    # into cache; actual complete-request claims require a later pipeline test.
    for count in (8,16,24,32):
        for repeat in range(rounds):
            order=list(variants);order=order[repeat%3:]+order[:repeat%3]
            if repeat%2:order.reverse()
            timing={name:measure(lambda x:F.linear(x,variants[name][:count*1024]),inputs) for name in order}
            rows.append({'heads':count,'round':repeat,'order':order,'us':timing})
            print('HEADS',rows[-1],flush=True)
    return {'rows':rows,'checks':checks,'allocations':allocations,
            'median_us':{count:{name:statistics.median(r['us'][name] for r in rows if r['heads']==count) for name in variants} for count in (8,16,24,32)},
            'method':'Same actual checkpoint BF16 head matrix; twelve distinct synthetic hidden rows; existing exact head-prefix shapes. Identical cuBLAS operations, rotating/reversed order, private-stream checks. Stage screening only, excluding LLM traffic.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('heads','codec'));p.add_argument('--tag',required=True);p.add_argument('--rounds',type=int,default=8);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and at least two rounds required')
    path=RESULTS/f'compressed_{a.mode}_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4)
    result=(codec_benchmark if a.mode=='codec' else heads_benchmark)(a.rounds)
    torch.cuda.synchronize();gc.collect();result.update({'codebooks':32,'torch':torch.__version__,'allocator_after':library().counters()})
    assert result['allocator_after']['live_allocations']==0,result['allocator_after']
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASS',result['allocator_after'],flush=True)


if __name__=='__main__':main()
