"""Five-kernel attention chains with exact projections and programmatic edges."""
import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from .common import RESULTS
from .compiler_audit import cuda
from .benchmark_norm_projection import flatten
from .benchmark_projection_pdl import graph_edges
from .dp4a_packing import pack_interleaved
from .dp4a_norm_pdl import linear as norm
from .dp4a_norm_projection import SELECTED
from .dp4a_layout_pdl_prefetch import linear as project
from .short_scales import PLAN
from .kernels import _qk_rope_cache
from .attention_native import launch as native
from .attention_quant import reduce_quant as reduce_reference
from .qk_rope_pdl import launch as rope
from .attention_pdl import launch as attention, library
from .attention_quant_pdl import reduce_quant as reduce_pdl
from .tune_weight_reads import measure


def load_layer(layer, capacity=256):
    raw = torch.load(RESULTS/f'norm_projection_capture_v1/{layer:02d}_qkv.pt', map_location='cuda', weights_only=True)
    weights = {}
    for name in ('qkv', 'out'):
        d = torch.load(RESULTS/f'gptq_v1_g32_d10/{layer:02d}_{name}.pt', map_location='cuda', weights_only=True)
        weights[name] = pack_interleaved(d['packed']), d['scales'].bfloat16()
    folder = RESULTS/'compiler_capture_v1'
    manifest = json.loads((folder/'manifest.json').read_text())
    nearest = min((0,17,35), key=lambda i:abs(i-layer))
    record = next(r for r in manifest['records'] if r['kind']=='attention' and r['label']==f'step10_layer{nearest}')
    d = cuda(torch.load(folder/record['file'], weights_only=True))
    checkpoint = Path('/workspace/models/moss-tts-v15')
    mapping = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    for key, name in (('qw','q_norm'),('kw','k_norm')):
        key_name = f'language_model.layers.{layer}.self_attn.{name}.weight'
        with safe_open(checkpoint/mapping[key_name], framework='pt', device='cpu') as f:
            d[key] = f.get_tensor(key_name).bfloat16().cuda()
    return {'layer':layer, 'raw':raw, 'weights':weights, 'attention':d, 'capacity':capacity,
            'cache_source':record['label']}


def chain(entry, config=None, *, index=0, debug=False):
    d = entry['attention']; raw = entry['raw']; weights = entry['weights']
    residual = raw['residual'][index:index+1] if raw['has_residual'] else None
    summed, qkv = norm(raw['x'][index:index+1], residual, raw['weight'], raw['eps'],
                       *weights['qkv'], **SELECTED['qkv'], trigger_mode=1)
    q = torch.empty((32,128), device=qkv.device, dtype=qkv.dtype)
    splits = entry['capacity']//32
    part = torch.empty((32,splits,128), device=q.device, dtype=torch.float32)
    lse = torch.empty((32,splits), device=q.device, dtype=torch.float32)
    values = qkv,d['qw'],d['kw'],d['cos'],d['sin'],q,d['k'],d['v'],d['position']
    if config is None or config['qk'] is None:
        _qk_rope_cache[(40,)](*values,d['k'].shape[-2],d['eps'],enable_fp_fusion=False)
    else:
        rope(*values,d['eps'],pdl=config['pdl'],trigger=config['qk'],preload=config.get('preload',False))
    if config is None or config['attention'] is None:
        native(q,d['k'],d['v'],d['position'],part,lse)
    else:
        attention(q,d['k'],d['v'],d['position'],part,lse,pdl=config['pdl'],trigger=config['attention'])
    if config is None or config['reduce'] is None:
        hidden, quantized = reduce_reference(part,lse)
    else:
        hidden, quantized = reduce_pdl(part,lse,pdl=config['pdl'],trigger=config['reduce'])
    output = project(hidden,*weights['out'],prequantized=quantized,**PLAN['out'],trigger_mode=3,prefetch=1)
    if debug:return summed,qkv,q,part,lse,hidden,quantized,output
    return output


def configurations():
    configs = {'control':None, 'clone_no_pdl':{'pdl':False,'qk':0,'attention':0,'reduce':0}}
    for at in (0,1,2,3):
        for rt in (0,1,2,3):
            configs[f'q1_a{at}_r{rt}'] = {'pdl':True,'qk':1,'attention':at,'reduce':rt}
    for qt in (0,1,2,3):
        configs[f'pre_q{qt}_a1_r1'] = {'pdl':True,'qk':qt,'attention':1,'reduce':1,'preload':True}
    for key in ('qk','attention','reduce'):
        c = {'pdl':True,'qk':None,'attention':None,'reduce':None}; c[key]=1
        configs[f'only_{key}']=c
    return configs


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--layers',type=int,choices=(1,36),default=1);p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--capacity',type=int,choices=(128,256,512,1024),default=256)
    p.add_argument('--configs',nargs='+');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'attention_pdl_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    assert torch.cuda.get_device_capability()==(9,0)
    torch.set_num_threads(4);library()
    configs=configurations()
    if a.configs:configs={name:configs[name] for name in a.configs}
    ring=[load_layer(i,a.capacity) for i in (range(36) if a.layers==36 else (17,))]
    if a.capacity==128:
        for e in ring:e['attention']['position'].fill_(127)
    checks=[];edges={};rows=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for entry in ring:
            for index in (0,10,31):
                expected=chain(entry,index=index,debug=True)
                for name,config in configs.items():
                    if config is None:continue
                    actual=chain(entry,config,index=index,debug=True)
                    counts=[int((a!=b).sum()) for a,b in zip(flatten(actual),flatten(expected),strict=True)]
                    checks.append({'layer':entry['layer'],'input':index,'config':name,'mismatches':counts})
            print('CHECKED',entry['layer'],flush=True)
        for name,config in configs.items():
            entry=ring[0]
            for _ in range(2):chain(entry,config)
            graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph,stream=stream):actual=chain(entry,config)
            edges[name]=graph_edges(graph);graph.replay()
            expected=chain(entry)
            assert torch.equal(actual,expected),name
            expected_edges=1+sum(config is not None and config['pdl'] and config[k] is not None for k in ('qk','attention','reduce'))
            assert len(edges[name])==4 and sum(e['type']==1 for e in edges[name])==expected_edges,(name,edges[name])
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'torch':torch.__version__,'layers':a.layers,'capacity':a.capacity,'configs':configs,
            'checks':checks,'graph_edges':edges,'rows':rows,
            'method':'Norm/QKV PDL -> QK/RoPE -> native split attention -> reduction/G32 quantization -> output PDL with scale preload. '
                     'Actual calibrated weights and normalization inputs from every tested layer; actual Q/K norm weights. '
                     'Frozen representative KV/rotary fixtures from nearest captured layer 0/17/35; these isolated chains do not form a full model trajectory. '
                     'Three changed inputs per layer, private-stream eager and graph equality checks, CUDA edge inspection. '
                     'Weights rotate over all tested layers; timings are chain microseconds, not TTFA.'}
    path.write_text(json.dumps(result,indent=2)+'\n')
    for repeat in range(a.rounds):
        order=list(configs);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        timings={name:measure(lambda e:chain(e,configs[name]),ring) for name in order}
        rows.append({'round':repeat,'order':order,'us':timings});print('ROUND',repeat,timings,flush=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
    result['summary']={name:{'median_us':statistics.median(r['us'][name] for r in rows),
                            'mismatches':sum(sum(c['mismatches']) for c in checks if c['config']==name)} for name in configs}
    path.write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
