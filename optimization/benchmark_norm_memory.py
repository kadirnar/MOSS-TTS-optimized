"""Audit H200 load policies, prefetch, CTA order and register limits."""
import argparse
import json
import re

import torch

from .common import RESULTS
from .compiler_audit import cuda
from .dp4a_packing import pack_interleaved
from .dp4a_norm_projection import linear as reference,SELECTED
from .dp4a_norm_memory import linear
from .benchmark_norm_projection import flatten
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--timing-layer',type=int,choices=(0,17,35),default=0);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'norm_memory_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4)
    folder=RESULTS/'compiler_capture_v1';manifest=json.loads((folder/'manifest.json').read_text())
    result={'method':'One selected layer/input with eight distinct packed weight/scale copies for timing; median nine CUDA graph timings, 24 calls per replay. Validation covers nine real frozen layer/step inputs per projection with private-stream comparisons against selected G32 norm fusion. Fixed four-warps and arithmetic layouts. This timing ring reuses normalization buffers; validate_norm_memory layer_ring tests all 36 layers and varies those buffers too.','cases':{}}
    for name in ('qkv','up'):
        fused=name=='up';base=SELECTED[name];inputs=[];weights={}
        for record in manifest['records']:
            if record['kind']!='norm_quant':continue
            index=int(re.search(r'norm(\d+)',record['label'])[1])
            if bool(index%2)!=fused:continue
            d=cuda(torch.load(folder/record['file'],weights_only=True));inputs.append((record['label'],index//2,d))
            if index//2 not in weights:
                saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{index//2:02d}_{name}.pt',map_location='cuda',weights_only=True)
                weights[index//2]=(pack_interleaved(saved['packed']),saved['scales'].bfloat16())
        def old(d,pair,debug=False):return reference(d['x'],d['residual'],d['weight'],d['eps'],*pair,fused=fused,debug=debug,**base)
        def new(d,pair,c,debug=False,return_kernel=False):return linear(d['x'],d['residual'],d['weight'],d['eps'],*pair,fused=fused,debug=debug,return_kernel=return_kernel,**base,**c)
        timing=next(row for row in inputs if row[1]==a.timing_layer)
        _,layer,d=timing;w,s=weights[layer];ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
        configs=[{'weight_cache':wc,'eviction':ep,'scale_cache':sc}
                 for wc,ep in [('', ''),('.cg',''),('.ca',''),('','evict_first'),('','evict_last'),('.cg','evict_first'),('.ca','evict_last')]
                 for sc in ('','.cg')]
        configs += [{'weight_cache':wc,'prefetch':pf} for pf in (128,256,512) for wc in ('','.cg')]
        configs += [{'weight_cache':wc,'swizzle':sw} for sw in (2,4,8,16,32) for wc in ('','.cg')]
        configs += [{'weight_cache':wc,'max_registers':mr} for mr in ((160,128,96,64) if fused else (56,48,40,32)) for wc in ('','.cg')]
        case={'reference_us':measure(lambda pair:old(d,pair),ring),'base_config':base,'timing_input':timing[0],'timing_layer':a.timing_layer,'candidates':[]};result['cases'][name]=case
        for ci,c in enumerate(configs):
            row={'config':c,'checks':0,'exact_checks':0,'mismatches':0}
            try:
                for _,layer,d in inputs:
                    pair=weights[layer];expected=old(d,pair,True)
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):actual=new(d,pair,c,True)
                    torch.cuda.current_stream().wait_stream(stream)
                    counts=[int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]
                    row['checks']+=1;row['exact_checks']+=not any(counts);row['mismatches']+=sum(counts)
                d=timing[2]
                row['us']=measure(lambda pair:new(d,pair,c),ring)
                _,compiled=new(d,ring[0],c,return_kernel=True)
                row['resources']={'registers':compiled.n_regs,'spills':compiled.n_spills,'shared_bytes':compiled.metadata.shared}
                for ext in ('ptx','ttgir'):(RESULTS/f'norm_memory_{name}_{a.tag}_{ci}.{ext}').write_text(compiled.asm[ext])
            except Exception as e:row['error']=repr(e)
            case['candidates'].append(row)
            print(name,row,flush=True);path.write_text(json.dumps(result,indent=2)+'\n')
        valid=[r for r in case['candidates'] if r['exact_checks']==len(inputs) and 'us' in r]
        case['best_exact']=min(valid,key=lambda r:r['us']) if valid else None
        print('DONE',name,'REF',case['reference_us'],'BEST',case['best_exact'],flush=True)
        path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
