"""Retain compiled async-weight kernels and actual SM90 instruction listings."""
import argparse
import hashlib
import json
import subprocess

import torch
from .common import RESULTS
from .benchmark_projection_pdl import load_layer
from .benchmark_async_weights import Chain
from .benchmark_group128 import quantize
from .dp4a_packing import pack_interleaved
from .dp4a_async_weights import configured
from .dp4a_layout_pdl_prefetch import linear as control
from .short_scales import PLAN


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--out-rows',type=int,choices=(8,16),default=16);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    folder=RESULTS/f'async_weights_audit_{a.tag}';folder.mkdir(exist_ok=False)
    torch.set_num_threads(4);e=load_layer(17);kernels={}
    for name,options in {'control':None,'copy_sw8':{'mode':1,'swizzle':8},'tma':{'mode':2,'swizzle':0},
                         'tma_compact':{'mode':3,'swizzle':0},'tma_partial':{'mode':4,'swizzle':0},
                         'tma_partial_r8w4':{'mode':4,'swizzle':0,'tile':{'rows':8,'warps':4}}}.items():
        _,compiled=Chain(options)(e,audit=True);kernels['down_'+name]=compiled['down']
    saved=torch.load(RESULTS/'gptq_v1_g32_d10/17_out.pt',map_location='cuda',weights_only=True)
    w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16()
    x=torch.load(RESULTS/'calibration_v1/17_out.pt',weights_only=True)[231:232].cuda();q=quantize(x,32)
    for name,options in {'control':None,'copy_sw8_r16':{'mode':1,'swizzle':8},'tma_r16':{'mode':2,'swizzle':0},
                         'copy_cg_r16':{'mode':1,'swizzle':1,'cache':1}}.items():
        fn=control if options is None else configured(**options)
        _,k=fn(x,w,s,prequantized=q,**{**PLAN['out'],**({} if options is None else {'rows':a.out_rows})},
               trigger_mode=3,prefetch=1,return_kernel=True)
        kernels['out_'+name.replace('r16',f'r{a.out_rows}')]=k
    for rows in (8,16):
        _,k=control(x,w,s,prequantized=q,**{**PLAN['out'],'rows':rows},trigger_mode=3,prefetch=3,return_kernel=True)
        kernels[f'out_register_r{rows}']=k
    torch.cuda.synchronize();result={}
    for name,k in kernels.items():
        prefix=folder/name
        for suffix in ('ptx','ttgir','cubin'):
            value=k.asm[suffix];prefix.with_suffix('.'+suffix).write_bytes(value if isinstance(value,bytes) else value.encode())
        sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(prefix.with_suffix('.cubin'))],text=True)
        prefix.with_suffix('.sass').write_text(sass)
        result[name]={'registers':k.n_regs,'spills':k.n_spills,'shared_bytes':k.metadata.shared,
            'cubin_sha256':hashlib.sha256(k.asm['cubin']).hexdigest(),
            'pdl_waits_in_ptx':k.asm['ptx'].count('griddepcontrol.wait'),
            'pdl_triggers_in_ptx':k.asm['ptx'].count('griddepcontrol.launch_dependents'),
            'async_copy_lines':[line.strip() for line in k.asm['ptx'].splitlines() if any(op in line for op in ('cp.async','mbarrier.'))],
            'sass_copy_sync_lines':[line.strip() for line in sass.splitlines() if any(op in line for op in ('LDGSTS','UTMALDG','UBLK','ACQBULK','DEPBAR','SYNCS','MEMBAR'))]}
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('SAVED',folder,len(result),flush=True)


if __name__=='__main__':main()
