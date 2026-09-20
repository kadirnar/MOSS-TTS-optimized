"""Save compiled evidence for prefill fusion and QKV prefetch addressing."""
import argparse
import hashlib
import json
import subprocess

import torch

from .common import RESULTS
from .benchmark_bulk_address import Chain
from .benchmark_projection_pdl import load_layer
from .prefill_qkv import _prefill_qkv


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    folder=RESULTS/f'prefill_qkv_audit_{args.tag}';folder.mkdir(exist_ok=False)
    entry=load_layer(17);kernels={}
    for name,options in (('address_control',{}),('address_candidate',{'qkv':{'address_mode':1}})):
        _,compiled=Chain(options)(entry,audit=True);kernels[name]=compiled['qkv']
    x=torch.randn((1,160,6144),device='cuda',dtype=torch.bfloat16)
    weight=torch.ones(128,device='cuda',dtype=torch.bfloat16)
    cos=torch.ones((1,160,128),device='cuda',dtype=torch.bfloat16);sin=torch.zeros_like(cos)
    q=torch.empty((1,160,32,128),device='cuda',dtype=torch.bfloat16)
    keys=torch.empty((1,8,1024,128),device='cuda',dtype=torch.bfloat16);values=torch.empty_like(keys)
    pos=torch.arange(160,device='cuda')
    kernels['prefill']=_prefill_qkv[(160,40)](x,weight,weight,cos,sin,q,keys,values,pos,1024,1e-6,num_warps=4)
    torch.cuda.synchronize();result={}
    for name,kernel in kernels.items():
        prefix=folder/name
        for suffix in ('ptx','ttgir','cubin'):
            data=kernel.asm[suffix]
            prefix.with_suffix('.'+suffix).write_bytes(data if isinstance(data,bytes) else data.encode())
        sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(prefix.with_suffix('.cubin'))],text=True)
        prefix.with_suffix('.sass').write_text(sass)
        ptx=kernel.asm['ptx'];lines=ptx.splitlines()
        hints=[i for i,line in enumerate(lines) if 'cp.async.bulk.prefetch' in line]
        waits=[i for i,line in enumerate(lines) if 'griddepcontrol.wait' in line]
        if hints:assert waits and max(hints)<min(waits)
        result[name]={'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared,
            'cubin_sha256':hashlib.sha256(kernel.asm['cubin']).hexdigest(),
            'runtime_modulo_in_ptx':'rem.' in ptx,'prefetch_ptx_lines_zero_based':hints,
            'wait_ptx_lines_zero_based':waits,'all_hints_precede_wait':bool(hints) and max(hints)<min(waits),
            'prefetch_sass_lines':[line.strip() for line in sass.splitlines() if 'UBLKPF' in line],
            'pdl_wait_sass_lines':[line.strip() for line in sass.splitlines() if 'ACQBULK' in line]}
    assert result['address_control']['runtime_modulo_in_ptx']
    assert not result['address_candidate']['runtime_modulo_in_ptx']
    assert not any(row['spills'] for row in result.values())
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',folder,flush=True)


if __name__=='__main__':main()
