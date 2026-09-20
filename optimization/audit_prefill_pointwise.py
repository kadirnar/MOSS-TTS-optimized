"""Compiled evidence for pointwise fusions and rejected split-PDL schedules."""
import argparse
import hashlib
import json
import subprocess

import torch

from .common import RESULTS
from .prefill_pointwise import silu_mul
from .kernels import _add_rmsnorm
from .benchmark_split_norm_pdl import Chain
from .benchmark_projection_pdl import load_layer


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    folder=RESULTS/f'prefill_pointwise_audit_{args.tag}';folder.mkdir(exist_ok=False)
    x=torch.randn((1,160,24576),device='cuda',dtype=torch.bfloat16);_,activation=silu_mul(x,return_kernel=True)
    a=torch.randn((1,160,4096),device='cuda',dtype=torch.bfloat16);res=torch.randn_like(a);weight=torch.ones(4096,device='cuda',dtype=torch.bfloat16)
    summed=torch.empty_like(a);out=torch.empty_like(a)
    norm=_add_rmsnorm[(160,)](a,res,weight,summed,out,4096,1e-6,4096)
    kernels={'prefill_activation':activation,'prefill_residual_norm':norm};entry=load_layer(17)
    for name,config in (('control',{}),('split',{'up':{}}),('split_cap168',{'up':{'registers':168}}),('split_cap128',{'up':{'registers':128}})):
        _,compiled=Chain(config)(entry,audit=True)
        kernels[name+'_up']=compiled['up']
        if 'up_normalizer' in compiled:kernels[name+'_normalizer']=compiled['up_normalizer']
    torch.cuda.synchronize();result={}
    for name,kernel in kernels.items():
        prefix=folder/name
        for suffix in ('ptx','ttgir','cubin'):
            data=kernel.asm[suffix];prefix.with_suffix('.'+suffix).write_bytes(data if isinstance(data,bytes) else data.encode())
        sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(prefix.with_suffix('.cubin'))],text=True)
        prefix.with_suffix('.sass').write_text(sass)
        result[name]={'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared,
            'cubin_sha256':hashlib.sha256(kernel.asm['cubin']).hexdigest(),
            'pdl_waits_in_ptx':kernel.asm['ptx'].count('griddepcontrol.wait'),
            'pdl_launches_in_ptx':kernel.asm['ptx'].count('griddepcontrol.launch_dependents'),
            'bulk_hints_in_ptx':kernel.asm['ptx'].count('cp.async.bulk.prefetch')}
    assert result['prefill_activation']['spills']==result['prefill_residual_norm']['spills']==0
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('Saved',folder,flush=True)


if __name__=='__main__':main()
