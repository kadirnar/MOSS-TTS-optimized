"""Archive compiled G64 kernels, including rejected register-heavy layouts."""
import argparse
import hashlib
import json
import subprocess

import torch
from .common import RESULTS
from .benchmark_group64_a64 import Chain,load


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    folder=RESULTS/f'group64_audit_{a.tag}';folder.mkdir(exist_ok=False)
    torch.set_num_threads(4);e=load(17);kernels={}
    for name,stages in (
        ('control',{}),('g64_nominal',{'up':{},'down':{},'qkv':{}}),
        ('g64_factored',{'up':{'factor':1},'qkv':{'factor':1}}),
        ('a64',{'up':{'activation_group':64},'qkv':{'activation_group':64}}),
        ('a64_spilled',{'up':{'activation_group':64,'rows':64},'down':{'activation_group':64},'qkv':{'activation_group':64}})):
        _,compiled=Chain(stages)(e,audit=True)
        for stage,k in compiled.items():kernels[name+'_'+stage]=k
    torch.cuda.synchronize();result={}
    for name,k in kernels.items():
        prefix=folder/name
        for suffix in ('ptx','ttgir','cubin'):
            value=k.asm[suffix];prefix.with_suffix('.'+suffix).write_bytes(value if isinstance(value,bytes) else value.encode())
        prefix.with_suffix('.sass').write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(prefix.with_suffix('.cubin'))],text=True))
        result[name]={'registers':k.n_regs,'spills':k.n_spills,'shared_bytes':k.metadata.shared,
            'cubin_sha256':hashlib.sha256(k.asm['cubin']).hexdigest(),
            'pdl_waits_in_ptx':k.asm['ptx'].count('griddepcontrol.wait'),
            'pdl_triggers_in_ptx':k.asm['ptx'].count('griddepcontrol.launch_dependents'),
            'bulk_hints_in_ptx':k.asm['ptx'].count('cp.async.bulk.prefetch')}
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('SAVED',folder,flush=True)


if __name__=='__main__':main()
