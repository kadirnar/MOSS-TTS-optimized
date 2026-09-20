"""Retain PTX/cubin/SASS for selected bulk hints and the unmodified control."""
import argparse
import hashlib
import json
import subprocess

import torch

from .common import RESULTS
from .benchmark_bulk_prefetch import Chain
from .benchmark_projection_pdl import load_layer


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    folder=RESULTS/f'bulk_prefetch_audit_{a.tag}';folder.mkdir(exist_ok=False)
    entry=load_layer(17);result={}
    for name,options in [('control',{}),('prefix',{s:{'divisor':16} for s in ('up','down','qkv')}),('full',{s:{'divisor':1} for s in ('up','down','qkv')})]:
        _,kernels=Chain(options)(entry,audit=True);result[name]={}
        for stage,k in kernels.items():
            prefix=folder/f'{name}_{stage}'
            for suffix in ('ptx','ttgir','cubin'):
                data=k.asm[suffix];file=prefix.with_suffix('.'+suffix)
                file.write_bytes(data if isinstance(data,bytes) else data.encode())
            sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(prefix.with_suffix('.cubin'))],text=True)
            prefix.with_suffix('.sass').write_text(sass)
            ptx=k.asm['ptx'];hints=[i for i,line in enumerate(ptx.splitlines()) if 'cp.async.bulk.prefetch' in line]
            waits=[i for i,line in enumerate(ptx.splitlines()) if 'griddepcontrol.wait' in line]
            if hints:assert waits and max(hints)<min(waits)
            result[name][stage]={'registers':k.n_regs,'spills':k.n_spills,'shared_bytes':k.metadata.shared,
                'prefetch_ptx_line_numbers_zero_based':hints,'wait_ptx_line_numbers_zero_based':waits,
                'all_hints_precede_wait':bool(hints) and max(hints)<min(waits),
                'cubin_sha256':hashlib.sha256(k.asm['cubin']).hexdigest(),
                'prefetch_sass_lines':[line.strip() for line in sass.splitlines() if 'UBLKPF' in line or 'UTMAPF' in line or 'CCTL' in line],
                'pdl_wait_sass_lines':[line.strip() for line in sass.splitlines() if 'ACQBULK' in line],
                'runtime_modulo_in_ptx':'rem.' in ptx}
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('Saved',folder,flush=True)


if __name__=='__main__':main()
