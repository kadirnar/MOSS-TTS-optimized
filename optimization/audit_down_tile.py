"""Archive emitted code and resources for the old and selected down tiles."""
import argparse
import hashlib
import json
import subprocess

import torch
import triton
from .common import RESULTS
from .benchmark_down_preload import Chain, load_layer, exact
from .qkv_cluster_binary import load_bundle


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True)
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe tag required')
    folder = RESULTS/f'down_tile_audit_{args.tag}'
    folder.mkdir(exist_ok=False)
    choices, launchers = load_bundle(RESULTS/'qkv_cluster_bundle_v6')
    entry = load_layer(17)
    result = {'codebooks': 32, 'torch': torch.__version__, 'triton': triton.__version__, 'kernels': {}}
    expected = None
    for name, tile in (('control', None), ('down8', {'rows': 8, 'warps': 4, 'prefetch': 1})):
        chain = Chain(tile, choices['c8_t2_exact'], launchers['c8_t2_exact'])
        actual, kernel = chain(entry, audit=True)
        if expected is None:
            expected = actual
        else:
            assert exact(actual, expected)
        for key in ('ptx', 'ttgir', 'cubin'):
            value = kernel.asm[key]; path = folder/f'{name}.{key}'
            path.write_bytes(value) if isinstance(value, bytes) else path.write_text(value)
        disassembly = subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(folder/f'{name}.cubin')],
                                     check=True, text=True, capture_output=True).stdout
        (folder/f'{name}.sass').write_text(disassembly)
        ptx = kernel.asm['ptx']; before, separator, after = ptx.partition('griddepcontrol.wait')
        assert separator and 'griddepcontrol.launch_dependents' in after
        result['kernels'][name] = {'registers': kernel.n_regs, 'spills': kernel.n_spills,
            'shared_bytes': kernel.metadata.shared, 'tile': chain.tile,
            'cubin_sha256': hashlib.sha256(kernel.asm['cubin']).hexdigest(),
            'static_global_loads_before_wait': before.count('ld.global'),
            'static_global_loads_after_wait': after.count('ld.global'),
            'sass_wait_present': 'ACQBULK' in disassembly,
            'note': 'Static load-instruction counts, not bytes transferred or achieved occupancy.'}
    (folder/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
