"""Reassemble the exact Triton 3.8 gate/up PTX with CUDA 13 resource controls.

Only PTX version, static shared declaration, register cap and optional shared
spill pragma change. Arithmetic, dependency instructions and weights stay fixed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
import torch
from .common import RESULTS
from .gateup_cluster_binary import Binary
from .ptx_resources import ReassembledKernel


def configs():
    choices = {}
    for source, label in (('c1_ig1ir2', 'ir2'), ('c1', 'ir4')):
        choices[label+'_original'] = {'source': source, 'original': True}
        for cap, static, spill in ((None, False, False), (None, True, False),
                (144, False, False), (128, False, False), (112, False, False), (96, False, False),
                (128, True, False), (112, True, False), (96, True, False),
                (128, True, True), (112, True, True), (96, True, True)):
            name = f'{label}_r{cap or 0}_' + ('spill' if spill else 'static' if static else 'dynamic')
            choices[name] = {'source': source, 'original': False,
                'registers': cap, 'static_shared': static, 'shared_spilling': spill}
    return choices


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True)
    p.add_argument('--configs', nargs='+', choices=tuple(configs()))
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe tag required')
    torch.set_num_threads(4); torch.empty(0, device='cuda')
    source = RESULTS/'gateup_cluster_bundle_screen_v1'
    original = json.loads((source/'manifest.json').read_text())
    assert original['complete']
    folder = RESULTS/f'gateup_resource_bundle_{args.tag}'; folder.mkdir(exist_ok=False)
    for name in ('gateup_resources.py', 'ptx_resources.py', 'gateup_cluster.py', 'gateup_cluster_binary.py', 'qkv_cluster_binary.py'):
        shutil.copyfile(Path(__file__).with_name(name), folder/name)
    choices = configs()
    if args.configs:
        choices = {n: choices[n] for n in args.configs}
    manifest = {'format': 'gateup_cluster_cubin_v1', 'complete': False,
        'torch': torch.__version__, 'triton': original['triton'], 'source_bundle': str(source),
        'compiler': subprocess.check_output(['/workspace/cuda-13.0-ptxas/ptxas', '--version'], text=True),
        'configs': {}, 'build_options': choices, 'binaries': {},
        'scope': 'Original explicit arithmetic PTX; CUDA 13.0 register caps and static/shared-spill alternatives. '
                 'TTGIR is pre-reassembly IR. Spills is null for reassembled binaries: use compiler spill bytes '
                 'and driver local/shared allocation fields, which are distinct from Triton spill counts.'}
    def save():
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    for name, choice in choices.items():
        source_name = choice['source']; manifest['configs'][name] = original['configs'][source_name]
        for add in (False, True):
            for debug in (False, True):
                original_prefix = f'{source_name}_add{int(add)}_debug{int(debug)}'
                record = copy.deepcopy(original['binaries'][original_prefix])
                prefix = f'{name}_add{int(add)}_debug{int(debug)}'; record['prefix'] = prefix
                if choice['original']:
                    for suffix in ('ptx', 'ttgir', 'cubin', 'sass'):
                        shutil.copyfile(source/(original_prefix+'.'+suffix), folder/(prefix+'.'+suffix))
                    record['original_cubin'] = True
                else:
                    binary = Binary(source, original['binaries'][original_prefix])
                    compiled = SimpleNamespace(metadata=binary.metadata, asm=binary.asm,
                        src=SimpleNamespace(signature=dict(record['pointer_signature'])))
                    kernel = ReassembledKernel(compiled, **{k:choice[k] for k in
                        ('registers', 'static_shared', 'shared_spilling')})
                    resources = kernel.resources; build = RESULTS/'ptx_resource_build'/resources['build']
                    for suffix in ('ptx', 'cubin', 'log', 'original.ptx'):
                        shutil.copyfile(Path(str(build)+'.'+suffix), folder/(prefix+'.'+suffix))
                    shutil.copyfile(source/(original_prefix+'.ttgir'), folder/(prefix+'.ttgir'))
                    sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(folder/(prefix+'.cubin'))], text=True)
                    (folder/(prefix+'.sass')).write_text(sass)
                    log = (folder/(prefix+'.log')).read_text()
                    spill = re.search(r'Function properties for _kernel\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
                    assert spill, log
                    record['resources'] = {**resources, 'compiler_stack_bytes': int(spill[1]),
                        'compiler_spill_store_bytes': int(spill[2]), 'compiler_spill_load_bytes': int(spill[3])}
                    record['registers'] = resources['registers']; record['spills'] = None
                    record['metadata']['shared'] = resources['dynamic_shared_bytes']
                    record['cubin_sha256'] = resources['cubin_sha256']
                assert hashlib.sha256((folder/(prefix+'.cubin')).read_bytes()).hexdigest() == record['cubin_sha256']
                manifest['binaries'][prefix] = record
                save()
                print('EXPORTED', prefix, record['registers'], record.get('resources', {}), flush=True)
    torch.cuda.synchronize(); manifest['complete'] = True; save()
    print('SAVED', folder, flush=True)


if __name__ == '__main__':
    main()
