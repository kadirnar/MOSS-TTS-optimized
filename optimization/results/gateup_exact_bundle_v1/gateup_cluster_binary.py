"""Export gate/up cluster cubins with Triton 3.8; run in selected host runtime."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess
import torch
import triton
from .common import RESULTS
from .ptx_resources import Attribute, LaunchConfig, check
from .qkv_cluster_binary import Binary as BaseBinary
from .gateup_cluster import linear, configs


class Binary(BaseBinary):
    def __call__(self, arguments, grid, eps):
        if eps != self.record['eps'] or grid != 384:
            raise ValueError('Binary specialization mismatch')
        dtypes = {'*bf16': torch.bfloat16, '*fp32': torch.float32, '*u8': torch.uint8, '*i8': torch.int8}
        tensors = []
        for name, kind in self.record['pointer_signature']:
            t = arguments[name]
            if t is None or not t.is_cuda or not t.is_contiguous() or t.dtype != dtypes[kind]:
                raise ValueError('Binary tensor mismatch: '+name)
            tensors.append(t)
        if any(t.device.index != self.device for t in tensors):
            raise ValueError('Binary and buffers must use the same CUDA device')
        values = [C.c_uint64(t.data_ptr()) for t in tensors] + [C.c_uint64(0), C.c_uint64(0)]
        pointers = (C.c_void_p*len(values))(*[C.cast(C.pointer(v), C.c_void_p) for v in values])
        attributes = (Attribute*3)()
        attributes[0].id = 6; attributes[0].value.serialization = 1
        dims = tuple(self.metadata.cluster_dims)
        assert dims == (self.metadata.num_ctas, 1, 1)
        attributes[1].id = 4
        for i, value in enumerate(dims):
            C.cast(C.byref(attributes[1].value), C.POINTER(C.c_uint))[i] = value
        attributes[2].id = 5; attributes[2].value.serialization = 1
        cfg = LaunchConfig(grid*dims[0], 1, 1, 128, 1, 1, self.metadata.shared,
            torch.cuda.current_stream(tensors[0].device).cuda_stream, attributes,
            3 if self.metadata.num_ctas > 1 else 1)
        check(self.lib.cuLaunchKernelEx(C.byref(cfg), self.function, pointers, None), 'gate/up cluster launch')


def load_bundle(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['format'] != 'gateup_cluster_cubin_v1' or not manifest['complete']:
        raise ValueError('Complete gate/up bundle required')
    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError('SM90 bundle only')
    torch.empty(0, device='cuda')
    kernels = {n: Binary(folder, r) for n, r in manifest['binaries'].items()}
    launchers = {}
    for name, options in manifest['configs'].items():
        def make(name, options):
            def fn(*args, **kwargs):
                if any(kwargs.get(k) != v for k, v in options.items()):
                    raise ValueError('Binary configuration differs')
                key = f'{name}_add{int(args[1] is not None)}_debug{int(kwargs.get("debug", False))}'
                return linear(*args, **kwargs, compiled=kernels[key])
            return fn
        launchers[name] = make(name, options)
    return manifest['configs'], launchers


@torch.inference_mode()
def main():
    from .benchmark_projection_pdl import load_layer
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--configs', nargs='+', choices=tuple(configs()), required=True)
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe tag required')
    choices = {n: configs()[n] for n in args.configs}
    folder = RESULTS/f'gateup_cluster_bundle_{args.tag}'
    folder.mkdir(exist_ok=False)
    for name in ('gateup_cluster.py', 'gateup_cluster_binary.py', 'qkv_cluster_prepare.py', 'qkv_cluster_binary.py'):
        (folder/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    torch.set_num_threads(4)
    entry = load_layer(17); raw = entry['raw']
    assert raw['has_residual']
    manifest = {'format': 'gateup_cluster_cubin_v1', 'complete': False,
        'torch': torch.__version__, 'triton': triton.__version__, 'configs': choices, 'binaries': {}}
    for name, options in choices.items():
        for add in (False, True):
            for debug in (False, True):
                _, k = linear(raw['x'][:1], raw['residual'][:1] if add else None,
                    raw['weight'], raw['eps'], *entry['weights']['up'],
                    **options, debug=debug, return_kernel=True)
                prefix = f'{name}_add{int(add)}_debug{int(debug)}'
                for suffix in ('ptx', 'ttgir', 'cubin'):
                    v = k.asm[suffix]
                    (folder/(prefix+'.'+suffix)).write_bytes(v if isinstance(v, bytes) else v.encode())
                (folder/(prefix+'.sass')).write_text(subprocess.check_output(
                    ['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(folder/(prefix+'.cubin'))], text=True))
                signature = [(key, value) for key, value in k.src.signature.items() if value.startswith('*')]
                assert all(v == 'constexpr' or v.startswith('*') for v in k.src.signature.values())
                params = re.search(r'\.visible \.entry [^(]+\((.*?)\)\s*\.reqntid', k.asm['ptx'], re.S)
                assert params and len(re.findall(r'\.param\b', params[1])) == len(signature)+2
                metadata = {key: getattr(k.metadata, key) for key in
                    ('name', 'shared', 'num_warps', 'num_ctas', 'launch_pdl', 'global_scratch_size', 'profile_scratch_size')}
                metadata['cluster_dims'] = [k.metadata.num_ctas, 1, 1]
                manifest['binaries'][prefix] = {'prefix': prefix, 'metadata': metadata,
                    'registers': k.n_regs, 'spills': k.n_spills, 'pointer_signature': signature,
                    'eps': raw['eps'], 'cubin_sha256': hashlib.sha256(k.asm['cubin']).hexdigest()}
                print('EXPORTED', prefix, k.n_regs, k.n_spills, metadata['shared'], flush=True)
    torch.cuda.synchronize()
    manifest['complete'] = True
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('SAVED', folder, flush=True)


if __name__ == '__main__':
    main()
