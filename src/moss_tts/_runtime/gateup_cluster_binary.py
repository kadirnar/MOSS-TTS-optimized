"""Export gate/up cluster cubins with Triton 3.8; run in selected host runtime."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch
from .cuda_driver import Attribute, LaunchConfig, check
from .qkv_cluster_binary import Binary as BaseBinary
from .gateup_cluster import linear


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
