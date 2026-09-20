"""Experimental native attention with immutable historical-KV preloads.

The caller must order all position/history writes before the QKV producer's
dependency wait/trigger. Only the current cache row may be written by QKV.
No current-token cache or Q read is allowed before our grid dependency wait.
"""
import ctypes
import functools
import hashlib
from pathlib import Path
import subprocess
import torch
from .common import RESULTS


@functools.lru_cache(None)
def library():
    source = Path(__file__).with_suffix('.cu')
    dependency = source.with_name('attention_pdl.cu')
    tag = hashlib.sha256(source.read_bytes() + dependency.read_bytes()).hexdigest()[:16]
    folder = RESULTS / 'attention_history_build' / tag
    folder.mkdir(parents=True, exist_ok=True)
    for path in (source, dependency, Path(__file__)):
        (folder / path.name).write_bytes(path.read_bytes())
    so = folder / 'attention.so'
    if not so.exists():
        command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_90',
                   '--shared', '-Xcompiler', '-fPIC', '-Xptxas=-v', str(source), '-o', str(so)]
        result = subprocess.run(command, text=True, capture_output=True)
        (folder / 'build.log').write_text(result.stdout + result.stderr)
        result.check_returncode()
        (folder / 'kernels.sass').write_text(subprocess.check_output(
            ['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(so)], text=True))
    lib = ctypes.CDLL(str(so))
    lib.launch_attention_history.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*5 + [ctypes.c_void_p]
    lib.launch_attention_history.restype = ctypes.c_int
    return lib


def launch(q, k, v, position, partial, lse, *, mode=3, packed=True, trigger=2):
    if mode not in (0, 1, 2, 3) or trigger not in (1, 2):
        raise ValueError('Unsupported history preload or trigger')
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError('SM90 only')
    if q.shape != (32, 128) or k.ndim != 4 or k.shape[:2] != (1, 8) or k.shape[-1] != 128 or v.shape != k.shape:
        raise ValueError('Selected Q/K/V shapes required')
    length = k.shape[-2]
    if length % 32 or partial.ndim != 3 or partial.shape[0] != 32 or partial.shape[2] != 128 or not 0 < partial.shape[1]*32 <= length or lse.shape != partial.shape[:2]:
        raise ValueError('Invalid split attention buffers')
    tensors = (q, k, v, position, partial, lse)
    dtypes = (torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int64, torch.float32, torch.float32)
    if position.numel() != 1 or any(not t.is_cuda or t.device != q.device or t.dtype != dt or
            not t.is_contiguous() or t.data_ptr() % 16 for t, dt in zip(tensors, dtypes)):
        raise ValueError('Aligned contiguous same-device typed inputs required')
    status = library().launch_attention_history(*[t.data_ptr() for t in tensors],
        length, partial.shape[1], mode, int(packed), trigger, torch.cuda.current_stream(q.device).cuda_stream)
    if status:
        raise RuntimeError(f'History attention launch failed: CUDA error {status}')


def configs():
    result = {}
    for producer in (1, 2):
        for trigger in (1, 2):
            for mode, packed in ((0, True), (1, True), (2, True), (3, True),
                                 (1, False), (2, False), (3, False)):
                name = f'q{producer}_a{trigger}_m{mode}_' + ('packed' if packed else 'float')
                result[name] = {'producer': producer, 'trigger': trigger, 'mode': mode, 'packed': packed}
    return result
