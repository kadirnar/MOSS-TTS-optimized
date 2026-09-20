"""Native SM90 split attention, using the caller's current CUDA stream."""
import ctypes
import functools
import hashlib
from pathlib import Path
import subprocess
import torch


@functools.lru_cache(None)
def library():
    source=Path(__file__).with_suffix('.cu')
    tag=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder=source.parent/'results'/'attention_native_build';folder.mkdir(exist_ok=True)
    so=folder/f'attention_{tag}.so'
    snapshot=folder/f'attention_{tag}.cu'
    if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    if not so.exists():
        command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_90',
            '--shared','-Xcompiler','-fPIC','-Xptxas=-v',str(source),'-o',str(so)]
        result=subprocess.run(command,text=True,capture_output=True)
        (folder/f'attention_{tag}.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
    lib=ctypes.CDLL(str(so))
    lib.launch_attention_ordered.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.launch_attention_ordered.restype=ctypes.c_int
    return lib


def launch(q,k,v,position,partial,lse):
    if q.shape!=(32,128) or k.ndim!=4 or k.shape[:2]!=(1,8) or k.shape[-1]!=128 or v.shape!=k.shape:
        raise ValueError('Native attention requires Q [32,128], KV [1,8,L,128]')
    length=k.shape[-2]
    if length%32 or partial.ndim!=3 or partial.shape[0]!=32 or partial.shape[2]!=128 or not 0<partial.shape[1]*32<=length or lse.shape!=partial.shape[:2]:
        raise ValueError('Invalid native split-attention buffers')
    tensors=(q,k,v,position,partial,lse)
    dtypes=(torch.bfloat16,torch.bfloat16,torch.bfloat16,torch.int64,torch.float32,torch.float32)
    if not q.is_cuda or position.numel()!=1 or any(t.device!=q.device or t.dtype!=dt or not t.is_contiguous() or t.data_ptr()%16 for t,dt in zip(tensors,dtypes)):
        raise ValueError('Native attention requires contiguous aligned CUDA tensors with matching dtypes/devices')
    result=library().launch_attention_ordered(q.data_ptr(),k.data_ptr(),v.data_ptr(),
        position.data_ptr(),partial.data_ptr(),lse.data_ptr(),k.shape[-2],
        partial.shape[1],torch.cuda.current_stream(q.device).cuda_stream)
    if result:raise RuntimeError(f'Native attention launch failed: CUDA error {result}')


def enable_native_attention(llm):
    if llm.graph is not None or llm.prefill_graphs:
        raise RuntimeError('Enable native attention before graph capture')
    modules=[layer.self_attn for layer in llm.model.language_model.layers]
    if any(not m._triton_decode or m._decode_backend is not None or
           (m._decode_block,m._decode_warps)!=(32,4) for m in modules):
        raise ValueError('Native attention requires the custom B32/W4 decode path')
    if torch.cuda.get_device_capability()!= (9,0):
        raise ValueError('This native attention build is validated for SM90 only')
    library()  # Build/load outside graph capture.
    for m in modules:m._native_attention=True
