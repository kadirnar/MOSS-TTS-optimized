"""Experimental exact SM90 attention-output projection, R4/W8."""
import ctypes
import functools
import hashlib
from pathlib import Path
import subprocess
import torch


@functools.lru_cache(None)
def library():
    source=Path(__file__).with_suffix('.cu');tag=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder=source.parent/'results/dp4a_output_native_build';folder.mkdir(exist_ok=True)
    so=folder/f'output_{tag}.so';snapshot=folder/f'output_{tag}.cu'
    if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    if not so.exists():
        command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_90','--shared','-Xcompiler','-fPIC','-Xptxas=-v',str(source),'-o',str(so)]
        result=subprocess.run(command,text=True,capture_output=True)
        (folder/f'output_{tag}.log').write_text(result.stdout+result.stderr);result.check_returncode()
    lib=ctypes.CDLL(str(so));fn=lib.launch_output_native
    fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
    return lib


def linear(x,w,s,*,prequantized):
    n,k2=w.shape;q,sx=prequantized
    if k2!=2048 or n<=0 or x.numel()!=4096 or x.dtype!=torch.bfloat16:
        raise ValueError('Native output projection requires one BF16 K4096 row')
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)!=(9,0):
        raise ValueError('Native output projection is qualified only for SM90')
    if any(t.device!=x.device or not t.is_contiguous() or t.data_ptr()%16 for t in (x,w,s,q,sx)):
        raise ValueError('Aligned contiguous buffers on the same device required')
    if w.dtype!=torch.uint8 or s.dtype!=torch.float32 or s.shape!=(n,128) or q.dtype!=torch.int8 or q.numel()!=4096 or sx.dtype!=torch.float32 or sx.numel()!=128:
        raise ValueError('Interleaved INT4 weights, FP32 scales and G32 INT8 activation required')
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    code=library().launch_output_native(q.data_ptr(),sx.data_ptr(),w.data_ptr(),s.data_ptr(),y.data_ptr(),n,torch.cuda.current_stream(x.device).cuda_stream)
    if code:raise RuntimeError(f'Native output CUDA error {code}')
    return y
