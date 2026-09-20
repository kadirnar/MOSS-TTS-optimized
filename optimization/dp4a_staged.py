"""Experimental native SM90 projection with explicit old arithmetic order."""
import ctypes
import functools
import hashlib
from pathlib import Path
import subprocess
import torch


@functools.lru_cache(None)
def library():
    source=Path(__file__).with_suffix('.cu');tag=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder=source.parent/'results/dp4a_staged_build';folder.mkdir(exist_ok=True)
    so=folder/f'dp4a_{tag}.so';snapshot=folder/f'dp4a_{tag}.cu'
    if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    if not so.exists():
        command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_90','--shared','-Xcompiler','-fPIC','-Xptxas=-v',str(source),'-o',str(so)]
        result=subprocess.run(command,text=True,capture_output=True)
        (folder/f'dp4a_{tag}.log').write_text(result.stdout+result.stderr);result.check_returncode()
    lib=ctypes.CDLL(str(so));fn=lib.launch_dp4a_staged
    fn.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*6+[ctypes.c_void_p];fn.restype=ctypes.c_int
    return lib


def linear(x,w,s,*,fused=False,rows=4,warps=4,prequantized):
    n2,k2=w.shape;k=k2*2;n=n2//2 if fused else n2
    if k not in (4096,12288) or warps not in (4,8) or rows not in ((32,64) if fused else (4,8,16)) or x.numel()!=k or x.dtype!=torch.bfloat16:
        raise ValueError('Specialized native projection shape/configuration required')
    if fused and (k!=4096 or n2%64 or s.dtype!=torch.bfloat16 or n%rows):
        raise ValueError('Native fused gate/up requires K4096 and complete G32 groups')
    q,sx=prequantized
    tensors=(x,w,s,q,sx)
    if not x.is_cuda or any(not t.is_contiguous() or t.device!=x.device or t.data_ptr()%16 for t in tensors):
        raise ValueError('Aligned contiguous CUDA tensors on the same device required')
    if w.dtype!=torch.uint8 or s.shape!=(n2,k//32) or s.dtype not in (torch.bfloat16,torch.float32) or q.dtype!=torch.int8 or q.numel()!=k or sx.dtype!=torch.float32 or sx.numel()!=k//32:
        raise ValueError('Invalid packed weight or activation buffers')
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=torch.bfloat16)
    oq=torch.empty(n if fused else 0,device=x.device,dtype=torch.int8);os=torch.empty(n//32 if fused else 0,device=x.device)
    code=library().launch_dp4a_staged(q.data_ptr(),sx.data_ptr(),w.data_ptr(),s.data_ptr(),y.data_ptr(),oq.data_ptr(),os.data_ptr(),n,k,int(s.dtype==torch.bfloat16),int(fused),rows,warps,torch.cuda.current_stream(x.device).cuda_stream)
    if code:raise RuntimeError(f'Native projection CUDA error {code}')
    return (y,(oq,os)) if fused else y
