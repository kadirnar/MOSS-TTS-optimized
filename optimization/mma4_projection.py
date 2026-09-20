"""Native CUDA experiment issuing INT4 PTX with exact INT8 decomposition.

The inspected SM90 machine code emulates these INT4 matrix instructions through
INT8 IMMA and conversion sequences. This is not a native INT4 hardware fast path.
"""
import ctypes
import functools
import hashlib
from pathlib import Path
import subprocess

import torch


@functools.lru_cache(None)
def library():
    source=Path(__file__).with_suffix('.cu');tag=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder=source.parent/'results/mma4_build';folder.mkdir(exist_ok=True)
    binary=folder/f'mma4_{tag}.so'
    if not binary.exists():
        command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_90','--shared','-Xcompiler','-fPIC','-Xptxas=-v',str(source),'-o',str(binary)]
        result=subprocess.run(command,text=True,capture_output=True)
        (folder/f'mma4_{tag}.log').write_text(result.stdout+result.stderr)
        (folder/f'mma4_{tag}.cu').write_bytes(source.read_bytes());result.check_returncode()
    lib=ctypes.CDLL(str(binary));fn=lib.launch_mma4_integer
    fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p];fn.restype=ctypes.c_int
    fn=lib.launch_mma4_projection
    fn.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*7+[ctypes.c_void_p];fn.restype=ctypes.c_int
    return lib


def pack_groups(signed):
    """[group, M, 32] signed nibbles into contiguous per-warp A fragments."""
    if signed.ndim!=3 or signed.shape[1] not in (8,16) or signed.shape[2]!=32 or signed.dtype!=torch.int8:
        raise ValueError('Signed INT4 groups with M8/M16 rows required')
    if bool(((signed < -8)|(signed > 7)).any()):raise ValueError('INT4 range exceeded')
    groups,m,_=signed.shape
    codes=signed.to(torch.uint8)&15
    packed=(codes[...,::2]|(codes[...,1::2]<<4)).contiguous().view(torch.int32)
    return packed.reshape(groups,m//8,8,4).permute(0,2,3,1).contiguous().reshape(groups,m*4).view(torch.uint8)


def integer(w,q,m):
    if m not in (8,16) or q.ndim!=2 or q.shape[1]!=32 or q.dtype!=torch.int8 or w.numel()!=q.shape[0]*m*16 or w.dtype!=torch.uint8:
        raise ValueError('Invalid native integer buffers')
    if not q.is_cuda or any(t.device!=q.device or not t.is_contiguous() or t.data_ptr()%16 for t in (q,w)):
        raise ValueError('Aligned contiguous CUDA tensors required')
    y=torch.empty((q.shape[0],m),device=q.device,dtype=torch.int32)
    code=library().launch_mma4_integer(w.data_ptr(),q.data_ptr(),y.data_ptr(),q.shape[0],m,torch.cuda.current_stream(q.device).cuda_stream)
    if code:raise RuntimeError(f'INT4 MMA CUDA error {code}')
    return y


def pack_weight(packed,m):
    """Original consecutive-nibble [N,K/2] storage to [tile,group,lane,reg]."""
    if packed.dtype!=torch.uint8 or packed.ndim!=2 or not packed.is_contiguous() or packed.shape[1]%16 or m not in (8,16):
        raise ValueError('Contiguous original INT4 weights and M8/M16 required')
    n,k2=packed.shape;g=k2//16;pad=(-n)%m
    if pad:packed=torch.cat((packed,torch.zeros(pad,k2,device=packed.device,dtype=packed.dtype)))
    return packed.view(torch.int32).reshape(-1,m,g,4).permute(0,2,1,3).reshape(-1,g,m//8,8,4).permute(0,1,3,4,2).contiguous().reshape(-1,m*4).view(torch.uint8)


def unpack_weight(packed,n,k,m):
    padded=(n+m-1)//m*m;g=k//32
    return packed.view(torch.int32).reshape(padded//m,g,8,4,m//8).permute(0,4,2,1,3).contiguous().reshape(padded,k//8).view(torch.uint8)[:n]


def linear(x,w,s,*,m,warps,unroll,fused=False,prequantized):
    q,sx=prequantized;k=x.numel();n2=s.shape[0];n=n2//2 if fused else n2
    if k not in (4096,12288) or m not in (8,16) or warps not in (4,8) or unroll not in (1,4):raise ValueError('Specialized MMA configuration required')
    if x.dtype!=torch.bfloat16 or w.dtype!=torch.uint8 or s.dtype not in (torch.bfloat16,torch.float32) or s.shape!=(n2,k//32) or q.dtype!=torch.int8 or q.numel()!=k or sx.dtype!=torch.float32 or sx.numel()!=k//32:
        raise ValueError('Invalid MMA activation/scale buffers')
    if w.numel()!=((n2+m-1)//m)*m*(k//2):raise ValueError('Invalid packed weight size')
    if fused and (k!=4096 or n2%64 or s.dtype!=torch.bfloat16):raise ValueError('Fused G32 output requires aligned K4096 BF16 scales')
    if not x.is_cuda or any(t.device!=x.device or not t.is_contiguous() or t.data_ptr()%16 for t in (x,w,s,q,sx)):
        raise ValueError('Aligned contiguous CUDA buffers required')
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=torch.bfloat16)
    oq=torch.empty(n if fused else 0,device=x.device,dtype=torch.int8);os=torch.empty(n//32 if fused else 0,device=x.device)
    code=library().launch_mma4_projection(q.data_ptr(),sx.data_ptr(),w.data_ptr(),s.data_ptr(),y.data_ptr(),oq.data_ptr(),os.data_ptr(),n,k,int(s.dtype==torch.bfloat16),int(fused),m,warps,unroll,torch.cuda.current_stream(x.device).cuda_stream)
    if code:raise RuntimeError(f'INT4 MMA projection CUDA error {code}')
    return (y,(oq,os)) if fused else y
