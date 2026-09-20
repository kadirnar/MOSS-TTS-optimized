"""Kernel-level trials; these numbers are NOT full SGLang/vLLM server TTFA."""
import ctypes
import json
import subprocess
from pathlib import Path
import torch
import triton
import triton.language as tl
from .common import RESULTS,ROOT,save_json


@triton.jit
def _silu(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    a=tl.load(X+i,i<N,0).to(tl.float32)
    b=tl.load(X+N+i,i<N,0).to(tl.float32)
    s=(a/(1+tl.exp(-a))).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,s*b,i<N)


@triton.jit
def _gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,R:tl.constexpr,BK:tl.constexpr):
    r=tl.program_id(0)*R+tl.arange(0,R)
    k=tl.arange(0,BK)
    x=tl.load(X+k,k<K,0).to(tl.float32)
    w=tl.load(W+r[:,None]*K+k[None,:],(r[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
    y=tl.sum(w*x[None,:],1)
    tl.store(Y+r,y,r<N)


def measure(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(30):fn()
    times=[]
    for _ in range(7):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record();graph.replay();e.record();e.synchronize()
        times.append(s.elapsed_time(e)*1000/30)
    return sorted(times)[3]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    n=12288
    x=torch.randn((2*n,),device='cuda',dtype=torch.bfloat16)
    y=torch.empty((n,),device='cuda',dtype=torch.bfloat16)
    ref=torch.nn.functional.silu(x[:n])*x[n:]
    results={}
    def trial(name,fn):
        try:
            out=fn()
            results[name]={'us':measure(fn),'max_abs_error':(out-ref).abs().max().item(),
                'bitwise_equal':torch.equal(out,ref)}
        except Exception as e:
            results[name]={'error':type(e).__name__+': '+str(e)[:1000]}
        print(name,results[name],flush=True)
        save_json('library_trials.json',results)
    trial('torch_silu',lambda:torch.nn.functional.silu(x[:n])*x[n:])
    def tri():_silu[(triton.cdiv(n,256),)](x,y,n,256);return y
    trial('triton_silu',tri)
    try:
        import sgl_kernel
        trial('sglang_silu',lambda:sgl_kernel.silu_and_mul(x[None])[0])
        import vllm._C
        def vll():torch.ops._C.silu_and_mul(y[None],x[None]);return y
        trial('vllm_silu',vll)
    except Exception as e:results['library_import']={'error':str(e)}
    subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_90','--shared','-Xcompiler','-fPIC',
        str(ROOT/'optimization/native_kernels.cu'),'-o',str(RESULTS/'native_kernels.so')],check=True)
    lib=ctypes.CDLL(str(RESULTS/'native_kernels.so'))
    lib.launch_silu.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    def native():
        status=lib.launch_silu(x.data_ptr(),y.data_ptr(),n,torch.cuda.current_stream().cuda_stream)
        if status:raise RuntimeError(f'CUDA launch status {status}')
        return y
    trial('cuda_c_silu',native)
    try:
        from .alternative_kernels import tile_silu,cute_silu
        tk=tile_silu(n)
        trial('tilelang_silu',lambda:tk(x))
        from cutlass.cute.runtime import from_dlpack
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda
        xc,yc=from_dlpack(x),from_dlpack(y)
        compiled=cute.compile(cute_silu,xc,yc,n,cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        def cu():compiled(xc,yc,cuda.CUstream(torch.cuda.current_stream().cuda_stream));return y
        trial('cute_dsl_silu',cu)
    except Exception as e:results['dsl_build']={'error':type(e).__name__+': '+str(e)[:1500]}
    # The two largest single-token projection shapes in this Qwen3 backbone.
    for rows,cols in [(24576,4096),(4096,12288)]:
        x=torch.randn((cols,),device='cuda',dtype=torch.bfloat16)
        w=torch.randn((rows,cols),device='cuda',dtype=torch.bfloat16)*0.02
        y=torch.empty((rows,),device='cuda',dtype=torch.bfloat16)
        ref=torch.nn.functional.linear(x,w)
        trial(f'cublas_gemv_{rows}x{cols}',lambda:torch.nn.functional.linear(x,w))
        for r in [1,2,4]:
            def tri_gemv():_gemv[(triton.cdiv(rows,r),)](x,w,y,rows,cols,r,triton.next_power_of_2(cols),num_warps=4 if cols<=4096 else 8);return y
            trial(f'triton_gemv_r{r}_{rows}x{cols}',tri_gemv)
        lib.launch_gemv.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        def nat_gemv():lib.launch_gemv(x.data_ptr(),w.data_ptr(),y.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream);return y
        trial(f'cuda_c_gemv_{rows}x{cols}',nat_gemv)
    save_json('library_trials.json',results)

if __name__=='__main__':main()
