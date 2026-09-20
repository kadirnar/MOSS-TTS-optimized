"""Small independent kernel trials for TileLang and NVIDIA CuTe DSL."""
import tilelang
import tilelang.language as T
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda


@tilelang.jit(out_idx=[1])
def tile_silu(N):
    @T.prim_func
    def main(X:T.Tensor((2*N,),"bfloat16"),Y:T.Tensor((N,),"bfloat16")):
        with T.Kernel(T.ceildiv(N,256),threads=128) as bx:
            for i in T.Parallel(256):
                j=bx*256+i
                if j<N:
                    a=T.cast(X[j],"float32")
                    s=T.cast(a/(1+T.exp(-a)),"bfloat16")
                    Y[j]=T.cast(s,"float32")*T.cast(X[N+j],"float32")
    return main


@cute.kernel
def cute_silu_kernel(x:cute.Tensor,y:cute.Tensor,n:cutlass.Constexpr):
    tid,_,_=cute.arch.thread_idx()
    block,_,_=cute.arch.block_idx()
    j=block*256+tid
    if j<n:
        a=x[j].to(cutlass.Float32)
        b=x[j+n].to(cutlass.Float32)
        s=(a/(1.0+cute.exp(-a))).to(cutlass.BFloat16).to(cutlass.Float32)
        y[j]=(s*b).to(cutlass.BFloat16)


@cute.jit
def cute_silu(x:cute.Tensor,y:cute.Tensor,n:cutlass.Constexpr,stream:cuda.CUstream):
    cute_silu_kernel(x,y,n).launch(grid=((n+255)//256,1,1),block=(256,1,1),stream=stream)
