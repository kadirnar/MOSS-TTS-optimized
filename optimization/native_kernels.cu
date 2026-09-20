#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

__global__ void silu_mul(const __nv_bfloat16* x, __nv_bfloat16* y, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
        float a=__bfloat162float(x[i]);
        float b=__bfloat162float(x[i+n]);
        // Preserve the rounding of the upstream BF16 SiLU intermediate.
        float s=__bfloat162float(__float2bfloat16_rn(a/(1.0f+expf(-a))));
        y[i]=__float2bfloat16_rn(s*b);
    }
}
extern "C" int launch_silu(void* x,void* y,int n,void* stream){
    silu_mul<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((__nv_bfloat16*)x,(__nv_bfloat16*)y,n);
    return (int)cudaGetLastError();
}

__global__ void gemv(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,int n,int k){
    int lane=threadIdx.x%32;
    int row=blockIdx.x*4+threadIdx.x/32;
    float v=0;
    if(row<n){
        for(int j=lane;j<k;j+=32) v=fmaf(__bfloat162float(w[row*k+j]),__bfloat162float(x[j]),v);
        for(int s=16;s>0;s/=2)v+=__shfl_down_sync(0xffffffff,v,s);
        if(lane==0)y[row]=__float2bfloat16_rn(v);
    }
}
extern "C" int launch_gemv(void*x,void*w,void*y,int n,int k,void*stream){
    gemv<<<(n+3)/4,128,0,(cudaStream_t)stream>>>((__nv_bfloat16*)x,(__nv_bfloat16*)w,(__nv_bfloat16*)y,n,k);
    return (int)cudaGetLastError();
}
