#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

// Preserve the selected Triton 3.7.1, B32/W4 reduction order explicitly.
// Separate shared buffers eliminate reuse barriers between the three reductions.
__device__ __forceinline__ float add(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float mul(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float exp_approx(float x){
    float y; x=mul(x,0x1.715476p+0f);
    asm("ex2.approx.f32 %0, %1;":"=f"(y):"f"(x));return y;
}
__device__ __forceinline__ float div_full(float x,float y){
    float z;asm("div.full.f32 %0, %1, %2;":"=f"(z):"f"(x),"f"(y));return z;
}
__device__ __forceinline__ float four_sum(const float* x,int stride){
    return add(add(x[0],x[2*stride]),add(x[stride],x[3*stride]));
}
__device__ __forceinline__ void load8(const __nv_bfloat16* p,float (&x)[8],bool valid=true){
    uint4 packed=make_uint4(0,0,0,0);
    if(valid)packed=__ldg(reinterpret_cast<const uint4*>(p));
    unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
    #pragma unroll
    for(int j=0;j<4;++j){
        x[2*j]=__bfloat162float(__ushort_as_bfloat16(words[j]&65535));
        x[2*j+1]=__bfloat162float(__ushort_as_bfloat16(words[j]>>16));
    }
}

__global__ void attention_ordered(const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,const __nv_bfloat16* __restrict__ v,
    const int64_t* __restrict__ position,float* __restrict__ partial,
    float* __restrict__ lse,int length,int splits){
    const int head=blockIdx.x,split=blockIdx.y,tid=threadIdx.x;
    const int lane=tid%32,warp=tid/32,d=(lane%16)*8;
    const int p=*position;
    if(split*32>p){
        partial[(head*splits+split)*128+tid]=0;
        if(tid==0)lse[head*splits+split]=-INFINITY;
        return;
    }
    __shared__ float maxima[4],denominators[4],weighted[4][128];
    float qq[8],logits[4],prob[4];
    load8(q+head*128+d,qq);
    #pragma unroll
    for(int r=0;r<4;++r){
        int t=split*32+warp*2+lane/16+r*8;
        float kk[8];
        load8(k+((head/4)*length+t)*128+d,kk,t<=p);
        float z=mul(qq[1],kk[1]);
        z=__fmaf_rn(qq[0],kk[0],z);
        #pragma unroll
        for(int j=2;j<8;++j)z=__fmaf_rn(qq[j],kk[j],z);
        #pragma unroll
        for(int offset=8;offset>0;offset/=2)z=add(z,__shfl_xor_sync(0xffffffff,z,offset));
        logits[r]=t<=p?mul(z,0x1.6a09e6p-4f):-INFINITY;
    }
    float maximum=fmaxf(fmaxf(fmaxf(logits[0],logits[1]),logits[2]),logits[3]);
    maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,16));
    if(lane==0)maxima[warp]=maximum;
    __syncthreads();
    maximum=fmaxf(fmaxf(maxima[0],maxima[2]),fmaxf(maxima[1],maxima[3]));
    #pragma unroll
    for(int r=0;r<4;++r)prob[r]=exp_approx(__fsub_rn(logits[r],maximum));
    float denominator=add(add(add(prob[0],prob[1]),prob[2]),prob[3]);
    denominator=add(denominator,__shfl_xor_sync(0xffffffff,denominator,16));
    if(lane==0)denominators[warp]=denominator;
    __syncthreads();
    denominator=four_sum(denominators,1);
    float vv[4][8];
    #pragma unroll
    for(int r=0;r<4;++r){
        int t=split*32+warp*2+lane/16+r*8;
        load8(v+((head/4)*length+t)*128+d,vv[r],t<=p);
    }
    #pragma unroll
    for(int j=0;j<8;++j){
        float z=mul(prob[1],vv[1][j]);
        z=__fmaf_rn(prob[0],vv[0][j],z);
        z=__fmaf_rn(prob[2],vv[2][j],z);
        z=__fmaf_rn(prob[3],vv[3][j],z);
        z=add(z,__shfl_xor_sync(0xffffffff,z,16));
        if(lane<16)weighted[warp][d+j]=z;
    }
    __syncthreads();
    partial[(head*splits+split)*128+tid]=div_full(four_sum(&weighted[0][tid],128),denominator);
    if(tid==0)lse[head*splits+split]=add(maximum,logf(denominator));
}

extern "C" int launch_attention_ordered(void* q,void* k,void* v,void* position,
    void* partial,void* lse,int length,int splits,void* stream){
    attention_ordered<<<dim3(32,splits),128,0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,
        (const int64_t*)position,(float*)partial,(float*)lse,length,splits);
    return (int)cudaGetLastError();
}
