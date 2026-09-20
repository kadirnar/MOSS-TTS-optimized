#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Exact signed INT8 decomposition: x = unsigned_low_nibble + 16*signed_high.
__device__ __forceinline__ uint32_t compact4(uint32_t x){
    x &= 0x0f0f0f0fu;
    x = (x | (x >> 4)) & 0x00ff00ffu;
    return (x | (x >> 8)) & 0x0000ffffu;
}
__device__ __forceinline__ uint2 split8(uint2 x){
    return make_uint2(compact4(x.x)|(compact4(x.y)<<16),
                      compact4(x.x>>4)|(compact4(x.y>>4)<<16));
}

template<int M> __device__ __forceinline__ int2 dot32(const uint32_t* w,uint2 b){
    if constexpr(M==8){
        uint32_t a=__ldg(w);int lo0,lo1,hi0,hi1;
        asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 "
            "{%0,%1},{%2},{%3},{0,0};"
            :"=r"(lo0),"=r"(lo1):"r"(a),"r"(b.x));
        asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
            "{%0,%1},{%2},{%3},{0,0};"
            :"=r"(hi0),"=r"(hi1):"r"(a),"r"(b.y));
        return make_int2(lo0+16*hi0,0);
    }else{
        uint2 a=__ldg((const uint2*)w);int l0,l1,l2,l3,h0,h1,h2,h3;
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32 "
            "{%0,%1,%2,%3},{%4,%5},{%6},{0,0,0,0};"
            :"=r"(l0),"=r"(l1),"=r"(l2),"=r"(l3):"r"(a.x),"r"(a.y),"r"(b.x));
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
            "{%0,%1,%2,%3},{%4,%5},{%6},{0,0,0,0};"
            :"=r"(h0),"=r"(h1),"=r"(h2),"=r"(h3):"r"(a.x),"r"(a.y),"r"(b.y));
        return make_int2(l0+16*h0,l2+16*h2);
    }
}

template<int M> __global__ void integer_groups(const uint32_t* w,const uint2* q,int* y){
    int lane=threadIdx.x,group=blockIdx.x;
    uint2 b=split8(__ldg(q+group*4+lane%4));
    int2 d=dot32<M>(w+group*(M*4)+lane*(M/8),b);
    if(lane%4==0){
        y[group*M+lane/4]=d.x;
        if constexpr(M==16)y[group*M+lane/4+8]=d.y;
    }
}

extern "C" int launch_mma4_integer(void* w,void* q,void* y,int groups,int m,void* stream){
    if(groups<=0)return (int)cudaErrorInvalidValue;
    if(m==8)integer_groups<8><<<groups,32,0,(cudaStream_t)stream>>>((uint32_t*)w,(uint2*)q,(int*)y);
    else if(m==16)integer_groups<16><<<groups,32,0,(cudaStream_t)stream>>>((uint32_t*)w,(uint2*)q,(int*)y);
    else return (int)cudaErrorInvalidValue;
    return (int)cudaGetLastError();
}
