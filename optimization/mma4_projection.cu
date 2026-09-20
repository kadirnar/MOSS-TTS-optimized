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

__device__ __forceinline__ float bf(float x){return __bfloat162float(__float2bfloat16_rn(x));}
__device__ __forceinline__ float warp_sum(float x){
    #pragma unroll
    for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
    return x;
}
__device__ __forceinline__ float silu(float x){
    float e,z=__fmul_rn(x,-0x1.715476p+0f),y;
    asm("ex2.approx.f32 %0,%1;":"=f"(e):"f"(z));
    float den=__fadd_rn(1.f,e);
    asm("div.full.f32 %0,%1,%2;":"=f"(y):"f"(x),"f"(den));
    return bf(y);
}
template<bool SHORT> __device__ __forceinline__ float scale(const void* p,int i,bool valid){
    if(!valid)return 0.f;
    if constexpr(SHORT)return __bfloat162float(__ldg((const __nv_bfloat16*)p+i));
    else return __ldg((const float*)p+i);
}

template<int K,int M,int ROWS,int WARPS,int U,bool SHORT,bool FUSED>
__global__ void projection(const uint2* __restrict__ q,const float* __restrict__ xs,
    const uint32_t* __restrict__ w,const void* __restrict__ scales,__nv_bfloat16* __restrict__ y,
    int8_t* __restrict__ oq,float* __restrict__ os,int n){
    constexpr int G=K/32,BG=K==4096?128:512,WG=K==4096?1:2,FC=K==4096?4:8;
    extern __shared__ float shared[];
    float* gs=shared;float* us=gs+ROWS*BG;
    float* values=us+(FUSED?ROWS*BG:0);
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,base=blockIdx.x*ROWS;
    #pragma unroll 1
    for(int tile=0;tile<ROWS/M;++tile){
        #pragma unroll U
        for(int g=warp;g<G;g+=WARPS){
            uint2 b=split8(__ldg(q+g*4+lane%4));
            int offset=(((base/M+tile)*G+g)*32+lane)*(M/8);
            int2 z=dot32<M>(w+offset,b);
            int local=tile*M+lane/4;
            if(lane%4==0){
                gs[local*BG+g]=(float)z.x;
                if constexpr(M==16)gs[(local+8)*BG+g]=(float)z.y;
            }
            if constexpr(FUSED){
                int2 u=dot32<M>(w+offset+(n/M)*G*M*4,b);
                if(lane%4==0){
                    us[local*BG+g]=(float)u.x;
                    if constexpr(M==16)us[(local+8)*BG+g]=(float)u.y;
                }
            }
        }
    }
    __syncthreads();
    // Preserve the selected G32 floating epilogue: local 1,0,2,3 order,
    // sequential FMAs, descending warp XOR, then the down warp-pair sum.
    #pragma unroll
    for(int local=warp/WG;local<ROWS;local+=WARPS/WG){
        float a[FC],b[FC],c[FC],d[FC];int row=base+local;
        #pragma unroll
        for(int j=0;j<FC;++j){
            int g=lane*4+j%4+(K==4096?0:(warp%2)*128+(j/4)*256);
            bool valid=g<G&&row<n;
            float sx=g<G?__ldg(xs+g):0.f;
            a[j]=valid?gs[local*BG+g]:0.f;b[j]=__fmul_rn(scale<SHORT>(scales,row*G+g,valid),sx);
            if constexpr(FUSED){c[j]=valid?us[local*BG+g]:0.f;d[j]=__fmul_rn(scale<SHORT>(scales,(row+n)*G+g,valid),sx);}
        }
        float z=__fmul_rn(a[1],b[1]);z=__fmaf_rn(a[0],b[0],z);
        #pragma unroll
        for(int j=2;j<FC;++j)z=__fmaf_rn(a[j],b[j],z);
        z=warp_sum(z);
        if constexpr(FUSED){
            float u=__fmul_rn(c[1],d[1]);u=__fmaf_rn(c[0],d[0],u);
            #pragma unroll
            for(int j=2;j<FC;++j)u=__fmaf_rn(c[j],d[j],u);
            u=warp_sum(u);z=bf(__fmul_rn(silu(bf(z)),bf(u)));
            if(lane==0&&row<n){values[local]=z;y[row]=__float2bfloat16_rn(z);}
        }else if constexpr(WG==2){
            if(lane==0)values[local*2+warp%2]=z;
        }else if(lane==0&&row<n)y[row]=__float2bfloat16_rn(z);
    }
    if constexpr(FUSED){
        __syncthreads();
        if(warp<ROWS/32){
            float z=values[warp*32+lane],maximum=fabsf(z);
            #pragma unroll
            for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
            float s=fmaxf(__fdiv_rn(maximum,127.f),1e-8f),inv=__fdiv_rn(1.f,s);
            oq[base+warp*32+lane]=(int8_t)__float2int_rn(__fmul_rn(z,inv));
            if(lane==0)os[base/32+warp]=s;
        }
    }else if constexpr(WG==2){
        __syncthreads();
        if(tid<ROWS&&base+tid<n)y[base+tid]=__float2bfloat16_rn(__fadd_rn(values[tid*2],values[tid*2+1]));
    }
}

template<int K,int M,int WARPS,int U,bool SHORT,bool FUSED>
int run(void* q,void* xs,void* w,void* s,void* y,void* oq,void* os,int n,void* stream){
    constexpr int ROWS=FUSED?32:M,BG=K==4096?128:512;
    constexpr int BYTES=(ROWS*BG*(FUSED?2:1)+ROWS*2)*4;
    projection<K,M,ROWS,WARPS,U,SHORT,FUSED><<<(n+ROWS-1)/ROWS,WARPS*32,BYTES,(cudaStream_t)stream>>>(
        (uint2*)q,(float*)xs,(uint32_t*)w,s,(__nv_bfloat16*)y,(int8_t*)oq,(float*)os,n);
    return (int)cudaGetLastError();
}

#define RUN(M,W,U) if(m==M&&warps==W&&unroll==U)return run<K,M,W,U,SHORT,FUSED>(q,xs,w,s,y,oq,os,n,stream)
template<int K,bool SHORT,bool FUSED>
int choose(void* q,void* xs,void* w,void* s,void* y,void* oq,void* os,int n,int m,int warps,int unroll,void* stream){
    RUN(8,4,1);RUN(8,4,4);RUN(8,8,1);RUN(8,8,4);
    RUN(16,4,1);RUN(16,4,4);RUN(16,8,1);RUN(16,8,4);
    return (int)cudaErrorInvalidValue;
}
#undef RUN
extern "C" int launch_mma4_projection(void* q,void* xs,void* w,void* s,void* y,void* oq,void* os,
    int n,int k,int short_scale,int fused,int m,int warps,int unroll,void* stream){
    if(n<=0)return (int)cudaErrorInvalidValue;
    if(fused&&k==4096&&short_scale&&n%32==0)return choose<4096,true,true>(q,xs,w,s,y,oq,os,n,m,warps,unroll,stream);
    if(!fused&&k==4096&&short_scale)return choose<4096,true,false>(q,xs,w,s,y,oq,os,n,m,warps,unroll,stream);
    if(!fused&&k==4096&&!short_scale)return choose<4096,false,false>(q,xs,w,s,y,oq,os,n,m,warps,unroll,stream);
    if(!fused&&k==12288&&!short_scale)return choose<12288,false,false>(q,xs,w,s,y,oq,os,n,m,warps,unroll,stream);
    return (int)cudaErrorInvalidValue;
}
