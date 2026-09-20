#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// Exact selected Triton 3.7.1 reduction: local products 1,0,2,3, then
// remaining register groups; warp XOR 16,8,4,2,1; down adds two warps.
__device__ __forceinline__ float rndbf(float x){return __bfloat162float(__float2bfloat16_rn(x));}
__device__ __forceinline__ float warp_sum(float x){
    #pragma unroll
    for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
    return x;
}
__device__ __forceinline__ float full_div(float x,float y){
    float out;asm("div.full.f32 %0, %1, %2;":"=f"(out):"f"(x),"f"(y));return out;
}
__device__ __forceinline__ float silu(float x){
    float e;float z=__fmul_rn(x,-0x1.715476p+0f);
    asm("ex2.approx.f32 %0, %1;":"=f"(e):"f"(z));
    return rndbf(full_div(x,__fadd_rn(1.f,e)));
}
struct Activation {uint4 a,b;};
__device__ __forceinline__ Activation load_a(const int8_t* q,int group,bool valid){
    Activation a{make_uint4(0,0,0,0),make_uint4(0,0,0,0)};
    if(valid){a.a=__ldg((const uint4*)(q+group*32));a.b=__ldg((const uint4*)(q+group*32+16));}
    return a;
}
__device__ __forceinline__ int dot8(uint32_t w,uint32_t a,uint32_t b,int sum){
    int lo=(int)((w<<4)&0xf0f0f0f0u),hi=(int)(w&0xf0f0f0f0u);
    sum=__dp4a(lo,(int)a,sum);return __dp4a(hi,(int)b,sum);
}
__device__ __forceinline__ int dot32(const uint32_t* w,const Activation& a,bool valid){
    uint4 b=make_uint4(0,0,0,0);if(valid)b=__ldg((const uint4*)w);
    // Four independent two-instruction chains expose integer dot parallelism.
    // Their exact INT32 sum fits even at signed-byte extrema.
    int z0=dot8(b.x,a.a.x,a.a.y,0),z1=dot8(b.y,a.a.z,a.a.w,0);
    int z2=dot8(b.z,a.b.x,a.b.y,0),z3=dot8(b.w,a.b.z,a.b.w,0);
    return ((z0+z1)+(z2+z3))>>4;
}
template<bool SHORT> __device__ __forceinline__ float load_scale(const void* s,int i,bool valid){
    if(!valid)return 0.f;
    if constexpr(SHORT)return __bfloat162float(__ldg((const __nv_bfloat16*)s+i));
    else return __ldg((const float*)s+i);
}
template<int K,bool SHORT,bool CACHE,bool PERM> __device__ __forceinline__ float projection(
    const int8_t* q,const float* xs,const uint32_t* w,const void* s,int row,bool valid_row,
    const Activation* cached,const float* xsc,int part){
    constexpr int G=K/32,C=K==4096?4:8;
    int lane=threadIdx.x%32;float sums[C],scales[C];
    #pragma unroll
    for(int j=0;j<C;++j){
        int g=lane*4+(j%4)+(K==4096?0:part*128+(j/4)*256);bool valid=g<G;
        Activation a;if constexpr(CACHE)a=cached[j];else a=load_a(q,g,valid);
        int physical=PERM?(g/128)*128+(g%4)*32+(g%128)/4:g;
        int dot=dot32(w+row*(K/8)+physical*4,a,valid&&valid_row);
        float sx;if constexpr(CACHE)sx=xsc[j];else sx=valid?__ldg(xs+g):0.f;
        sums[j]=(float)dot;scales[j]=__fmul_rn(load_scale<SHORT>(s,row*G+physical,valid&&valid_row),sx);
    }
    float value=__fmul_rn(sums[1],scales[1]);
    value=__fmaf_rn(sums[0],scales[0],value);
    #pragma unroll
    for(int j=2;j<C;++j)value=__fmaf_rn(sums[j],scales[j],value);
    return warp_sum(value);
}

template<int K,bool SHORT,bool FUSED,int WARPS,int REPEAT,bool CACHE,bool PERM>
__global__ void native_projection(const int8_t* __restrict__ q,const float* __restrict__ xs,
    const uint32_t* __restrict__ w,const void* __restrict__ s,__nv_bfloat16* __restrict__ y,
    int8_t* __restrict__ oq,float* __restrict__ os,int n){
    constexpr int WG=K==4096?1:2;
    constexpr int ROWS=FUSED?32:WARPS/WG*REPEAT;
    constexpr int COUNT=FUSED?32/WARPS:REPEAT;
    constexpr int C=K==4096?4:8;
    int lane=threadIdx.x%32,warp=threadIdx.x/32,part=warp%WG;
    Activation cache[C];float xsc[C];
    if constexpr(CACHE){
        #pragma unroll
        for(int j=0;j<C;++j){
            int g=lane*4+j%4+(K==4096?0:part*128+(j/4)*256);bool valid=g<K/32;
            cache[j]=load_a(q,g,valid);xsc[j]=valid?__ldg(xs+g):0.f;
        }
    }
    __shared__ float shared[ROWS*WG];
    #pragma unroll
    for(int it=0;it<COUNT;++it){
        int local=warp/WG+it*(WARPS/WG),row=blockIdx.x*ROWS+local;
        float value=projection<K,SHORT,CACHE,PERM>(q,xs,w,s,row,row<n,cache,xsc,part);
        if constexpr(FUSED){
            float up=projection<K,SHORT,CACHE,PERM>(q,xs,w,s,row+n,row<n,cache,xsc,part);
            value=rndbf(__fmul_rn(silu(rndbf(value)),rndbf(up)));
            if(lane==0){shared[local]=value;if(row<n)y[row]=__float2bfloat16_rn(value);}
        }else if constexpr(WG==2){
            if(lane==0)shared[local*2+part]=value;
        }else{
            if(lane==0&&row<n)y[row]=__float2bfloat16_rn(value);
        }
    }
    if constexpr(FUSED){
        __syncthreads();
        if(warp==0){
            float value=shared[lane],maximum=fabsf(value);
            #pragma unroll
            for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
            float scale=fmaxf(__fdiv_rn(maximum,127.f),1e-8f),inv=__fdiv_rn(1.f,scale);
            int row=blockIdx.x*32+lane;
            if(row<n)oq[row]=(int8_t)__float2int_rn(__fmul_rn(value,inv));
            if(lane==0)os[blockIdx.x]=scale;
        }
    }else if constexpr(WG==2){
        __syncthreads();
        if(threadIdx.x<ROWS){int row=blockIdx.x*ROWS+threadIdx.x;
            if(row<n)y[row]=__float2bfloat16_rn(__fadd_rn(shared[threadIdx.x*2],shared[threadIdx.x*2+1]));}
    }
}

template<int K,bool SHORT,bool FUSED,int WARPS,int REPEAT,bool CACHE,bool PERM>
int invoke(void* q,void* xs,void* w,void* s,void* y,void* oq,void* os,int n,void* stream){
    constexpr int ROWS=FUSED?32:WARPS/(K==4096?1:2)*REPEAT;
    native_projection<K,SHORT,FUSED,WARPS,REPEAT,CACHE,PERM><<<(n+ROWS-1)/ROWS,WARPS*32,0,(cudaStream_t)stream>>>(
        (int8_t*)q,(float*)xs,(uint32_t*)w,s,(__nv_bfloat16*)y,(int8_t*)oq,(float*)os,n);
    return (int)cudaGetLastError();
}
#define RUN(W,R) if(warps==W&&repeat==R){if(cache&1)return invoke<K,SHORT,FUSED,W,R,true,PERM>(q,xs,w,s,y,oq,os,n,stream);else return invoke<K,SHORT,FUSED,W,R,false,PERM>(q,xs,w,s,y,oq,os,n,stream);}
template<int K,bool SHORT,bool FUSED,bool PERM>
int choose(void* q,void* xs,void* w,void* s,void* y,void* oq,void* os,int n,int warps,int repeat,int cache,void* stream){
    if constexpr(FUSED){RUN(4,1);RUN(8,1);}
    else {RUN(2,1);RUN(2,2);RUN(2,4);RUN(4,1);RUN(4,2);RUN(4,4);RUN(8,1);RUN(8,2);RUN(8,4);}
    return (int)cudaErrorInvalidValue;
}
#undef RUN
template<bool PERM> int dispatch_dp4a(void* q,void* xs,void* w,void* s,void* y,void* oq,void* os,
    int n,int k,int short_scale,int fused,int warps,int repeat,int cache,void* stream){
    if(fused&&k==4096&&short_scale)return choose<4096,true,true,PERM>(q,xs,w,s,y,oq,os,n,warps,repeat,cache,stream);
    if(!fused&&k==4096&&short_scale)return choose<4096,true,false,PERM>(q,xs,w,s,y,oq,os,n,warps,repeat,cache,stream);
    if(!fused&&k==4096&&!short_scale)return choose<4096,false,false,PERM>(q,xs,w,s,y,oq,os,n,warps,repeat,cache,stream);
    if(!fused&&k==12288&&!short_scale)return choose<12288,false,false,PERM>(q,xs,w,s,y,oq,os,n,warps,repeat,cache,stream);
    return (int)cudaErrorInvalidValue;
}

extern "C" int launch_dp4a_native(void* q,void* xs,void* w,void* s,void* y,void* oq,void* os,
    int n,int k,int short_scale,int fused,int warps,int repeat,int cache,void* stream){
    if(cache&2)return dispatch_dp4a<true>(q,xs,w,s,y,oq,os,n,k,short_scale,fused,warps,repeat,cache,stream);
    return dispatch_dp4a<false>(q,xs,w,s,y,oq,os,n,k,short_scale,fused,warps,repeat,cache,stream);
}
