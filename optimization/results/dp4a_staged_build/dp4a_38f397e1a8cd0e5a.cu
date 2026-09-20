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
    int sum=dot8(b.x,a.a.x,a.a.y,0);
    sum=dot8(b.y,a.a.z,a.a.w,sum);sum=dot8(b.z,a.b.x,a.b.y,sum);sum=dot8(b.w,a.b.z,a.b.w,sum);
    return sum>>4;
}
template<bool SHORT> __device__ __forceinline__ float load_scale(const void* s,int i,bool valid){
    if(!valid)return 0.f;
    if constexpr(SHORT)return __bfloat162float(__ldg((const __nv_bfloat16*)s+i));
    else return __ldg((const float*)s+i);
}

// Integer groups use coalesced loads; one shared-memory handoff gives each
// warp the exact four-group register order used by the selected Triton path.
template<int K,bool SHORT,bool FUSED,int ROWS,int WARPS>
__global__ void staged_projection(const int8_t* __restrict__ q,const float* __restrict__ xs,
    const uint32_t* __restrict__ w,const void* __restrict__ s,__nv_bfloat16* __restrict__ y,
    int8_t* __restrict__ oq,float* __restrict__ os,int n){
    constexpr int G=K/32,BG=K==4096?128:512,THREADS=WARPS*32,IG=BG<THREADS?BG:THREADS;
    constexpr int RG=THREADS/IG,GC=BG/IG,WG=K==4096?1:2,FC=K==4096?4:8;
    extern __shared__ float buffer[];
    float* gs=buffer;float* gc=gs+ROWS*BG;float* us=gc+ROWS*BG;
    float* uc=us+(FUSED?ROWS*BG:0);float* values=uc+(FUSED?ROWS*BG:0);
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,group=tid%IG;
    Activation ax[GC];float xsc[GC];
    #pragma unroll
    for(int j=0;j<GC;++j){int g=group+j*IG;ax[j]=load_a(q,g,g<G);xsc[j]=g<G?__ldg(xs+g):0.f;}
    #pragma unroll
    for(int r=0;r<ROWS/RG;++r){
        int local=tid/IG+r*RG,row=blockIdx.x*ROWS+local;
        #pragma unroll
        for(int j=0;j<GC;++j){
            int g=group+j*IG;bool valid=g<G&&row<n;
            gs[local*BG+g]=(float)dot32(w+row*(K/8)+g*4,ax[j],valid);
            gc[local*BG+g]=__fmul_rn(load_scale<SHORT>(s,row*G+g,valid),xsc[j]);
            if constexpr(FUSED){
                us[local*BG+g]=(float)dot32(w+(row+n)*(K/8)+g*4,ax[j],valid);
                uc[local*BG+g]=__fmul_rn(load_scale<SHORT>(s,(row+n)*G+g,valid),xsc[j]);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for(int local=warp/WG;local<ROWS;local+=WARPS/WG){
        float a[FC],b[FC],c[FC],d[FC];
        #pragma unroll
        for(int j=0;j<FC;++j){
            int g=lane*4+j%4+(K==4096?0:(warp%2)*128+(j/4)*256);
            a[j]=gs[local*BG+g];b[j]=gc[local*BG+g];
            if constexpr(FUSED){c[j]=us[local*BG+g];d[j]=uc[local*BG+g];}
        }
        float z=__fmul_rn(a[1],b[1]);z=__fmaf_rn(a[0],b[0],z);
        #pragma unroll
        for(int j=2;j<FC;++j)z=__fmaf_rn(a[j],b[j],z);
        z=warp_sum(z);
        if constexpr(FUSED){
            float u=__fmul_rn(c[1],d[1]);u=__fmaf_rn(c[0],d[0],u);
            #pragma unroll
            for(int j=2;j<FC;++j)u=__fmaf_rn(c[j],d[j],u);
            u=warp_sum(u);z=rndbf(__fmul_rn(silu(rndbf(z)),rndbf(u)));
            if(lane==0){values[local]=z;y[blockIdx.x*ROWS+local]=__float2bfloat16_rn(z);}
        }else if constexpr(WG==2){
            if(lane==0)values[local*2+warp%2]=z;
        }else{int row=blockIdx.x*ROWS+local;if(lane==0&&row<n)y[row]=__float2bfloat16_rn(z);}
    }
    if constexpr(FUSED){
        __syncthreads();
        if(warp<ROWS/32){
            float z=values[warp*32+lane],maximum=fabsf(z);
            #pragma unroll
            for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
            float scale=fmaxf(__fdiv_rn(maximum,127.f),1e-8f),inv=__fdiv_rn(1.f,scale);
            oq[blockIdx.x*ROWS+warp*32+lane]=(int8_t)__float2int_rn(__fmul_rn(z,inv));
            if(lane==0)os[blockIdx.x*(ROWS/32)+warp]=scale;
        }
    }else if constexpr(WG==2){
        __syncthreads();
        if(tid<ROWS){int row=blockIdx.x*ROWS+tid;if(row<n)y[row]=__float2bfloat16_rn(__fadd_rn(values[tid*2],values[tid*2+1]));}
    }
}

template<int K,bool SHORT,bool FUSED,int ROWS,int WARPS>
int run(void* q,void* xs,void* w,void* s,void* y,void* oq,void* os,int n,void* stream){
    constexpr int BG=K==4096?128:512;
    constexpr int BYTES=(ROWS*BG*(FUSED?4:2)+ROWS*2)*sizeof(float);
    static const cudaError_t configured=cudaFuncSetAttribute(staged_projection<K,SHORT,FUSED,ROWS,WARPS>,cudaFuncAttributeMaxDynamicSharedMemorySize,BYTES);
    if(configured!=cudaSuccess)return (int)configured;
    staged_projection<K,SHORT,FUSED,ROWS,WARPS><<<(n+ROWS-1)/ROWS,WARPS*32,BYTES,(cudaStream_t)stream>>>(
        (int8_t*)q,(float*)xs,(uint32_t*)w,s,(__nv_bfloat16*)y,(int8_t*)oq,(float*)os,n);
    return (int)cudaGetLastError();
}
#define RUN(R,W) if(rows==R&&warps==W)return run<K,SHORT,FUSED,R,W>(q,xs,w,s,y,oq,os,n,stream)
template<int K,bool SHORT,bool FUSED>
int choose_staged(void* q,void* xs,void* w,void* s,void* y,void* oq,void* os,int n,int rows,int warps,void* stream){
    if constexpr(FUSED){RUN(32,4);RUN(32,8);RUN(64,4);RUN(64,8);}
    else{RUN(4,4);RUN(4,8);RUN(8,4);RUN(8,8);RUN(16,4);RUN(16,8);}
    return (int)cudaErrorInvalidValue;
}
#undef RUN
extern "C" int launch_dp4a_staged(void* q,void* xs,void* w,void* s,void* y,void* oq,void* os,
    int n,int k,int short_scale,int fused,int rows,int warps,void* stream){
    if(fused&&k==4096&&short_scale&&n%rows==0)return choose_staged<4096,true,true>(q,xs,w,s,y,oq,os,n,rows,warps,stream);
    if(!fused&&k==4096&&short_scale)return choose_staged<4096,true,false>(q,xs,w,s,y,oq,os,n,rows,warps,stream);
    if(!fused&&k==4096&&!short_scale)return choose_staged<4096,false,false>(q,xs,w,s,y,oq,os,n,rows,warps,stream);
    if(!fused&&k==12288&&!short_scale)return choose_staged<12288,false,false>(q,xs,w,s,y,oq,os,n,rows,warps,stream);
    return (int)cudaErrorInvalidValue;
}
