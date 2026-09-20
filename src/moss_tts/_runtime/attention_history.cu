// Experimental historical-KV preloads. No current-token read precedes the wait.
// Numerical helpers and the unchanged reference remain in the included source.
#include "attention_pdl.cu"

__device__ __forceinline__ uint4 historical_pack(const __nv_bfloat16* p,bool valid){
    uint4 result=make_uint4(0,0,0,0);
    if(valid)asm volatile("ld.global.v4.u32 {%0,%1,%2,%3},[%4];"
        :"=r"(result.x),"=r"(result.y),"=r"(result.z),"=r"(result.w):"l"(p):"memory");
    return result;
}
__device__ __forceinline__ void unpack_history(uint4 packed,float (&x)[8]){
    unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
    #pragma unroll
    for(int j=0;j<4;++j){
        x[2*j]=__bfloat162float(__ushort_as_bfloat16(words[j]&65535));
        x[2*j+1]=__bfloat162float(__ushort_as_bfloat16(words[j]>>16));
    }
}

template<int PRE,bool PACKED,int TRIGGER>
__global__ void attention_history(const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,const __nv_bfloat16* __restrict__ v,
    const int64_t* __restrict__ position,float* __restrict__ partial,
    float* __restrict__ lse,int length,int splits){
    const int head=blockIdx.x,split=blockIdx.y,tid=threadIdx.x;
    const int lane=tid%32,warp=tid/32,d=(lane%16)*8;
    // Caller guarantees position and historical t<p cache rows are immutable
    // before the preceding QKV producer can trigger us. Current t==p is loaded
    // only after the ordinary grid dependency wait below.
    const int p=*position;
    uint4 pk[4],pv[4];float ek[4][8],ev[4][8];
    #pragma unroll
    for(int r=0;r<4;++r){
        const int t=split*32+warp*2+lane/16+r*8;
        if constexpr(PRE&1){
            if constexpr(PACKED)pk[r]=historical_pack(k+((head/4)*length+t)*128+d,t<p);
            else unpack_history(historical_pack(k+((head/4)*length+t)*128+d,t<p),ek[r]);
        }
        if constexpr(PRE&2){
            if constexpr(PACKED)pv[r]=historical_pack(v+((head/4)*length+t)*128+d,t<p);
            else unpack_history(historical_pack(v+((head/4)*length+t)*128+d,t<p),ev[r]);
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if constexpr(TRIGGER==1)cudaTriggerProgrammaticLaunchCompletion();
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
        if constexpr(PRE&1){
            if(t==p)load8(k+((head/4)*length+t)*128+d,kk);
            else if constexpr(PACKED)unpack_history(pk[r],kk);
            else {
                #pragma unroll
                for(int j=0;j<8;++j)kk[j]=ek[r][j];
            }
        }else load8(k+((head/4)*length+t)*128+d,kk,t<=p);
        float z=mul(qq[1],kk[1]);
        z=__fmaf_rn(qq[0],kk[0],z);
        #pragma unroll
        for(int j=2;j<8;++j)z=__fmaf_rn(qq[j],kk[j],z);
        #pragma unroll
        for(int offset=8;offset>0;offset/=2)z=add(z,__shfl_xor_sync(0xffffffff,z,offset));
        logits[r]=t<=p?mul(z,0x1.6a09e6p-4f):-INFINITY;
    }
    if constexpr(TRIGGER==2)cudaTriggerProgrammaticLaunchCompletion();
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
        if constexpr(PRE&2){
            if(t==p)load8(v+((head/4)*length+t)*128+d,vv[r]);
            else if constexpr(PACKED)unpack_history(pv[r],vv[r]);
            else {
                #pragma unroll
                for(int j=0;j<8;++j)vv[r][j]=ev[r][j];
            }
        }else load8(v+((head/4)*length+t)*128+d,vv[r],t<=p);
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
    if constexpr(TRIGGER==3)cudaTriggerProgrammaticLaunchCompletion();
    partial[(head*splits+split)*128+tid]=div_full(four_sum(&weighted[0][tid],128),denominator);
    if(tid==0)lse[head*splits+split]=add(maximum,logf(denominator));
}

template<int PRE,bool PACKED,int TRIGGER>
int run_history(void* q,void* k,void* v,void* position,void* partial,void* lse,int length,int splits,void* stream){
    cudaLaunchConfig_t config{};
    config.gridDim=dim3(32,splits);config.blockDim=dim3(128);config.stream=(cudaStream_t)stream;
    cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed=1;config.attrs=&attribute;config.numAttrs=1;
    return (int)cudaLaunchKernelEx(&config,attention_history<PRE,PACKED,TRIGGER>,
        (const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,
        (const int64_t*)position,(float*)partial,(float*)lse,length,splits);
}
template<int PRE,bool PACKED>
int history_select(void* q,void* k,void* v,void* position,void* partial,void* lse,int length,int splits,int trigger,void* stream){
    if(trigger==1)return run_history<PRE,PACKED,1>(q,k,v,position,partial,lse,length,splits,stream);
    if(trigger==2)return run_history<PRE,PACKED,2>(q,k,v,position,partial,lse,length,splits,stream);
    return (int)cudaErrorInvalidValue;
}
extern "C" int launch_attention_history(void* q,void* k,void* v,void* position,void* partial,void* lse,int length,int splits,int mode,int packed,int trigger,void* stream){
    #define SELECT(P,B) return history_select<P,B>(q,k,v,position,partial,lse,length,splits,trigger,stream)
    if(mode==0){SELECT(0,true);}
    if(mode==1){if(packed){SELECT(1,true);}else{SELECT(1,false);}}
    if(mode==2){if(packed){SELECT(2,true);}else{SELECT(2,false);}}
    if(mode==3){if(packed){SELECT(3,true);}else{SELECT(3,false);}}
    return (int)cudaErrorInvalidValue;
    #undef SELECT
}
