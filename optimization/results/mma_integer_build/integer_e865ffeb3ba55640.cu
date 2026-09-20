#include <cuda_runtime.h>
#include <stdint.h>

// Diagonal eight-group tile; the second A row block supplies a second
// independent weight row against the same activation groups.
__global__ void integer_groups(const uint2* w,const uint32_t* q,int* y,int groups){
    int lane=threadIdx.x,group=blockIdx.x*8+lane/4,c=lane%4;
    uint2 p=__ldg(w+blockIdx.x*32+lane);
    uint32_t a0=(p.x<<4)&0xf0f0f0f0u,a1=p.x&0xf0f0f0f0u;
    uint32_t a2=(p.y<<4)&0xf0f0f0f0u,a3=p.y&0xf0f0f0f0u;
    uint32_t b0=__ldg(q+group*8+c),b1=__ldg(q+group*8+c+4);
    int d0,d1,d2,d3;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{0,0,0,0};"
        :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    if(c==(lane/4)/2){
        y[group]=((lane/4)%2?d1:d0)>>4;
        y[groups+group]=((lane/4)%2?d3:d2)>>4;
    }
}
extern "C" int launch_integer_groups(void* w,void* q,void* y,int groups,void* stream){
    if(groups<=0||groups%8)return (int)cudaErrorInvalidValue;
    integer_groups<<<groups/8,32,0,(cudaStream_t)stream>>>((uint2*)w,(uint32_t*)q,(int*)y,groups);
    return (int)cudaGetLastError();
}
