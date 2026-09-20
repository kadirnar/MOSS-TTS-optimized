"""Experimental unchanged-value G32 weight staging before PDL waits.

Compare cooperative cp.async with Hopper TMA. Only immutable packed weights
and scales are read before the dependency wait; activations remain protected.
"""
import types

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import async_copy,tma,mbarrier
from .pdl_control import wait,trigger
from .bulk_address import _hint
from . import dp4a_layout_pdl_prefetch as original


@gluon.jit
def _append_groups(a,b,R:gl.constexpr,G:gl.constexpr):
    return gl.reshape(gl.permute(gl.join(a,b),(0,3,1,2)),(R,G*2,4))


@gluon.jit
def _kernel(Q,XS,W,S,Y,OQ,OS,N:gl.constexpr,K:gl.constexpr,BG:gl.constexpr,R:gl.constexpr,
            FUSED:gl.constexpr,WARPS:gl.constexpr,IG:gl.constexpr,IR:gl.constexpr,
            PDL:gl.constexpr,TRIGGER:gl.constexpr,PRE:gl.constexpr,DESC,
            MODE:gl.constexpr,SWIZZLE:gl.constexpr,DIV:gl.constexpr,CACHE:gl.constexpr):
    gl.static_assert(not FUSED and PRE==1)
    I:gl.constexpr=gl.BlockedLayout([1,IG,4],[1,32,1],[IR,WARPS//IR,1],[2,1,0])
    F:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[WARPS if K==4096 else WARPS//2,1 if K==4096 else 2],[1,0])
    base=gl.program_id(0)*R
    if DIV:_hint(W,S,N,K,R,False,DIV,0,0,False,0)
    if MODE==1:
        ri=base+gl.arange(0,R,layout=gl.SliceLayout(1,gl.SliceLayout(2,I)))
        gi=gl.arange(0,BG,layout=gl.SliceLayout(0,gl.SliceLayout(2,I)))
        c=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(0,I)))
        pos=gi[None,:,None]*4+gl.expand_dims(gl.expand_dims(c,0),0)
        memory=gl.allocate_shared_memory(gl.uint32,[R,BG,4],gl.SwizzledSharedLayout(4,1,SWIZZLE,[2,1,0]))
        async_copy.async_copy_global_to_shared(memory,gl.cast(W,gl.pointer_type(gl.uint32))+ri[:,None,None]*(K//8)+pos,
            mask=(ri[:,None,None]<N)&(pos<K//8),cache_modifier='.cg' if CACHE else '.ca')
        async_copy.commit_group()
    else:
        CHUNKS:gl.constexpr=3 if MODE==3 and K==12288 else 1
        memory=gl.allocate_shared_memory(DESC.dtype,[CHUNKS]+DESC.block_shape,DESC.layout)
        ready=mbarrier.allocate_mbarrier();mbarrier.init(ready,count=1)
        mbarrier.expect(ready,CHUNKS*DESC.block_type.nbytes)
        for part in gl.static_range(CHUNKS):
            tma.async_copy_global_to_shared(DESC,[base,part*4,0],ready,memory.index(part))
    rf=base+gl.arange(0,R,layout=gl.SliceLayout(1,F))
    gf=gl.arange(0,BG,layout=gl.SliceLayout(0,F))
    scales=gl.load(S+rf[:,None]*(K//32)+gf[None,:],(rf[:,None]<N)&(gf[None,:]<K//32),0).to(gl.float32)
    if PDL:wait()
    if PDL and TRIGGER==1:trigger()
    if MODE==1:
        async_copy.wait_group(0)
        packed=memory.load(I)
    else:
        mbarrier.wait(ready,phase=0);mbarrier.invalidate(ready)
        L:gl.constexpr=gl.BlockedLayout([1,1,4],[1,1,32],[IR,WARPS//IR,1],[2,1,0])
        if MODE==2 or K==4096:
            packed=gl.convert_layout(gl.reshape(memory.index(0).load(L),(R,BG,4)),I).to(gl.uint32)
        else:
            w0=gl.convert_layout(gl.reshape(memory.index(0).load(L),(R,128,4)),I).to(gl.uint32)
            zero=gl.full((R,128,4),0,gl.uint32,I)
            if MODE==3:
                w1=gl.convert_layout(gl.reshape(memory.index(1).load(L),(R,128,4)),I).to(gl.uint32)
                w2=gl.convert_layout(gl.reshape(memory.index(2).load(L),(R,128,4)),I).to(gl.uint32)
                packed=gl.convert_layout(_append_groups(_append_groups(w0,w1,R,128),_append_groups(w2,zero,R,128),R,256),I)
            else:
                packed=gl.convert_layout(_append_groups(_append_groups(w0,zero,R,128),_append_groups(zero,zero,R,128),R,256),I)
                ri=base+gl.arange(0,R,layout=gl.SliceLayout(1,gl.SliceLayout(2,I)))
                gi=gl.arange(0,BG,layout=gl.SliceLayout(0,gl.SliceLayout(2,I)))
                c=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(0,I)))
                pos=gi[None,:,None]*4+gl.expand_dims(gl.expand_dims(c,0),0)
                rest=gl.load(gl.cast(W,gl.pointer_type(gl.uint32))+ri[:,None,None]*(K//8)+pos,
                    (ri[:,None,None]<N)&(pos>=512)&(pos<K//8),0)
                packed=gl.where(pos<512,packed,rest)
    value=original._projection(Q,XS,W,S,base,N,K,BG,R,I,F,packed,scales,3).to(gl.bfloat16)
    if PDL and TRIGGER==3:trigger()
    O:gl.constexpr=gl.BlockedLayout([1],[32],[WARPS],[0])
    value=gl.convert_layout(value,O);rows=base+gl.arange(0,R,layout=O)
    gl.store(Y+rows,value,rows<N)


class _Dispatch:
    def __init__(self,mode,swizzle,divisor,cache):
        self.options=(mode,swizzle,divisor,cache);self.descriptors={}
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            w=args[2];n,k,bg,rows=args[7:11];descriptor=None
            if self.options[0]>=2:
                key=(w.data_ptr(),n,k,rows)
                if key not in self.descriptors:
                    tensor=w.view(torch.int32).reshape(n,k//1024,128)
                    layout=gl.NVMMASharedLayout(self.options[1],32,rank=3)
                    self.descriptors[key]=TensorDescriptor.from_tensor(tensor,[rows,bg//32 if self.options[0]==2 else 4,128],layout)
                descriptor=self.descriptors[key]
            return _kernel[grid](*args,descriptor,*self.options,**kwargs)
        return launch


def configured(*,mode=1,swizzle=1,divisor=0,cache=0):
    if mode not in (1,2,3,4) or divisor not in (0,16) or cache not in (0,1):raise ValueError('Unsupported staging mode')
    if swizzle not in ((1,2,4,8) if mode==1 else (0,32,64,128)):raise ValueError('Unsupported shared-memory layout')
    if torch.cuda.get_device_capability()!=(9,0):raise ValueError('Experiment targets SM90')
    source=original.linear;namespace=dict(source.__globals__)
    namespace['_kernel']=_Dispatch(mode,swizzle,divisor,cache)
    fn=types.FunctionType(source.__code__,namespace,source.__name__,source.__defaults__,source.__closure__)
    fn.__kwdefaults__=dict(source.__kwdefaults__)
    return fn
