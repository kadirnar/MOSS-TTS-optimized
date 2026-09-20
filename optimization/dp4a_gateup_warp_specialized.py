"""Experimental gate/up parallel partitions with exact four-warp normalization."""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, fence_async_shared

from .dp4a_norm_pdl import _round,_projection
from .pdl_control import wait,trigger


@gluon.jit
def _compute(X,RES,NW,W,S,SUM,N:gl.constexpr,EPS:gl.constexpr,PART:gl.constexpr,MODE:gl.constexpr):
    wait()
    V:gl.constexpr=gl.BlockedLayout([8],[32],[4],[0])
    i=gl.arange(0,4096,layout=V)
    x=(gl.load(X+i).to(gl.float32)+gl.load(RES+i).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
    if PART==0:gl.store(SUM+i,x,gl.program_id(0)==0)
    inv=gl.rsqrt(gl.sum(x*x,0)/4096+EPS)
    normalized=((x*inv).to(gl.bfloat16).to(gl.float32)*gl.load(NW+i).to(gl.float32)).to(gl.bfloat16)
    grouped=gl.reshape(normalized.to(gl.float32),(128,32))
    scale=gl.maximum(gl.div_rn(gl.max(gl.abs(grouped),1),127.0),1e-8)
    q=_round(grouped*gl.div_rn(1.0,scale)[:,None])
    q4=gl.reshape(q,(1024,4))
    c=gl.arange(0,4,layout=gl.SliceLayout(0,q4.type.layout))
    words=gl.sum((q4&255)<<(c[None,:]*8),1)
    a,b=gl.split(gl.reshape(words,(512,2)))
    I:gl.constexpr=gl.BlockedLayout([1,2,4],[1,32,1],[4,1,1],[2,1,0])
    F:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    a=gl.convert_layout(gl.reshape(a,(128,4)),gl.SliceLayout(0,I))
    b=gl.convert_layout(gl.reshape(b,(128,4)),gl.SliceLayout(0,I))
    scale=gl.convert_layout(scale,gl.SliceLayout(0,F))
    R:gl.constexpr=32 if MODE==0 else 16
    base=gl.program_id(0)*32+(PART*N if MODE==0 else PART*16)
    value=_projection(a,b,scale,W,S,base,N*2,R,I,F).to(gl.bfloat16)
    if MODE==1:
        up=_projection(a,b,scale,W,S,base+N,N*2,R,I,F).to(gl.bfloat16).to(gl.float32)
        gate=value.to(gl.float32)
        silu=(gate/(1+gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value=(silu*up).to(gl.bfloat16)
    return gl.convert_layout(value,gl.BlockedLayout([1],[32],[4],[0]))


@gluon.jit
def _worker(X,RES,NW,W,S,SUM,Y,OQ,OS,buffer,ready,CONFIG:gl.constexpr):
    N:gl.constexpr=CONFIG[0];EPS:gl.constexpr=CONFIG[1];MODE:gl.constexpr=CONFIG[2]
    value=_compute(X,RES,NW,W,S,SUM,N,EPS,1,MODE)
    buffer.store(value)
    fence_async_shared()
    mbarrier.arrive(ready,count=1)


@gluon.jit
def _default(X,RES,NW,W,S,SUM,Y,OQ,OS,buffer,ready,CONFIG:gl.constexpr):
    N:gl.constexpr=CONFIG[0];EPS:gl.constexpr=CONFIG[1];MODE:gl.constexpr=CONFIG[2]
    value=_compute(X,RES,NW,W,S,SUM,N,EPS,0,MODE)
    mbarrier.wait(ready,phase=0)
    peer=buffer.load(gl.BlockedLayout([1],[32],[4],[0]))
    if MODE==0:
        gate=value.to(gl.float32);up=peer.to(gl.float32)
        silu=(gate/(1+gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value=(silu*up).to(gl.bfloat16)
    else:
        value=gl.join(value,peer)
        value=gl.permute(value,(1,0))
        value=gl.reshape(value,(32,))
        value=gl.convert_layout(value,gl.BlockedLayout([1],[32],[4],[0]))
    rows=gl.program_id(0)*32+gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]))
    gl.store(Y+rows,value)
    f=value.to(gl.float32)
    scale=gl.maximum(gl.div_rn(gl.max(gl.abs(f),0),127.0),1e-8)
    quant=_round(f*gl.div_rn(1.0,scale))
    gl.store(OQ+rows,quant.to(gl.int8));gl.store(OS+gl.program_id(0),scale)


@gluon.jit
def _kernel(X,RES,NW,W,S,SUM,Y,OQ,OS,N:gl.constexpr,EPS:gl.constexpr,MODE:gl.constexpr,WORKER_REGS:gl.constexpr):
    wait();trigger()
    R:gl.constexpr=32 if MODE==0 else 16
    buffer=gl.allocate_shared_memory(gl.bfloat16,[R],gl.SwizzledSharedLayout(1,1,1,[0]))
    ready=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout())
    mbarrier.init(ready,count=1)
    config:gl.constexpr=(N,EPS,MODE)
    args=(X,RES,NW,W,S,SUM,Y,OQ,OS,buffer,ready,config)
    gl.warp_specialize([(_default,args),(_worker,args)],[4],[WORKER_REGS])
    mbarrier.invalidate(ready)


def linear(x,residual,norm_weight,eps,w,s,*,mode=0,max_registers=112,worker_registers=96,return_kernel=False):
    if mode not in (0,1) or max_registers not in (80,96,112,128,160,192) or worker_registers not in (64,80,96,112,128,160):
        raise ValueError('Unsupported partition configuration')
    if x.numel()!=4096 or x.dtype!=torch.bfloat16 or residual is None or residual.shape!=x.shape or residual.dtype!=x.dtype:
        raise ValueError('This gate/up experiment requires a BF16 K4096 row and residual')
    n2,k2=w.shape;n=n2//2
    if k2!=2048 or n%32 or w.dtype!=torch.uint8 or s.shape!=(n2,128) or s.dtype!=torch.bfloat16 or norm_weight.shape!=(4096,) or norm_weight.dtype!=x.dtype:
        raise ValueError('Original G32 gate/up weights and norm required')
    if not x.is_cuda or any(t.device!=x.device or not t.is_contiguous() for t in (x,residual,norm_weight,w,s)):
        raise ValueError('Contiguous same-device CUDA tensors required')
    summed=torch.empty_like(x);y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    oq=torch.empty(n,device=x.device,dtype=torch.int8);os=torch.empty(n//32,device=x.device)
    kernel=_kernel[(n//32,)](x,residual,norm_weight,w,s,summed,y,oq,os,n,eps,mode,worker_registers,
                            num_warps=4,maxnreg=max_registers,launch_pdl=True)
    result=(summed,y,(oq,os))
    return (result,kernel) if return_kernel else result
