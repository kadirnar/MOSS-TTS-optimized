"""Experimental single-CTA norm/quant producer plus overlapping projections."""
import types

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from . import dp4a_norm_pdl as original
from .bulk_address import _hint
from .pdl_control import wait,trigger


@gluon.jit
def _normalize(X,RES,NW,SUM,Q,XS,NY,NQ,NS,EPS:gl.constexpr,ADD:gl.constexpr,
               DEBUG:gl.constexpr,TRIGGER:gl.constexpr):
    wait()
    if TRIGGER==1:trigger()
    V:gl.constexpr=gl.BlockedLayout([8],[32],[4],[0])
    i=gl.arange(0,4096,layout=V);x=gl.load(X+i).to(gl.float32)
    if ADD:
        x=(x+gl.load(RES+i).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
        gl.store(SUM+i,x)
    inv=gl.rsqrt(gl.sum(x*x,0)/4096+EPS)
    normalized=((x*inv).to(gl.bfloat16).to(gl.float32)*gl.load(NW+i).to(gl.float32)).to(gl.bfloat16)
    grouped=gl.reshape(normalized.to(gl.float32),(128,32))
    scale=gl.maximum(gl.div_rn(gl.max(gl.abs(grouped),1),127.0),1e-8)
    quant=original._round(grouped*gl.div_rn(1.0,scale)[:,None])
    if TRIGGER==2:trigger()
    gi=gl.arange(0,128,layout=scale.type.layout)
    gl.store(Q+i,gl.reshape(quant,(4096,)).to(gl.int8));gl.store(XS+gi,scale)
    if DEBUG:
        gl.store(NY+i,normalized);gl.store(NQ+i,gl.reshape(quant,(4096,)).to(gl.int8));gl.store(NS+gi,scale)
    if TRIGGER==3:trigger()


@gluon.jit
def _project(Q,XS,W,S,Y,OQ,OS,N:gl.constexpr,R:gl.constexpr,FUSED:gl.constexpr,
             IG:gl.constexpr,IR:gl.constexpr,TRIGGER:gl.constexpr,DIV:gl.constexpr):
    if DIV:_hint(W,S,N,4096,R,FUSED,DIV,0,0,False,0 if FUSED else 1)
    wait()
    if TRIGGER==1:trigger()
    I:gl.constexpr=gl.BlockedLayout([1,IG,4],[1,32,1],[IR,4//IR,1],[2,1,0])
    F:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    A:gl.constexpr=gl.SliceLayout(0,I)
    g=gl.arange(0,128,layout=gl.SliceLayout(1,A));c=gl.arange(0,4,layout=gl.SliceLayout(0,A))
    words=gl.cast(Q,gl.pointer_type(gl.uint32))+g[:,None]*8+c[None,:]*2
    a=gl.load(words);b=gl.load(words+1)
    gf=gl.arange(0,128,layout=gl.SliceLayout(0,F));scale=gl.load(XS+gf)
    base=gl.program_id(0)*R
    value=original._projection(a,b,scale,W,S,base,N*2 if FUSED else N,R,I,F).to(gl.bfloat16)
    if FUSED:
        up=original._projection(a,b,scale,W,S,base+N,N*2,R,I,F).to(gl.bfloat16).to(gl.float32)
        gate=value.to(gl.float32);silu=(gate/(1+gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value=(silu*up).to(gl.bfloat16)
    if TRIGGER==3:trigger()
    O:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);value=gl.convert_layout(value,O)
    rows=base+gl.arange(0,R,layout=O);gl.store(Y+rows,value,rows<N)
    if FUSED:
        output=gl.reshape(value.to(gl.float32),(R//32,32))
        output_scale=gl.maximum(gl.div_rn(gl.max(gl.abs(output),1),127.0),1e-8)
        quant=original._round(output*gl.div_rn(1.0,output_scale)[:,None])
        gl.store(OQ+rows,gl.reshape(quant,(R,)).to(gl.int8),rows<N)
        groups=gl.program_id(0)*(R//32)+gl.arange(0,R//32,layout=output_scale.type.layout)
        gl.store(OS+groups,output_scale,groups<N//32)


class _Dispatch:
    def __init__(self,norm_trigger,divisor,registers):
        self.norm_trigger=norm_trigger;self.divisor=divisor;self.registers=registers;self.last_normalizer=None

    def __getitem__(self,grid):
        def launch(x,res,nw,w,s,summed,y,oq,os,ny,nq,ns,n,rows,eps,add,fused,ig,ir,debug,pdl,trigger_mode,**kwargs):
            if not pdl or trigger_mode not in (1,3):raise ValueError('Qualified PDL trigger required')
            if torch.cuda.get_device_capability(x.device)!=(9,0):raise ValueError('SM90 experiment only')
            if any(t.data_ptr()%16 for t in (w,s)):raise ValueError('Aligned bulk-prefetch buffers required')
            q=torch.empty(4096,device=x.device,dtype=torch.int8);sx=torch.empty(128,device=x.device)
            self.last_normalizer=_normalize[(1,)](x,res,nw,summed,q,sx,ny,nq,ns,eps,add,debug,self.norm_trigger,num_warps=4,launch_pdl=True)
            options={} if self.registers is None else {'maxnreg':self.registers}
            return _project[grid](q,sx,w,s,y,oq,os,n,rows,fused,ig,ir,trigger_mode,self.divisor,num_warps=4,launch_pdl=True,**options)
        return launch


def configured(*,norm_trigger=1,divisor=16,registers=None):
    if norm_trigger not in (1,2,3) or divisor not in (0,16):raise ValueError('Unsupported split-normalization configuration')
    if registers not in (None,128,144,160,168,176,192):raise ValueError('Unsupported register cap')
    dispatch=_Dispatch(norm_trigger,divisor,registers);namespace=dict(original.linear.__globals__);namespace['_kernel']=dispatch
    fn=types.FunctionType(original.linear.__code__,namespace,original.linear.__name__,original.linear.__defaults__,original.linear.__closure__)
    fn.__kwdefaults__=dict(original.linear.__kwdefaults__);fn.dispatch=dispatch
    return fn
