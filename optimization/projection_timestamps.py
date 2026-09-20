"""Sampled SM90 lane timestamps for selected G32 projection dependency waits.

Diagnostic only: globaltimer is target-specific, and probes perturb scheduling.
The final timestamp records lane-zero store issue, not whole-grid completion.
"""
import types
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from .pdl_control import wait,trigger
from .bulk_address import _hint
from . import dp4a_norm_pdl as norm
from . import dp4a_layout_pdl_prefetch as projection
from .dp4a_norm_pdl import _round,_dot,_projection


@gluon.jit
def _stamp(T,STAGE:gl.constexpr,STRIDE:gl.constexpr):
    block=gl.program_id(0)
    gl.inline_asm_elementwise("""{
        .reg .u32 tid; .reg .pred lane,sample,p; .reg .u64 tick;
        mov.u32 tid,%tid.x;
        setp.eq.u32 lane,tid,0;
        setp.ne.u32 sample,$2,0;
        and.pred p,lane,sample;
        @p mov.u64 tick,%globaltimer;
        @p st.global.u64 [$1],tick;
        mov.u32 $0,0;
    }""",'=r,l,r',[T+block*4+STAGE,(block%STRIDE)==0],dtype=gl.int32,is_pure=False,pack=1)


@gluon.jit
def _norm(X,RES,NW,W,S,SUM,Y,OQ,OS,NY,NQ,NS,N:gl.constexpr,R:gl.constexpr,
            EPS:gl.constexpr,ADD:gl.constexpr,FUSED:gl.constexpr,IG:gl.constexpr,IR:gl.constexpr,
            DEBUG:gl.constexpr,PDL:gl.constexpr,TRIGGER:gl.constexpr,T,STRIDE:gl.constexpr,MODE:gl.constexpr,DIV:gl.constexpr):
    if MODE==2:_stamp(T,0,STRIDE)
    if DIV:_hint(W,S,N,4096,R,FUSED,DIV,0,0,False,0 if FUSED else 1)
    if MODE:_stamp(T,1,STRIDE)
    if PDL:wait()
    if MODE:_stamp(T,2,STRIDE)
    if PDL and TRIGGER==1:trigger()
    # Four warps are fixed because the normalization's FP32 order is part
    # of the tested arithmetic contract.
    V:gl.constexpr=gl.BlockedLayout([8],[32],[4],[0])
    i=gl.arange(0,4096,layout=V)
    x=gl.load(X+i).to(gl.float32)
    if ADD:
        x=(x+gl.load(RES+i).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
        gl.store(SUM+i,x,gl.program_id(0)==0)
    inv=gl.rsqrt(gl.sum(x*x,0)/4096+EPS)
    normalized=((x*inv).to(gl.bfloat16).to(gl.float32)*gl.load(NW+i).to(gl.float32)).to(gl.bfloat16)
    grouped=gl.reshape(normalized.to(gl.float32),(128,32))
    scale=gl.maximum(gl.div_rn(gl.max(gl.abs(grouped),1),127.0),1e-8)
    q=_round(grouped*gl.div_rn(1.0,scale)[:,None])
    if DEBUG:
        gl.store(NY+i,normalized,gl.program_id(0)==0)
        gl.store(NQ+i,gl.reshape(q,(4096,)).to(gl.int8),gl.program_id(0)==0)
        gi=gl.arange(0,128,layout=scale.type.layout)
        gl.store(NS+gi,scale,gl.program_id(0)==0)
    q4=gl.reshape(q,(1024,4))
    c=gl.arange(0,4,layout=gl.SliceLayout(0,q4.type.layout))
    words=gl.sum((q4&255)<<(c[None,:]*8),1)
    a,b=gl.split(gl.reshape(words,(512,2)))
    I:gl.constexpr=gl.BlockedLayout([1,IG,4],[1,32,1],[IR,4//IR,1],[2,1,0])
    F:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    a=gl.convert_layout(gl.reshape(a,(128,4)),gl.SliceLayout(0,I))
    b=gl.convert_layout(gl.reshape(b,(128,4)),gl.SliceLayout(0,I))
    scale=gl.convert_layout(scale,gl.SliceLayout(0,F))
    if PDL and TRIGGER==2:trigger()
    base=gl.program_id(0)*R
    value=_projection(a,b,scale,W,S,base,N*2 if FUSED else N,R,I,F).to(gl.bfloat16)
    if FUSED:
        up=_projection(a,b,scale,W,S,base+N,N*2,R,I,F).to(gl.bfloat16).to(gl.float32)
        gate=value.to(gl.float32)
        silu=(gate/(1+gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value=(silu*up).to(gl.bfloat16)
    if PDL and TRIGGER==3:trigger()
    O:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
    value=gl.convert_layout(value,O)
    rows=base+gl.arange(0,R,layout=O)
    gl.store(Y+rows,value,rows<N)
    if FUSED:
        output=gl.reshape(value.to(gl.float32),(R//32,32))
        output_scale=gl.maximum(gl.div_rn(gl.max(gl.abs(output),1),127.0),1e-8)
        quant=_round(output*gl.div_rn(1.0,output_scale)[:,None])
        gl.store(OQ+rows,gl.reshape(quant,(R,)).to(gl.int8),rows<N)
        groups=gl.program_id(0)*(R//32)+gl.arange(0,R//32,layout=output_scale.type.layout)
        gl.store(OS+groups,output_scale,groups<N//32)

    if MODE==2:_stamp(T,3,STRIDE)


@gluon.jit
def _project(Q,XS,W,S,Y,OQ,OS,N:gl.constexpr,K:gl.constexpr,BG:gl.constexpr,R:gl.constexpr,
            FUSED:gl.constexpr,WARPS:gl.constexpr,IG:gl.constexpr,IROWS:gl.constexpr,PDL:gl.constexpr,TRIGGER:gl.constexpr,PRE:gl.constexpr,T,STRIDE:gl.constexpr,MODE:gl.constexpr,DIV:gl.constexpr):
    if MODE==2:_stamp(T,0,STRIDE)
    if DIV:_hint(W,S,N,K,R,FUSED,DIV,0,0,False,0)
    I:gl.constexpr=gl.BlockedLayout([1,IG,4],[1,32,1],[IROWS,WARPS//IROWS,1],[2,1,0])
    F:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[WARPS if K==4096 else WARPS//2,1 if K==4096 else 2],[1,0])
    base=gl.program_id(0)*R
    gl.static_assert(not FUSED)
    pw=0;ps=0
    if PRE&2:
        ri=base+gl.arange(0,R,layout=gl.SliceLayout(1,gl.SliceLayout(2,I)))
        gi=gl.arange(0,BG,layout=gl.SliceLayout(0,gl.SliceLayout(2,I)))
        c=gl.arange(0,4,layout=gl.SliceLayout(0,gl.SliceLayout(0,I)))
        pos=gi[None,:,None]*4+gl.expand_dims(gl.expand_dims(c,0),0)
        pw=gl.load(gl.cast(W,gl.pointer_type(gl.uint32))+ri[:,None,None]*(K//8)+pos,(ri[:,None,None]<N)&(pos<K//8),0)
    if PRE&1:
        rf=base+gl.arange(0,R,layout=gl.SliceLayout(1,F))
        gf=gl.arange(0,BG,layout=gl.SliceLayout(0,F))
        ps=gl.load(S+rf[:,None]*(K//32)+gf[None,:],(rf[:,None]<N)&(gf[None,:]<K//32),0).to(gl.float32)
    if MODE:_stamp(T,1,STRIDE)
    if PDL:wait()
    if MODE:_stamp(T,2,STRIDE)
    if PDL and TRIGGER==1:trigger()
    value=projection._projection(Q,XS,W,S,base,N,K,BG,R,I,F,pw,ps,PRE).to(gl.bfloat16)
    if FUSED:
        up=projection._projection(Q,XS,W,S,base+N,N*2,K,BG,R,I,F).to(gl.bfloat16).to(gl.float32)
        gate=value.to(gl.float32)
        silu=(gate/(1+gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value=(silu*up).to(gl.bfloat16)
    if PDL and TRIGGER==3:trigger()
    V:gl.constexpr=gl.BlockedLayout([1],[32],[WARPS],[0])
    value=gl.convert_layout(value,V)
    rows=base+gl.arange(0,R,layout=V)
    gl.store(Y+rows,value,rows<N)
    if FUSED:
        grouped=gl.reshape(value.to(gl.float32),(R//32,32))
        scale=gl.maximum(gl.div_rn(gl.max(gl.abs(grouped),1),127.0),1e-8)
        inv=gl.div_rn(1.0,scale)
        product=grouped*inv[:,None]
        quant=gl.inline_asm_elementwise('cvt.rni.s32.f32 $0,$1;',constraints='=r,f',args=[product],dtype=gl.int32,is_pure=True,pack=1)
        gl.store(OQ+rows,gl.reshape(quant,(R,)).to(gl.int8),rows<N)
        groups=gl.program_id(0)*(R//32)+gl.arange(0,R//32,layout=scale.type.layout)
        gl.store(OS+groups,scale,groups<N//32)

    if MODE==2:_stamp(T,3,STRIDE)


class _Dispatch:
    def __init__(self,kernel,trace,stride,mode,divisor):
        self.kernel=kernel;self.trace=trace;self.options=(stride,mode,divisor)
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            if self.trace.shape!=(grid[0],4):raise ValueError('One timestamp row per CTA required')
            return self.kernel[grid](*args,self.trace,*self.options,**kwargs)
        return launch


def configured(kind,trace,*,stride=16,mode=2,divisor=16):
    if kind not in ('norm','projection') or mode not in (0,1,2) or stride not in (1,4,16,64) or divisor not in (0,16):
        raise ValueError('Unsupported diagnostic probe configuration')
    if torch.cuda.get_device_capability()!=(9,0):raise ValueError('Timestamp probe targets SM90')
    if not trace.is_cuda or trace.dtype!=torch.int64 or not trace.is_contiguous() or trace.ndim!=2 or trace.shape[1]!=4:
        raise ValueError('Contiguous CUDA int64 trace required')
    source=norm.linear if kind=='norm' else projection.linear
    namespace=dict(source.__globals__)
    namespace['_kernel']=_Dispatch(_norm if kind=='norm' else _project,trace,stride,mode,divisor)
    fn=types.FunctionType(source.__code__,namespace,source.__name__,source.__defaults__,source.__closure__)
    fn.__kwdefaults__=dict(source.__kwdefaults__)
    return fn
