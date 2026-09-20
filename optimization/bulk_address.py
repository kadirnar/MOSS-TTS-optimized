"""Experimental address specialization for Hopper bulk L2 hints.

Only immutable weight/scale buffers may be prefetched. Hints are not producer
synchronization: the original body still waits before reading activations.
The original arithmetic JIT bodies and host validation are reused unchanged.
"""
import types

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from . import dp4a_norm_pdl as norm
from . import dp4a_layout_pdl_prefetch as projection


@gluon.jit
def _bulk(pointer,size,POLICY:gl.constexpr):
    if POLICY==0:
        gl.inline_asm_elementwise("""{
            .reg .b32 tid; .reg .pred p;
            mov.u32 tid,%tid.x;
            setp.eq.u32 p,tid,0;
            @p cp.async.bulk.prefetch.L2.global [$1],$2;
            mov.u32 $0,0;
        }""",'=r,l,r',[pointer,size.to(gl.int32)],dtype=gl.int32,is_pure=False,pack=1)
    else:
        gl.inline_asm_elementwise("""{
            .reg .b32 tid; .reg .pred p; .reg .b64 policy;
            mov.u32 tid,%tid.x;
            setp.eq.u32 p,tid,0;
            createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 policy,1.0;
            @p cp.async.bulk.prefetch.L2.global.L2::cache_hint [$1],$2,policy;
            mov.u32 $0,0;
        }""",'=r,l,r',[pointer,size.to(gl.int32)],dtype=gl.int32,is_pure=False,pack=1)


@gluon.jit
def _hint(W,S,N:gl.constexpr,K:gl.constexpr,R:gl.constexpr,FUSED:gl.constexpr,
          DIV:gl.constexpr,AHEAD:gl.constexpr,POLICY:gl.constexpr,SCALES:gl.constexpr,MODE:gl.constexpr):
    block=gl.program_id(0)
    if AHEAD or MODE==0:block=(block+AHEAD)%gl.num_programs(0)
    row=block*R
    if MODE==2 and N%R==0:size=gl.program_id(0)*0+R*K//2//DIV
    else:
        size=gl.minimum(R,N-row)*(K//2)
        size=gl.minimum(size,R*K//2//DIV)
    _bulk(gl.cast(W,gl.pointer_type(gl.uint8))+row*(K//2),size,POLICY)
    if FUSED:_bulk(gl.cast(W,gl.pointer_type(gl.uint8))+(row+N)*(K//2),size,POLICY)
@gluon.jit
def _norm(X,RES,NW,W,S,SUM,Y,OQ,OS,NY,NQ,NS,N:gl.constexpr,R:gl.constexpr,
          EPS:gl.constexpr,ADD:gl.constexpr,FUSED:gl.constexpr,IG:gl.constexpr,IR:gl.constexpr,
          DEBUG:gl.constexpr,PDL:gl.constexpr,TRIGGER:gl.constexpr,
          DIV:gl.constexpr,AHEAD:gl.constexpr,POLICY:gl.constexpr,SCALES:gl.constexpr,MODE:gl.constexpr):
    if DIV:
        _hint(W,S,N,4096,R,FUSED,DIV,AHEAD,POLICY,False,MODE)
        if SCALES:
            block=gl.program_id(0)
            if AHEAD or MODE==0:block=(block+AHEAD)%gl.num_programs(0)
            row=block*R
            count=gl.minimum(R,N-row)*128
            _bulk(S+row*128,count*(S.dtype.element_ty.primitive_bitwidth//8),POLICY)
            if FUSED:_bulk(S+(row+N)*128,count*(S.dtype.element_ty.primitive_bitwidth//8),POLICY)
    norm._kernel(X,RES,NW,W,S,SUM,Y,OQ,OS,NY,NQ,NS,N,R,EPS,ADD,FUSED,IG,IR,DEBUG,PDL,TRIGGER)


@gluon.jit
def _project(Q,XS,W,S,Y,OQ,OS,N:gl.constexpr,K:gl.constexpr,BG:gl.constexpr,R:gl.constexpr,
             FUSED:gl.constexpr,WARPS:gl.constexpr,IG:gl.constexpr,IROWS:gl.constexpr,
             PDL:gl.constexpr,TRIGGER:gl.constexpr,PRE:gl.constexpr,
             DIV:gl.constexpr,AHEAD:gl.constexpr,POLICY:gl.constexpr,SCALES:gl.constexpr,MODE:gl.constexpr):
    if DIV:
        _hint(W,S,N,K,R,False,DIV,AHEAD,POLICY,False,MODE)
        if SCALES:
            block=gl.program_id(0)
            if AHEAD or MODE==0:block=(block+AHEAD)%gl.num_programs(0)
            row=block*R
            count=gl.minimum(R,N-row)*(K//32)
            _bulk(S+row*(K//32),count*(S.dtype.element_ty.primitive_bitwidth//8),POLICY)
    projection._kernel(Q,XS,W,S,Y,OQ,OS,N,K,BG,R,FUSED,WARPS,IG,IROWS,PDL,TRIGGER,PRE)


class _Dispatch:
    def __init__(self,kernel,options):self.kernel=kernel;self.options=options

    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            return self.kernel[grid](*args,*self.options,**kwargs)
        return launch


def configured(kind,*,divisor=16,ahead=0,policy=0,scales=False,address_mode=1):
    """Copy the existing host callable with only its JIT dispatch replaced.

    This preserves all original input checks and allocations without mutating
    the selected module globals. Multiple variants can coexist in one process.
    """
    if kind not in ('norm','projection') or divisor not in (0,1,2,4,8,16,32,64) or ahead not in (0,1,4,16,64) or policy not in (0,1) or address_mode not in (0,1,2):
        raise ValueError('Unsupported bulk-prefetch configuration')
    if torch.cuda.get_device_capability()!=(9,0):raise ValueError('This experiment targets SM90')
    source=norm.linear if kind=='norm' else projection.linear
    namespace=dict(source.__globals__)
    namespace['_kernel']=_Dispatch(_norm if kind=='norm' else _project,(divisor,ahead,policy,scales,address_mode))
    fn=types.FunctionType(source.__code__,namespace,source.__name__,source.__defaults__,source.__closure__)
    fn.__kwdefaults__=dict(source.__kwdefaults__)
    def checked(*args,**kwargs):
        wi,si=(4,5) if kind=='norm' else (1,2)
        w=args[wi] if len(args)>wi else kwargs['w']
        s=args[si] if len(args)>si else kwargs['s']
        if divisor and (w.data_ptr()%16 or scales and s.data_ptr()%16):
            raise ValueError('Bulk prefetch requires 16-byte aligned weight/scale addresses')
        return fn(*args,**kwargs)
    return checked
