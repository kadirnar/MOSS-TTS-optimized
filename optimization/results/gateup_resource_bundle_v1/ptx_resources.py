"""Experimental SM90 PTX reassembly with static shared memory and register caps.

The selected Triton PTX instructions remain unchanged. Only the PTX version,
shared-memory declaration, optional register cap and spilling pragma change.
This isolated driver launcher is not installed in the selected model path.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess

import torch

from .common import RESULTS


class AttributeValue(C.Union):
    _fields_ = [('pad', C.c_char * 64), ('serialization', C.c_int), ('alignment', C.c_uint64)]


class Attribute(C.Structure):
    _fields_ = [('id', C.c_int), ('pad', C.c_char * 4), ('value', AttributeValue)]


class LaunchConfig(C.Structure):
    _fields_ = [(name, C.c_uint) for name in ('gx','gy','gz','bx','by','bz','shared')]+[
        ('stream', C.c_void_p), ('attrs', C.POINTER(Attribute)), ('count', C.c_uint)]


def check(result, operation):
    if result: raise RuntimeError(f'{operation}: CUDA driver error {result}')


class ReassembledKernel:
    def __init__(self, compiled, *, compiler='/workspace/cuda-13.0-ptxas/ptxas',
                 registers=None, shared_spilling=False, static_shared=True):
        meta=compiled.metadata
        if C.sizeof(Attribute)!=72 or C.sizeof(LaunchConfig)!=56:
            raise RuntimeError('This driver launcher requires the 64-bit CUDA ABI')
        if meta.num_ctas!=1 or meta.global_scratch_size or meta.profile_scratch_size:
            raise ValueError('Only single-CTA kernels without global scratch are supported')
        if torch.cuda.get_device_capability()!=(9,0):raise ValueError('SM90 experiment only')
        if registers is not None and registers not in (32,40,48,56,64,80,96,112,128,144,160,192):
            raise ValueError('Unsupported register budget')
        if shared_spilling and not static_shared:raise ValueError('Shared spills require static shared memory')
        self.pointer_names=[k for k,v in compiled.src.signature.items() if v.startswith('*')]
        if any(v!='constexpr' and not v.startswith('*') for v in compiled.src.signature.values()):
            raise ValueError('Only pointer runtime arguments supported')
        ptx=compiled.asm['ptx'];original=ptx
        ptx,n=re.subn(r'\.version \d+\.\d+', '.version 9.0', ptx,count=1);assert n==1
        if static_shared:
            ptx,n=re.subn(r'\.extern \.shared \.align 16 \.b8 global_smem\[\];',
                         f'.shared .align 16 .b8 global_smem[{meta.shared}];',ptx,count=1)
            assert n==1 and meta.shared>0
        if registers is not None:
            ptx,n=re.subn(r'(\.reqntid [^\n]+\n)',rf'\1.maxnreg {registers}\n',ptx,count=1);assert n==1
        if shared_spilling:
            ptx,n=re.subn(r'(\.reqntid [^\n]+\n(?:\.maxnreg \d+\n)?\{)',
                         r'\1\n.pragma "enable_smem_spilling";',ptx,count=1);assert n==1
        params=re.search(r'\.visible \.entry [^(]+\((.*?)\)\s*\.reqntid',ptx,re.S)
        assert params and len(re.findall(r'\.param\b',params[1]))==len(self.pointer_names)+2
        self.threads=meta.num_warps*32;self.dynamic_shared=0 if static_shared else meta.shared
        self.pdl=meta.launch_pdl;self.name=meta.name
        compiler=Path(compiler);compiler_hash=hashlib.sha256(compiler.read_bytes()).hexdigest()
        key=hashlib.sha256((ptx+compiler_hash).encode()).hexdigest()[:24]
        folder=RESULTS/'ptx_resource_build';folder.mkdir(exist_ok=True)
        source=folder/(key+'.ptx');cubin=folder/(key+'.cubin');log=folder/(key+'.log')
        if not cubin.exists():
            source.write_text(ptx);(folder/(key+'.original.ptx')).write_text(original)
            command=[str(compiler),'-arch=sm_90a','-O3','-v',str(source),'-o',str(cubin)]
            result=subprocess.run(command,text=True,capture_output=True)
            log.write_text(result.stdout+result.stderr);result.check_returncode()
        self.lib=C.CDLL('libcuda.so.1');self.module=C.c_void_p();self.function=C.c_void_p()
        self.lib.cuModuleLoad.argtypes=[C.POINTER(C.c_void_p),C.c_char_p]
        self.lib.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
        self.lib.cuFuncGetAttribute.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p]
        self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes=[C.POINTER(C.c_int),C.c_void_p,C.c_int,C.c_size_t]
        self.lib.cuLaunchKernelEx.argtypes=[C.POINTER(LaunchConfig),C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
        check(self.lib.cuModuleLoad(C.byref(self.module),str(cubin).encode()),'module load')
        check(self.lib.cuModuleGetFunction(C.byref(self.function),self.module,self.name.encode()),'function lookup')
        attrs={}
        for name,code in [('static_shared_bytes',1),('local_bytes_per_thread',3),('registers',4)]:
            value=C.c_int();check(self.lib.cuFuncGetAttribute(C.byref(value),code,self.function),'function attribute');attrs[name]=value.value
        blocks=C.c_int();check(self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(blocks),self.function,self.threads,self.dynamic_shared),'occupancy')
        self.resources={**attrs,'dynamic_shared_bytes':self.dynamic_shared,'max_active_blocks_per_sm':blocks.value,
                        'threads':self.threads,'build':key,'register_cap':registers,'shared_spilling':shared_spilling,
                        'static_shared':static_shared,'compiler':str(compiler),'compiler_sha256':compiler_hash,
                        'original_ptx_sha256':hashlib.sha256(original.encode()).hexdigest(),
                        'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
                        'cubin_sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()}
        (folder/(key+'.json')).write_text(json.dumps(self.resources,indent=2)+'\n')

    def __call__(self, arguments, grid):
        tensors=[arguments[name] for name in self.pointer_names]
        if any(not t.is_cuda or t.device!=tensors[0].device or not t.is_contiguous() for t in tensors):
            raise ValueError('Contiguous same-device CUDA arguments required')
        values=[C.c_uint64(t.data_ptr()) for t in tensors]+[C.c_uint64(0),C.c_uint64(0)]
        pointers=(C.c_void_p*len(values))(*[C.cast(C.pointer(v),C.c_void_p) for v in values])
        attribute=Attribute();attribute.id=6;attribute.value.serialization=1
        config=LaunchConfig(grid,1,1,self.threads,1,1,self.dynamic_shared,
                            torch.cuda.current_stream(tensors[0].device).cuda_stream,
                            C.pointer(attribute) if self.pdl else None,1 if self.pdl else 0)
        check(self.lib.cuLaunchKernelEx(C.byref(config),self.function,pointers,None),'extended kernel launch')


class NormProjection:
    def __init__(self, entry, name, **config):
        from .dp4a_norm_pdl import linear,SELECTED
        self.fused=name=='up';d=entry['raw'] if self.fused else entry['next']
        self.eps=d['eps'];self.add=d['has_residual'];self.rows=SELECTED[name]['rows']
        x=d['x'][:1];res=d['residual'][:1] if self.add else None
        self.x_shape=x.shape;self.weight_shape=entry['weights'][name][0].shape
        _,compiled=linear(x,res,d['weight'],self.eps,*entry['weights'][name],fused=self.fused,
                          **SELECTED[name],return_kernel=True)
        self.kernel=ReassembledKernel(compiled,**config)

    def __call__(self,x,res,nw,eps,w,s):
        if eps!=self.eps or (res is not None)!=self.add:raise ValueError('Different normalization specialization')
        if x.shape!=self.x_shape or x.dtype!=torch.bfloat16 or w.shape!=self.weight_shape or w.dtype!=torch.uint8 or s.shape!=(w.shape[0],128) or s.dtype!=torch.bfloat16 or nw.shape!=(4096,) or nw.dtype!=x.dtype:
            raise ValueError('Different shape or dtype from the compiled normalization kernel')
        if res is not None and (res.shape!=x.shape or res.dtype!=x.dtype):raise ValueError('Invalid residual')
        n=w.shape[0]//2 if self.fused else w.shape[0]
        summed=torch.empty_like(x) if res is not None else x
        y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
        oq=torch.empty(n if self.fused else 0,device=x.device,dtype=torch.int8)
        os=torch.empty(n//32 if self.fused else 0,device=x.device)
        empty=torch.empty(0,device=x.device,dtype=x.dtype)
        args=dict(X=x,RES=res,NW=nw,W=w,S=s,SUM=summed,Y=y,OQ=oq,OS=os,NY=empty,NQ=empty,NS=empty)
        self.kernel(args,(n+self.rows-1)//self.rows)
        return (summed,y,(oq,os)) if self.fused else (summed,y)


class DownProjection:
    def __init__(self,entry,x,q,**config):
        from .dp4a_layout_pdl_prefetch import linear
        from .short_scales import PLAN
        self.x_shape=x.shape;self.weight_shape=entry['weights']['down'][0].shape
        _,compiled=linear(x,*entry['weights']['down'],prequantized=q,**PLAN['down'],
                          trigger_mode=3,prefetch=1,return_kernel=True)
        self.kernel=ReassembledKernel(compiled,**config)

    def __call__(self,x,q,w,s):
        if x.shape!=self.x_shape or x.dtype!=torch.bfloat16 or w.shape!=self.weight_shape or w.dtype!=torch.uint8 or s.shape!=(w.shape[0],x.numel()//32) or s.dtype!=torch.bfloat16 or q[0].numel()!=x.numel() or q[0].dtype!=torch.int8 or q[1].numel()!=x.numel()//32 or q[1].dtype!=torch.float32:
            raise ValueError('Different shape or dtype from the compiled down kernel')
        y=torch.empty((*x.shape[:-1],w.shape[0]),device=x.device,dtype=x.dtype)
        empty=torch.empty(0,device=x.device,dtype=x.dtype)
        self.kernel(dict(Q=q[0],XS=q[1],W=w,S=s,Y=y,OQ=empty,OS=empty),(w.shape[0]+3)//4)
        return y
