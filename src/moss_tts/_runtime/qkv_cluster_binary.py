"""Export isolated compiler cubins and launch them with the selected runtime.

Only this experimental fused kernel is compiled by Triton 3.8. The host model,
attention and all other Triton kernels continue to use the selected runtime.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from .cuda_driver import Attribute,LaunchConfig,check
from .qkv_cluster_prepare import launch


class Binary:
    def __init__(self,folder,record):
        self.record=record;self.metadata=SimpleNamespace(**record['metadata'])
        self.device=torch.cuda.current_device()
        self.n_regs=record['registers'];self.n_spills=record['spills']
        self.asm={'cubin':(folder/(record['prefix']+'.cubin')).read_bytes()}
        assert hashlib.sha256(self.asm['cubin']).hexdigest()==record['cubin_sha256']
        self.lib=C.CDLL('libcuda.so.1');self.module=C.c_void_p();self.function=C.c_void_p()
        self.lib.cuModuleLoad.argtypes=[C.POINTER(C.c_void_p),C.c_char_p]
        self.lib.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
        self.lib.cuLaunchKernelEx.argtypes=[C.POINTER(LaunchConfig),C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
        check(self.lib.cuModuleLoad(C.byref(self.module),str(folder/(record['prefix']+'.cubin')).encode()),'cluster module load')
        check(self.lib.cuModuleGetFunction(C.byref(self.function),self.module,self.metadata.name.encode()),'cluster function lookup')
        assert self.metadata.num_ctas in (1,2,4,8) and self.metadata.num_warps==4
        assert self.metadata.global_scratch_size==self.metadata.profile_scratch_size==0
        assert self.metadata.launch_pdl
        if C.sizeof(Attribute)!=72 or C.sizeof(LaunchConfig)!=56:
            raise RuntimeError('This driver launcher requires the 64-bit CUDA ABI')

    def __call__(self,arguments,grid,eps,head_eps,length):
        record=self.record
        if (eps,head_eps,length)!=(record['eps'],record['head_eps'],record['length']):raise ValueError('Binary specialization mismatch')
        dtypes={'*bf16':torch.bfloat16,'*fp32':torch.float32,'*u8':torch.uint8,'*i64':torch.int64}
        tensors=[]
        for name,kind in record['pointer_signature']:
            t=arguments[name]
            if t is None or not t.is_cuda or not t.is_contiguous() or t.dtype!=dtypes[kind]:raise ValueError('Binary tensor mismatch: '+name)
            tensors.append(t)
        if any(t.device!=tensors[0].device for t in tensors):raise ValueError('Same-device buffers required')
        if tensors[0].device.index!=self.device:raise ValueError('Binary module belongs to another CUDA device')
        values=[C.c_uint64(t.data_ptr()) for t in tensors]+[C.c_uint64(0),C.c_uint64(0)]
        pointers=(C.c_void_p*len(values))(*[C.cast(C.pointer(v),C.c_void_p) for v in values])
        attributes=(Attribute*3)();attributes[0].id=6;attributes[0].value.serialization=1
        dims=tuple(self.metadata.cluster_dims);assert dims[0]*dims[1]*dims[2]==self.metadata.num_ctas
        attributes[1].id=4
        for i,value in enumerate(dims):C.cast(C.byref(attributes[1].value),C.POINTER(C.c_uint))[i]=value
        attributes[2].id=5;attributes[2].value.serialization=1 # CU_CLUSTER_SCHEDULING_POLICY_SPREAD
        cfg=LaunchConfig(grid*dims[0],dims[1],dims[2],128,1,1,self.metadata.shared,
            torch.cuda.current_stream(tensors[0].device).cuda_stream,attributes,3 if self.metadata.num_ctas>1 else 1)
        check(self.lib.cuLaunchKernelEx(C.byref(cfg),self.function,pointers,None),'cluster launch')


def load_bundle(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['format']!='qkv_cluster_cubin_v1' or not manifest['complete']:raise ValueError('Complete cluster bundle required')
    if torch.cuda.get_device_capability()!=(9,0):raise ValueError('SM90 bundle only')
    torch.empty(0,device='cuda') # Establish the primary context before driver module loading.
    kernels={n:Binary(folder,r) for n,r in manifest['binaries'].items()}
    launchers={}
    for name,options in manifest['configs'].items():
        def make(name,options):
            def fn(*args,**kwargs):
                if any(kwargs.get(k)!=v for k,v in options.items()):raise ValueError('Binary configuration differs')
                key=f'{name}_add{int(args[1] is not None)}_debug{int(kwargs.get("debug",False))}'
                return launch(*args,**kwargs,compiled=kernels[key])
            return fn
        launchers[name]=make(name,options)
    return {'control':None,**manifest['configs']},launchers
