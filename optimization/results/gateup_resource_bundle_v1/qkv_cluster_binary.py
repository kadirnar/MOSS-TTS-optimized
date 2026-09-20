"""Export isolated compiler cubins and launch them with the selected runtime.

Only this experimental fused kernel is compiled by Triton 3.8. The host model,
attention and all other Triton kernels continue to use the selected runtime.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import torch
import triton
from .common import RESULTS
from .ptx_resources import Attribute,LaunchConfig,check
from .qkv_cluster_prepare import launch


class Binary:
    def __init__(self,folder,record):
        self.record=record;self.metadata=SimpleNamespace(**record['metadata'])
        self.device=torch.cuda.current_device()
        self.n_regs=record['registers'];self.n_spills=record['spills']
        self.asm={s:(folder/(record['prefix']+'.'+s)).read_bytes() for s in ('cubin','ptx','ttgir')}
        for s in ('ptx','ttgir'):self.asm[s]=self.asm[s].decode()
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


@torch.inference_mode()
def main():
    from .benchmark_attention_pdl import load_layer
    from .benchmark_qkv_cluster import configs
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--configs',nargs='+',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    choices={n:configs(False)[n] for n in a.configs};assert all(choices.values())
    folder=RESULTS/f'qkv_cluster_bundle_{a.tag}';folder.mkdir(exist_ok=False)
    for filename in ('qkv_cluster_prepare.py','qkv_cluster_binary.py'):
        (folder/filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    torch.set_num_threads(4);entries={False:load_layer(0),True:load_layer(17)}
    manifest={'format':'qkv_cluster_cubin_v1','complete':False,'torch':torch.__version__,'triton':triton.__version__,
        'configs':choices,'binaries':{},'kernel_source_sha256':hashlib.sha256(Path(__file__).with_name('qkv_cluster_prepare.py').read_bytes()).hexdigest()}
    for name,options in choices.items():
        for add,e in entries.items():
            d=e['attention'];raw=e['raw'];assert bool(raw['has_residual'])==add
            for debug in (False,True):
                _,k=launch(raw['x'][:1],raw['residual'][:1] if add else None,raw['weight'],raw['eps'],*e['weights']['qkv'],
                    d['qw'],d['kw'],d['cos'],d['sin'],d['k'],d['v'],d['position'],d['eps'],**options,debug=debug,return_kernel=True)
                prefix=f'{name}_add{int(add)}_debug{int(debug)}'
                for suffix in ('ptx','ttgir','cubin'):
                    v=k.asm[suffix];(folder/(prefix+'.'+suffix)).write_bytes(v if isinstance(v,bytes) else v.encode())
                (folder/(prefix+'.sass')).write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(folder/(prefix+'.cubin'))],text=True))
                signature=[(key,value) for key,value in k.src.signature.items() if value.startswith('*')]
                assert all(v=='constexpr' or v.startswith('*') for v in k.src.signature.values()),k.src.signature
                params=re.search(r'\.visible \.entry [^(]+\((.*?)\)\s*\.reqntid',k.asm['ptx'],re.S)
                assert params and len(re.findall(r'\.param\b',params[1]))==len(signature)+2
                metadata={key:getattr(k.metadata,key) for key in ('name','shared','num_warps','num_ctas','launch_pdl','global_scratch_size','profile_scratch_size')}
                # Triton 3.8 fixes cluster launch dimensions in its NVIDIA driver;
                # they are no longer fields of KernelMetadata.
                metadata['cluster_dims']=[k.metadata.num_ctas,1,1]
                manifest['binaries'][prefix]={'prefix':prefix,'metadata':metadata,
                    'registers':k.n_regs,'spills':k.n_spills,'pointer_signature':signature,'eps':raw['eps'],'head_eps':d['eps'],
                    'length':d['k'].shape[-2],'cubin_sha256':hashlib.sha256(k.asm['cubin']).hexdigest()}
                print('EXPORTED',prefix,flush=True)
    torch.cuda.synchronize();manifest['complete']=True
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('SAVED',folder,flush=True)


if __name__=='__main__':main()
