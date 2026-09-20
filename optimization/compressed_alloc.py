"""Experimental lossless CUDA allocations; no global Torch allocator changes.

Allocate outside CUDA graph capture, on the tensor's current stream. Keep tensor
owners alive until every graph using them has been destroyed. Final storage
destruction synchronizes its CUDA context because Torch cannot record streams on
this external allocation. Views retain native DLPack ownership automatically.
"""
import functools
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sysconfig

import torch


@functools.lru_cache(None)
def library():
    source=Path(__file__).with_suffix('.cpp')
    identity=source.read_bytes()+torch.__version__.encode()+sysconfig.get_config_var('SOABI').encode()
    tag=hashlib.sha256(identity).hexdigest()[:16]
    folder=source.parent/'results/compressed_alloc_build';folder.mkdir(exist_ok=True)
    binary=folder/f'allocator_{tag}.so'
    if not binary.exists():
        command=['g++','-O2','-std=c++17','-shared','-fPIC',str(source),
                 '-I'+sysconfig.get_path('include'),'-I'+str(Path(torch.__file__).parent/'include'),
                 '-I/usr/local/cuda/include','-L/usr/local/cuda/lib64/stubs','-lcuda','-o',str(binary)]
        result=subprocess.run(command,text=True,capture_output=True)
        (folder/f'allocator_{tag}.log').write_text(result.stdout+result.stderr)
        (folder/f'allocator_{tag}.cpp').write_bytes(source.read_bytes())
        result.check_returncode()
    spec=importlib.util.spec_from_file_location('_moss_compressed_alloc',binary)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


DTYPES={torch.uint8:(1,8),torch.int8:(0,8),torch.int32:(0,32),torch.int64:(0,64),
        torch.float16:(2,16),torch.bfloat16:(4,16),torch.float32:(2,32),torch.float64:(2,64)}


def empty(shape,*,dtype,device,compressed=True):
    device=torch.device(device)
    if device.type!='cuda' or dtype not in DTYPES:raise ValueError('Supported CUDA scalar dtype required')
    device=torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
    with torch.cuda.device(device):
        torch.cuda.init()
        if torch.cuda.get_device_capability(device)!=(9,0):raise ValueError('This allocation path is validated for SM90 only')
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Allocate VMM buffers before graph capture')
        capsule,metadata=library().allocate(tuple(shape),*DTYPES[dtype],device.index,int(compressed))
        value=torch.from_dlpack(capsule)
    assert value.shape==tuple(shape) and value.dtype==dtype and value.device==device
    return value,metadata


@torch.inference_mode()
def clone(value,*,compressed=True):
    if not value.is_cuda or not value.is_contiguous() or value.numel()==0:
        raise ValueError('Nonempty contiguous CUDA tensor required')
    out,metadata=empty(value.shape,dtype=value.dtype,device=value.device,compressed=compressed)
    # This asynchronous copy uses the current stream, just like ordinary clone.
    out.copy_(value)
    return out,metadata


@torch.inference_mode()
def enable_compressed_scales(llm):
    """Opt in before capture; replace only the selected out/down FP32 scales.

    Build and verify every replacement before mutating the model, so allocation
    failures cannot leave a partially converted model. Module buffers retain the
    DLPack owners for their entire graph lifetime. This is lossless allocation
    compression, not a scale-value or codebook-count change.
    """
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Install compressed scales before graph capture')
    if not getattr(llm,'norm_projection_fused',None) or getattr(llm,'compressed_scales',None):
        raise ValueError('Requires the G32 norm-projection preset, without prior compressed allocation')
    layers=llm.model.language_model.layers
    if len(layers)!=36:raise ValueError('Validated MOSS 8B backbone required')
    replacements=[];metadata=[]
    for index,layer in enumerate(layers):
        for module,name,shape in ((layer.self_attn,'out',(4096,128)),(layer.mlp,'down',(4096,384))):
            key='_quant_'+name+'_scale';source=getattr(module,key)
            if getattr(module,'_dp4a_group',None)!=32 or not getattr(module,'_scaled_dp4a',None) or getattr(module,'_group128_plan',None):
                raise ValueError('Only the selected G32 scaled-DP4A buffers are supported')
            if source.dtype!=torch.float32 or tuple(source.shape)!=shape:raise ValueError('Selected FP32 scale shape required')
            target,info=clone(source)
            if not torch.equal(target,source):raise RuntimeError('Compressed scale copy changed values')
            replacements.append((module,key,target));metadata.append({'layer':index,'projection':name,**info})
    for module,key,target in replacements:setattr(module,key,target)
    llm.compressed_scales={'allocations':metadata,'logical_bytes':sum(m['logical_bytes'] for m in metadata),
                           'allocated_bytes':sum(m['allocated_bytes'] for m in metadata),
                           'buffer_count':len(metadata),'codebooks':32,'all_copies_exact':True}
    return llm.compressed_scales
