"""Experimental exact native attention with programmatic dependencies."""
import ctypes
import functools
import hashlib
from pathlib import Path
import subprocess
from .paths import build_directory, nvcc
import torch


@functools.lru_cache(None)
def library():
    source=Path(__file__).with_suffix('.cu')
    tag=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder=build_directory('attention_pdl', tag)
    so=folder/f'attention_{tag}.so'
    snapshot=folder/f'attention_{tag}.cu'
    if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    if not so.exists():
        command=[nvcc(),'-O3','-std=c++17','-arch=sm_90',
            '--shared','-Xcompiler','-fPIC','-Xptxas=-v',str(source),'-o',str(so)]
        result=subprocess.run(command,text=True,capture_output=True)
        (folder/f'attention_{tag}.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
    lib=ctypes.CDLL(str(so))
    lib.launch_attention_pdl.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*4+[ctypes.c_void_p]
    lib.launch_attention_pdl.restype=ctypes.c_int
    return lib


def launch(q,k,v,position,partial,lse,*,pdl=True,trigger=1):
    if trigger not in (0,1,2,3):raise ValueError("Invalid PDL trigger")
    if q.shape!=(32,128) or k.ndim!=4 or k.shape[:2]!=(1,8) or k.shape[-1]!=128 or v.shape!=k.shape:
        raise ValueError('Native attention requires Q [32,128], KV [1,8,L,128]')
    length=k.shape[-2]
    if length%32 or partial.ndim!=3 or partial.shape[0]!=32 or partial.shape[2]!=128 or not 0<partial.shape[1]*32<=length or lse.shape!=partial.shape[:2]:
        raise ValueError('Invalid native split-attention buffers')
    tensors=(q,k,v,position,partial,lse)
    dtypes=(torch.bfloat16,torch.bfloat16,torch.bfloat16,torch.int64,torch.float32,torch.float32)
    if not q.is_cuda or position.numel()!=1 or any(t.device!=q.device or t.dtype!=dt or not t.is_contiguous() or t.data_ptr()%16 for t,dt in zip(tensors,dtypes)):
        raise ValueError('Native attention requires contiguous aligned CUDA tensors with matching dtypes/devices')
    result=library().launch_attention_pdl(q.data_ptr(),k.data_ptr(),v.data_ptr(),
        position.data_ptr(),partial.data_ptr(),lse.data_ptr(),k.shape[-2],
        partial.shape[1],int(pdl),trigger,torch.cuda.current_stream(q.device).cuda_stream)
    if result:raise RuntimeError(f'Native attention launch failed: CUDA error {result}')


def enable(llm,*,qk_trigger=1,attention_trigger=2,reduce_trigger=1,preload=False):
    """Install only on the selected exact G32/SM90 path, before graph capture."""
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Install attention PDL before capture')
    if torch.cuda.get_device_capability()!=(9,0):raise ValueError('Attention PDL is qualified only for SM90')
    if not getattr(llm,'projection_pdl',None):raise ValueError('Attention PDL requires the selected projection PDL preset')
    if any(t not in (0,1,2,3) for t in (qk_trigger,attention_trigger,reduce_trigger)):
        raise ValueError('Invalid PDL trigger')
    modules=[layer.self_attn for layer in llm.model.language_model.layers]
    if any(not getattr(m,'_native_attention',False) or getattr(m,'_quantize_attention_output',False)!=True
           or getattr(m,'_decode_backend',None) is not None or (m._decode_block,m._decode_warps)!=(32,4)
           for m in modules):
        raise ValueError('Attention PDL requires native B32/W4 attention and G32 output quantization')
    library()
    config={'pdl':True,'qk':qk_trigger,'attention':attention_trigger,'reduce':reduce_trigger,'preload':bool(preload)}
    llm.attention_pdl=config
    for module in modules:module._attention_pdl=config
    return dict(config)
