"""Experimental BF16 prefill activation and residual/normalization fusion."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .kernels import add_rmsnorm


@torch.inference_mode()
def make_silu_table(device):
    # Every BF16 bit pattern, evaluated by the installed reference CUDA kernel.
    bits=torch.arange(65536,device=device,dtype=torch.int32).to(torch.int16)
    return F.silu(bits.view(torch.bfloat16)).contiguous()


@triton.jit
def _silu(X,TABLE,Y,N:tl.constexpr,B:tl.constexpr,MODE:tl.constexpr):
    row=tl.program_id(0);d=tl.program_id(1)*B+tl.arange(0,B)
    gate=tl.load(X+row*(2*N)+d,d<N,0)
    up=tl.load(X+row*(2*N)+N+d,d<N,0).to(tl.float32)
    if MODE==2:
        index=gate.to(tl.uint16,bitcast=True).to(tl.int32)
        value=tl.load(TABLE+index).to(tl.float32)
    else:
        x=gate.to(tl.float32)
        if MODE==1:value=tl.div_rn(x,1.0+libdevice.exp(-x))
        else:value=x/(1.0+tl.exp(-x))
        value=value.to(tl.bfloat16).to(tl.float32)
    tl.store(Y+row*N+d,value*up,d<N)


def silu_mul(x,table=None,*,block=512,warps=4,mode=0,return_kernel=False):
    if x.ndim!=3 or x.shape[0]!=1 or x.shape[1]<2 or x.shape[2]!=24576 or x.dtype!=torch.bfloat16:
        raise ValueError('Batch-one BF16 Qwen3-8B prefill gate/up rows required')
    if not x.is_cuda or not x.is_contiguous():raise ValueError('Contiguous CUDA input required')
    if mode==2 and (table is None or table.shape!=(65536,) or table.dtype!=torch.bfloat16 or table.device!=x.device or not table.is_contiguous()):
        raise ValueError('Matching complete BF16 SiLU table required')
    if block not in (128,256,512,1024,2048) or warps not in (4,8) or mode not in (0,1,2):raise ValueError('Unsupported launch')
    out=torch.empty((1,x.shape[1],12288),device=x.device,dtype=x.dtype)
    kernel=_silu[(x.shape[1],triton.cdiv(12288,block))](x,table,out,12288,block,mode,num_warps=warps)
    return (out,kernel) if return_kernel else out


def hidden(llm,emb,position,mask):
    backbone=llm.model.language_model;rope=backbone.rotary_emb(emb,position[None])
    residual=emb;state=backbone.layers[0].input_layernorm(emb)
    for i,layer in enumerate(backbone.layers):
        attention,_=layer.self_attn(hidden_states=state,position_embeddings=rope,
            attention_mask=mask,past_key_values=llm.cache,cache_position=position)
        norm=layer.post_attention_layernorm
        residual,state=add_rmsnorm(attention,residual,norm.weight,norm.variance_epsilon)
        mlp=layer.mlp(state)
        norm=backbone.layers[i+1].input_layernorm if i+1<len(backbone.layers) else backbone.norm
        residual,state=add_rmsnorm(mlp,residual,norm.weight,norm.variance_epsilon)
    return state


def enable(llm):
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Install before graph capture')
    if getattr(llm,'prefill_pointwise',None):raise ValueError('Already installed')
    layers=llm.model.language_model.layers
    for layer in layers:
        m=layer.mlp
        if m._gate_up.shape!=(24576,4096) or m._gate_up.dtype!=torch.bfloat16 or getattr(m,'_fp8_prefill',False):
            raise ValueError('Qualified BF16 Qwen3-8B prefill required')
    for layer in layers:layer.mlp._prefill_silu_config={'mode':0,'block':512,'warps':4}
    llm._prefill_residual_fused=True
    llm.prefill_pointwise={'layers':len(layers),'silu_mode':'direct','residual_norm_fused':True,'block':512,'warps':4,'codebooks':32}
    return dict(llm.prefill_pointwise)
