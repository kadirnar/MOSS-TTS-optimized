"""Decode split reduction fused with grouped activation quantization."""
import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_quant(PART,LSE,OUT,Q,SCALE,SPLITS:tl.constexpr,BS:tl.constexpr):
    h=tl.program_id(0)
    s=tl.arange(0,BS)
    d=tl.arange(0,128)
    lse=tl.load(LSE+h*SPLITS+s,s<SPLITS,float('-inf'))
    a=tl.exp(lse-tl.max(lse,0))
    a=a/tl.sum(a,0)
    val=tl.load(PART+(h*SPLITS+s[:,None])*128+d[None,:],s[:,None]<SPLITS,0)
    out=tl.sum(a[:,None]*val,0).to(tl.bfloat16)
    tl.store(OUT+h*128+d,out)
    values=tl.reshape(out.to(tl.float32),(4,32))
    scale=tl.maximum(tl.div_rn(tl.max(tl.abs(values),1),127.0),1e-8)
    inv=tl.div_rn(1.0,scale)
    quant=tl.reshape(tl.extra.cuda.libdevice.nearbyint(values*inv[:,None]),(128,))
    tl.store(Q+h*128+d,quant.to(tl.int8))
    tl.store(SCALE+h*4+tl.arange(0,4),scale)


def reduce_quant(partial,lse,warps=4):
    splits=partial.shape[1]
    out=torch.empty((1,1,4096),device=partial.device,dtype=torch.bfloat16)
    q=torch.empty(4096,device=partial.device,dtype=torch.int8)
    scale=torch.empty(128,device=partial.device,dtype=torch.float32)
    _reduce_quant[(32,)](partial,lse,out,q,scale,splits,triton.next_power_of_2(splits),num_warps=warps)
    return out,(q,scale)


def enable_attention_quant(llm):
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Enable attention quantization before capture')
    modules=[layer.self_attn for layer in llm.model.language_model.layers]
    if any(not getattr(m,'_dp4a_grouped_activation',False) or m._dp4a_group!=32 or not m._triton_decode or m._decode_backend is not None for m in modules):
        raise ValueError('Requires grouped G32 DP4A with custom decode attention')
    for m in modules:m._quantize_attention_output=True
