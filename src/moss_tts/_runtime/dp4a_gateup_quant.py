"""Experimental gate/up/SiLU and following grouped quantizer fusion."""
import torch
import triton
import triton.language as tl
from .dp4a_packing import _projection as packed_projection
from .dp4a_direct import _projection as direct_projection
from .dp4a_scale import _projection as scale_projection
from .int4_dp4a import _int8_activation_grouped


@triton.jit
def _gateup_quant(Q,XS,W,S,Y,OQ,OS,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,
                  R:tl.constexpr,DIRECT:tl.constexpr,SCALE_MODE:tl.constexpr=0):
    rows=tl.program_id(0)*R+tl.arange(0,R)
    if SCALE_MODE:
        gate=scale_projection(Q,XS,W,S,rows,N*2,K,BG,R,SCALE_MODE).to(tl.bfloat16).to(tl.float32)
        up=scale_projection(Q,XS,W,S,rows+N,N*2,K,BG,R,SCALE_MODE).to(tl.bfloat16).to(tl.float32)
    elif DIRECT:
        gate=direct_projection(Q,XS,W,S,rows,N*2,K,BG,R).to(tl.bfloat16).to(tl.float32)
        up=direct_projection(Q,XS,W,S,rows+N,N*2,K,BG,R).to(tl.bfloat16).to(tl.float32)
    else:
        gate=packed_projection(Q,XS,W,S,rows,N*2,K,32,BG,R,True).to(tl.bfloat16).to(tl.float32)
        up=packed_projection(Q,XS,W,S,rows+N,N*2,K,32,BG,R,True).to(tl.bfloat16).to(tl.float32)
    silu=(gate/(1+tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    value=(silu*up).to(tl.bfloat16)
    tl.store(Y+rows,value,rows<N)
    grouped=tl.reshape(value.to(tl.float32),(R//32,32))
    scale=tl.maximum(tl.div_rn(tl.max(tl.abs(grouped),1),127.0),1e-8)
    inv=tl.div_rn(1.0,scale)
    quant=tl.reshape(tl.extra.cuda.libdevice.nearbyint(grouped*inv[:,None]),(R,)).to(tl.int8)
    tl.store(OQ+rows,quant,rows<N)
    groups=tl.program_id(0)*(R//32)+tl.arange(0,R//32)
    tl.store(OS+groups,scale,groups<N//32)


def gateup_quant(x,packed,scales,*,rows=32,warps=4,direct=True,prequantized=None,scale_mode=4):
    n2,k2=packed.shape;n=n2//2;k=k2*2
    if n2%64 or rows not in (32,64) or k%32 or x.numel()!=k or x.dtype!=torch.bfloat16:
        raise ValueError('Specialized gate/up fusion requires G32 input/output dimensions')
    if scale_mode not in (0,1,2,4,8) or scale_mode and not direct:raise ValueError('Scale layout hints require direct loads')
    if prequantized is None:
        q=torch.empty(k,device=x.device,dtype=torch.int8);sx=torch.empty(k//32,device=x.device)
        _int8_activation_grouped[(triton.cdiv(k//32,4),)](x,q,sx,k,32,4,num_warps=4)
    else:q,sx=prequantized
    out=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    oq=torch.empty(n,device=x.device,dtype=torch.int8);os=torch.empty(n//32,device=x.device)
    _gateup_quant[(triton.cdiv(n,rows),)](q,sx,packed,scales,out,oq,os,n,k,triton.next_power_of_2(k//32),rows,direct,scale_mode,num_warps=warps)
    return out,(oq,os)


def enable_gateup_quant(llm):
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Enable gate/up quantizer fusion before capture')
    modules=[layer.mlp for layer in llm.model.language_model.layers]
    for m in modules:
        cfg=(getattr(m,'_dp4a_packing',None) or {}).get('up',{})
        if not getattr(m,'_dp4a_grouped_activation',False) or m._dp4a_group!=32 or cfg.get('scheme')!='interleaved' or cfg.get('scale_dtype')!='bfloat16':
            raise ValueError('Gate/up quantizer fusion requires interleaved G32 DP4A and BF16 scales')
        if tuple(m._quant_up.shape)!=(24576,2048) or tuple(m._quant_up_scale.shape)!=(24576,128):
            raise ValueError('Gate/up quantizer fusion is specialized to MOSS-TTS-v1.5 8B')
    for m in modules:m._fused_gateup_quant=True
