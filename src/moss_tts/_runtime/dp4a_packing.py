"""Packed INT4 layout experiments with explicit PTX byte permutation.

PRMT semantics: NVIDIA PTX ISA, Data Movement and Conversion Instructions.
Interleaved layout stores values 0..3 in low nibbles and 4..7 in high nibbles
of one uint32. Scale arithmetic and output rounding remain FP32/BF16.
"""
import torch
import triton
import triton.language as tl
import json
from pathlib import Path
from .int4_dp4a import _int8_activation_grouped


def pack_interleaved(packed):
    n,k2=packed.shape
    if packed.dtype!=torch.uint8 or k2%4:raise ValueError('Expected signed-nibble pairs with K divisible by eight')
    unsigned=torch.stack([packed&15,packed>>4],dim=-1).reshape(n,-1,8)
    return (unsigned[:,:,:4]|(unsigned[:,:,4:]<<4)).reshape(n,k2).contiguous()


def unpack_interleaved(packed):
    n,k2=packed.shape
    words=packed.reshape(n,-1,4)
    unsigned=torch.cat([words&15,words>>4],dim=-1).reshape(n,-1)
    signed=unsigned.to(torch.int8)
    return torch.where(signed>=8,signed-16,signed)


@triton.jit
def _expand_prmt(w):
    return tl.inline_asm_elementwise("""{
        .reg .b32 shifted, bits, sign;
        shr.u32 shifted, $1, 4;
        prmt.b32 bits, $1, shifted, 0x5140;
        and.b32 bits, bits, 0x0f0f0f0f;
        and.b32 sign, bits, 0x08080808;
        mad.lo.u32 $0, sign, 30, bits;
    }""",constraints='=r,r',args=[w],dtype=tl.int32,is_pure=True,pack=1)


@triton.jit
def _sign_extend(lanes):
    return (lanes|((lanes&0x08080808)*30)).to(tl.int32)


@triton.jit
def _dot(a,b,c):
    return tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, $3;',
        constraints='=r,r,r,r',args=[a,b,c],dtype=tl.int32,is_pure=True,pack=1)


@triton.jit
def _projection(Q,XS,W,S,rows,N:tl.constexpr,K:tl.constexpr,GROUP:tl.constexpr,
                BG:tl.constexpr,R:tl.constexpr,INTERLEAVED:tl.constexpr):
    g=tl.arange(0,BG)
    if INTERLEAVED:
        chunk=tl.arange(0,GROUP//8)
        pos=g[:,None]*(GROUP//8)+chunk[None,:]
        w=tl.load(tl.cast(W,tl.pointer_type(tl.uint32))+rows[:,None,None]*(K//8)+pos[None,:,:],
            (rows[:,None,None]<N)&(pos[None,:,:]<K//8),0)
        # One 64-bit load supplies two adjacent packed INT8 activation words.
        x=tl.load(tl.cast(Q,tl.pointer_type(tl.uint64))+pos,pos<K//8,0)
        lo=_sign_extend(w&0x0f0f0f0f)
        hi=_sign_extend((w>>4)&0x0f0f0f0f)
        dot=_dot(lo,x[None,:,:].to(tl.int32),tl.full((R,BG,GROUP//8),0,tl.int32))
        dot=_dot(hi,(x[None,:,:]>>32).to(tl.int32),dot)
    else:
        chunk=tl.arange(0,GROUP//4)
        pos=g[:,None]*(GROUP//4)+chunk[None,:]
        w=tl.load(tl.cast(W,tl.pointer_type(tl.uint16))+rows[:,None,None]*(K//4)+pos[None,:,:],
            (rows[:,None,None]<N)&(pos[None,:,:]<K//4),0).to(tl.uint32)
        x=tl.load(tl.cast(Q,tl.pointer_type(tl.int32))+pos,pos<K//4,0)
        dot=_dot(_expand_prmt(w),x[None,:,:],tl.full((R,BG,GROUP//4),0,tl.int32))
    sums=tl.sum(dot,2).to(tl.float32)
    scale=tl.load(S+rows[:,None]*(K//GROUP)+g[None,:],(rows[:,None]<N)&(g[None,:]<K//GROUP),0).to(tl.float32)
    xscale=tl.load(XS+g,g<K//GROUP,0)
    return tl.sum(sums*(scale*xscale[None,:]),1)


@triton.jit
def _packed_gemv(Q,XS,W,S,Y,N:tl.constexpr,K:tl.constexpr,GROUP:tl.constexpr,
                 BG:tl.constexpr,R:tl.constexpr,INTERLEAVED:tl.constexpr,PAIRED:tl.constexpr):
    rows=tl.program_id(0)*R+tl.arange(0,R)
    if PAIRED:
        gate=_projection(Q,XS,W,S,rows,N*2,K,GROUP,BG,R,INTERLEAVED).to(tl.bfloat16).to(tl.float32)
        up=_projection(Q,XS,W,S,rows+N,N*2,K,GROUP,BG,R,INTERLEAVED).to(tl.bfloat16).to(tl.float32)
        silu=(gate/(1+tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        value=silu*up
    else:value=_projection(Q,XS,W,S,rows,N,K,GROUP,BG,R,INTERLEAVED)
    tl.store(Y+rows,value,rows<N)


def linear(x,packed,scales,*,group=32,rows=1,warps=1,interleaved=False,paired=False,prequantized=None):
    n,k2=packed.shape;k=k2*2
    if x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.numel()!=k:raise ValueError('One contiguous BF16 input row required')
    if prequantized is None:
        q=torch.empty(k,device=x.device,dtype=torch.int8)
        sx=torch.empty(k//group,device=x.device,dtype=torch.float32)
        _int8_activation_grouped[(triton.cdiv(k//group,4),)](x,q,sx,k,group,4,num_warps=4)
    else:q,sx=prequantized
    out_n=n//2 if paired else n
    out=torch.empty((*x.shape[:-1],out_n),device=x.device,dtype=x.dtype)
    _packed_gemv[(triton.cdiv(out_n,rows),)](q,sx,packed,scales,out,out_n,k,group,
        triton.next_power_of_2(k//group),rows,interleaved,paired,num_warps=warps)
    return out


def project(module,name,x,prequantized=None):
    if getattr(module,'_short_scale_projection',None)==name:
        from .short_scales import project as short_project
        if prequantized is None:raise ValueError('Short scales require the selected fused activation producers')
        return short_project(module,name,x,prequantized)
    cfg=module._dp4a_packing[name]
    if getattr(module,'_scaled_dp4a',None):
        from .dp4a_scaled import linear as scaled_linear
        return scaled_linear(x,getattr(module,'_quant_'+name),getattr(module,'_quant_'+name+'_scale'),
            **module._scaled_dp4a[name],paired=name=='up',prequantized=prequantized)
    if cfg.get('activation_load')=='direct':
        from .dp4a_direct import linear as direct_linear
        return direct_linear(x,getattr(module,'_quant_'+name),getattr(module,'_quant_'+name+'_scale'),
            rows=cfg['rows'],warps=cfg['warps'],paired=name=='up',prequantized=prequantized)
    return linear(x,getattr(module,'_quant_'+name),getattr(module,'_quant_'+name+'_scale'),
        group=module._dp4a_group,rows=cfg['rows'],warps=cfg['warps'],
        interleaved=cfg['scheme']=='interleaved',paired=name=='up',prequantized=prequantized)


@torch.inference_mode()
def install_packing(llm,plan_path):
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Install packing before graph capture')
    plan=json.loads(Path(plan_path).read_text())
    if plan['format']!='dp4a_packing_v1' or plan['group']!=32 or plan['codebooks']!=32:
        raise ValueError('Unsupported packing plan')
    cfgs=plan['projections']
    if set(cfgs)!={'qkv','out','up','down'}:raise ValueError('Incomplete packing plan')
    for cfg in cfgs.values():
        if cfg['scheme'] not in ('prmt','interleaved') or cfg['scale_dtype'] not in ('bfloat16','float32') or cfg['rows'] not in (1,2,4,8) or cfg['warps'] not in (1,2,4,8):
            raise ValueError('Invalid packing configuration')
        if cfg.get('activation_load','standard') not in ('standard','direct'):
            raise ValueError('Invalid activation load mode')
        if cfg.get('activation_load')=='direct' and cfg['scheme']!='interleaved':
            raise ValueError('Direct loads require interleaved weights')
    modules=[m for layer in llm.model.language_model.layers for m in (layer.self_attn,layer.mlp)]
    if any(not getattr(m,'_dp4a_grouped_activation',False) or m._dp4a_group!=32 for m in modules):
        raise ValueError('Packing requires grouped G32 DP4A activations')
    if any(getattr(m,'_dp4a_packing',None) for m in modules):raise ValueError('Weights are already repacked')
    # Check scale casts for every projection before mutating any weight layout.
    targets=[(m,name) for m in modules for name in (('qkv','out') if hasattr(m,'_qkv') else ('up','down'))]
    for m,name in targets:
        scales=getattr(m,'_quant_'+name+'_scale')
        if cfgs[name]['scale_dtype']=='bfloat16' and not torch.equal(scales,scales.bfloat16().float()):
            raise ValueError('Requested BF16 scale storage would change values')
    for m,name in targets:
        cfg=cfgs[name]
        if cfg['scheme']=='interleaved':setattr(m,'_quant_'+name,pack_interleaved(getattr(m,'_quant_'+name)))
        if cfg['scale_dtype']=='bfloat16':setattr(m,'_quant_'+name+'_scale',getattr(m,'_quant_'+name+'_scale').bfloat16())
    for m in modules:
        names=('qkv','out') if hasattr(m,'_qkv') else ('up','down')
        m._dp4a_packing={name:cfgs[name] for name in names}
    return plan
