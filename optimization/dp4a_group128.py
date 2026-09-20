"""Experimental G128 weights with G32/G128 activations and scaled integer DP4A.

Moving each nibble into the high four bits of an INT8 byte replaces signed
extension. DP4A then returns 16 times the desired integer dot product. Eight
terms fit in INT32 even for weight -8 and activation -128. Removing the factor
before FP32 conversion preserves every integer sum and subsequent arithmetic.
"""
import torch
import triton
import triton.language as tl
import json
from pathlib import Path
from .int4_dp4a import _int8_activation_grouped


@triton.jit
def _dot(w,p,valid,MODE:tl.constexpr):
    scaled=tl.inline_asm_elementwise("""{
        .reg .b32 a, b, lo, hi;
        .reg .pred valid;
        mov.b32 a, 0;
        mov.b32 b, 0;
        setp.ne.s32 valid, $3, 0;
        @valid ld.global.v2.u32 {a, b}, [$2];
        shl.b32 lo, $1, 4;
        and.b32 lo, lo, 0xf0f0f0f0;
        and.b32 hi, $1, 0xf0f0f0f0;
        dp4a.s32.s32 $0, lo, a, 0;
        dp4a.s32.s32 $0, hi, b, $0;
    }""",constraints='=r,r,l,r',args=[w,p,valid.to(tl.int32)],
        dtype=tl.int32,is_pure=True,pack=1)
    if MODE==1:return scaled>>4
    return scaled


@triton.jit
def _projection(Q,XS,W,S,rows,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,R:tl.constexpr,AG:tl.constexpr,
                MODE:tl.constexpr,SCALE_MODE:tl.constexpr):
    g=tl.arange(0,BG)
    pos=g[:,None]*(AG//8)+tl.arange(0,AG//8)[None,:]
    w=tl.load(tl.cast(W,tl.pointer_type(tl.uint32))+rows[:,None,None]*(K//8)+pos[None,:,:],
        (rows[:,None,None]<N)&(pos[None,:,:]<K//8),0)
    p=tl.cast(Q,tl.pointer_type(tl.uint64))+pos
    dot=_dot(w,p[None,:,:],pos[None,:,:]<K//8,MODE)
    isum=tl.sum(dot,2)
    if MODE==2:isum=isum>>4
    sums=isum.to(tl.float32)
    if SCALE_MODE:g=tl.max_contiguous(g,SCALE_MODE)
    scale=tl.load(S+rows[:,None]*(K//128)+(g//(128//AG))[None,:],(rows[:,None]<N)&(g[None,:]<K//AG),0).to(tl.float32)
    xscale=tl.load(XS+g,g<K//AG,0)
    return tl.sum(sums*(scale*xscale[None,:]),1)


@triton.jit
def _gemv(Q,XS,W,S,Y,OQ,OS,N:tl.constexpr,K:tl.constexpr,BG:tl.constexpr,R:tl.constexpr,AG:tl.constexpr,
          MODE:tl.constexpr,PAIRED:tl.constexpr,FUSED:tl.constexpr,SCALE_MODE:tl.constexpr):
    rows=tl.program_id(0)*R+tl.arange(0,R)
    if PAIRED:
        gate=_projection(Q,XS,W,S,rows,N*2,K,BG,R,AG,MODE,SCALE_MODE).to(tl.bfloat16).to(tl.float32)
        up=_projection(Q,XS,W,S,rows+N,N*2,K,BG,R,AG,MODE,SCALE_MODE).to(tl.bfloat16).to(tl.float32)
        silu=(gate/(1+tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        value=(silu*up).to(tl.bfloat16)
    else:value=_projection(Q,XS,W,S,rows,N,K,BG,R,AG,MODE,SCALE_MODE).to(tl.bfloat16)
    tl.store(Y+rows,value,rows<N)
    if FUSED:
        grouped=tl.reshape(value.to(tl.float32),(R//AG,AG))
        scale=tl.maximum(tl.div_rn(tl.max(tl.abs(grouped),1),127.0),1e-8)
        inv=tl.div_rn(1.0,scale)
        quant=tl.reshape(tl.extra.cuda.libdevice.nearbyint(grouped*inv[:,None]),(R,)).to(tl.int8)
        tl.store(OQ+rows,quant,rows<N)
        groups=tl.program_id(0)*(R//AG)+tl.arange(0,R//AG)
        tl.store(OS+groups,scale,groups<N//AG)


def linear(x,packed,scales,*,rows=4,warps=4,mode=2,paired=False,fused=False,prequantized=None,scale_mode=0,activation_group=128):
    n,k2=packed.shape;k=k2*2;ag=activation_group
    if x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.numel()!=k or k%128 or ag not in (32,128) or mode not in (1,2):
        raise ValueError('One contiguous BF16 row, G128 weights and G32/G128 activations required')
    if fused and (not paired or rows%ag or n%(2*ag)):raise ValueError('Fused output quantization requires complete activation groups')
    if rows not in (1,2,4,8,16,32,64,128) or warps not in (1,2,4,8,16) or paired and n%2:raise ValueError('Invalid projection tile')
    if prequantized is None:
        q=torch.empty(k,device=x.device,dtype=torch.int8);sx=torch.empty(k//ag,device=x.device)
        _int8_activation_grouped[(triton.cdiv(k//ag,4),)](x,q,sx,k,ag,4,num_warps=4)
    else:q,sx=prequantized
    if packed.dtype!=torch.uint8 or scales.shape!=(n,k//128) or scales.dtype not in (torch.float32,torch.bfloat16):raise ValueError('Invalid G128 weights/scales')
    if q.dtype!=torch.int8 or q.numel()!=k or sx.shape!=(k//ag,) or sx.dtype!=torch.float32:raise ValueError('Invalid activation buffers')
    if not x.is_cuda or any(t.device!=x.device or not t.is_contiguous() for t in (packed,scales,q,sx)):raise ValueError('Contiguous CUDA buffers required')
    out_n=n//2 if paired else n
    out=torch.empty((*x.shape[:-1],out_n),device=x.device,dtype=x.dtype)
    oq=torch.empty(out_n if fused else 0,device=x.device,dtype=torch.int8)
    os=torch.empty(out_n//ag if fused else 0,device=x.device)
    _gemv[(triton.cdiv(out_n,rows),)](q,sx,packed,scales,out,oq,os,out_n,k,triton.next_power_of_2(k//ag),rows,ag,mode,paired,fused,scale_mode,num_warps=warps)
    return (out,(oq,os)) if fused else out


def project(module,name,x,prequantized=None):
    return linear(x,getattr(module,'_quant_'+name),getattr(module,'_quant_'+name+'_scale'),
                  **module._group128_plan[name],activation_group=module._dp4a_group,
                  paired=name=='up',prequantized=prequantized)


def mlp(module,x,prequantized=None):
    value=project(module,'up',x,prequantized)
    if module._group128_plan['up']['fused']:hidden,q=value
    else:hidden,q=value,None
    return project(module,'down',hidden,q)


@triton.jit
def _attention_reduce(PART,LSE,OUT,Q,SCALE,SPLITS:tl.constexpr,BS:tl.constexpr):
    h=tl.program_id(0);s=tl.arange(0,BS);d=tl.arange(0,128)
    lse=tl.load(LSE+h*SPLITS+s,s<SPLITS,float('-inf'))
    a=tl.exp(lse-tl.max(lse,0));a=a/tl.sum(a,0)
    val=tl.load(PART+(h*SPLITS+s[:,None])*128+d[None,:],s[:,None]<SPLITS,0)
    out=tl.sum(a[:,None]*val,0).to(tl.bfloat16)
    tl.store(OUT+h*128+d,out)
    values=out.to(tl.float32)
    scale=tl.maximum(tl.div_rn(tl.max(tl.abs(values),0),127.),1e-8)
    q=tl.extra.cuda.libdevice.nearbyint(values*tl.div_rn(1.,scale)).to(tl.int8)
    tl.store(Q+h*128+d,q);tl.store(SCALE+h,scale)


def reduce_attention(partial,lse):
    splits=partial.shape[1]
    out=torch.empty((1,1,4096),device=partial.device,dtype=torch.bfloat16)
    q=torch.empty(4096,device=partial.device,dtype=torch.int8);scale=torch.empty(32,device=partial.device)
    _attention_reduce[(32,)](partial,lse,out,q,scale,splits,triton.next_power_of_2(splits),num_warps=4)
    return out,(q,scale)


@torch.inference_mode()
def install(llm,plan_path):
    from .dp4a_packing import pack_interleaved
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Install G128 kernels before graph capture')
    plan=json.loads(Path(plan_path).read_text())
    if plan.get('format')!='group128_v1' or plan.get('codebooks')!=32 or plan.get('activation_group') not in (32,128):raise ValueError('Invalid G128 plan')
    cfgs=plan['projections'];ag=plan['activation_group']
    if set(cfgs)!=set(('qkv','out','up','down')):raise ValueError('Incomplete G128 plan')
    modules=[m for layer in llm.model.language_model.layers for m in (layer.self_attn,layer.mlp)]
    for m in modules:
        if not getattr(m,'_use_dp4a',False) or m._dp4a_group!=128 or not getattr(m,'_dp4a_grouped_activation',False) or getattr(m,'_dp4a_packing',None):raise ValueError('Unpacked calibrated G128 backbone with grouped quantization required')
        for name in (('qkv','out') if hasattr(m,'_qkv') else ('up','down')):
            w,s=getattr(m,'_quant_'+name),getattr(m,'_quant_'+name+'_scale');n,k2=w.shape
            c=cfgs[name]
            if w.dtype!=torch.uint8 or s.shape!=(n,k2*2//128) or not torch.equal(s,s.bfloat16().float()):raise ValueError('Invalid static G128 scales')
            if c['rows'] not in (1,2,4,8,16,32,64,128) or c['warps'] not in (1,2,4,8,16) or c['mode'] not in (1,2):raise ValueError('Invalid kernel plan')
            if c['fused'] and (name!='up' or c['rows']%ag):raise ValueError('Invalid output quantization tile')
    for m in modules:
        names=('qkv','out') if hasattr(m,'_qkv') else ('up','down')
        for name in names:
            setattr(m,'_quant_'+name,pack_interleaved(getattr(m,'_quant_'+name)))
            setattr(m,'_quant_'+name+'_scale',getattr(m,'_quant_'+name+'_scale').bfloat16())
        m._group128_plan={name:dict(cfgs[name]) for name in names}
        m._dp4a_packing={name:{'scheme':'interleaved'} for name in names}
        m._dp4a_group=ag
        if hasattr(m,'_qkv') and plan.get('attention_quant'):
            if not m._triton_decode or m._decode_backend is not None:raise ValueError('G128 attention fusion requires the custom attention backend')
            m._quantize_attention_output=ag
    if plan.get('norm_projections'):
        if not llm.fused_residual or not getattr(llm,'fused_dp4a',False) or llm.dp4a_norm_layout_limit!=8:raise ValueError('G128 producer fusion requires the selected residual/norm path')
        llm.group128_norm_fused={name:dict(cfg) for name,cfg in plan['norm_projections'].items()}
    return plan
