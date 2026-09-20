"""Fused BF16 prefill Q/K normalization, RoPE and KV writes; experimental."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_qkv(X,QW,KW,COS,SIN,Q,KC,VC,POS,L:tl.constexpr,EPS:tl.constexpr):
    token=tl.program_id(0);h=tl.program_id(1)
    d=tl.arange(0,128);swap=(d+64)%128;p=tl.load(POS+token)
    if h<32:
        base=token*6144+h*128
        w=tl.load(QW+d).to(tl.float32);ws=tl.load(QW+swap).to(tl.float32)
    else:
        base=token*6144+4096+(h-32)*128
        w=tl.load(KW+d).to(tl.float32);ws=tl.load(KW+swap).to(tl.float32)
    x=tl.load(X+base+d).to(tl.float32);xs=tl.load(X+base+swap).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/128+EPS)
    x=((x*inv).to(tl.bfloat16).to(tl.float32)*w).to(tl.bfloat16).to(tl.float32)
    xs=((xs*inv).to(tl.bfloat16).to(tl.float32)*ws).to(tl.bfloat16).to(tl.float32)
    c=tl.load(COS+token*128+d).to(tl.float32);s=tl.load(SIN+token*128+d).to(tl.float32)
    # Arithmetic 0 - x does not preserve unary negation's sign for +0.
    rotated=tl.where(d<64,(xs.to(tl.int32,bitcast=True)^(-2147483648)).to(tl.float32,bitcast=True),xs)
    value=(x*c).to(tl.bfloat16).to(tl.float32)+(rotated*s).to(tl.bfloat16).to(tl.float32)
    if h<32:tl.store(Q+(token*32+h)*128+d,value)
    else:
        kh=h-32;tl.store(KC+(kh*L+p)*128+d,value)
        v=tl.load(X+token*6144+5120+kh*128+d);tl.store(VC+(kh*L+p)*128+d,v)


def project(qkv,q_weight,k_weight,cos,sin,keys,values,positions,eps):
    """Internal FastLLM operation; caller supplies unique in-range KV positions.

    FastLLM and PrefixPrefillCache construct the validated contiguous position
    ranges. Do not synchronize or inspect device position values in capture.
    """
    if qkv.ndim!=3 or qkv.shape[0]!=1 or qkv.shape[2]!=6144 or qkv.dtype!=torch.bfloat16:
        raise ValueError('One BF16 batch of Qwen3-8B prefill QKV rows required')
    n=qkv.shape[1]
    if n<2 or any(x.shape!=(1,n,128) or x.dtype!=torch.bfloat16 for x in (cos,sin)):
        raise ValueError('Matching BF16 rotary rows and at least two tokens required')
    if any(w.shape!=(128,) or w.dtype!=torch.bfloat16 for w in (q_weight,k_weight)):
        raise ValueError('BF16 head normalization weights required')
    if keys.shape!=values.shape or keys.ndim!=4 or keys.shape[:2]!=(1,8) or keys.shape[3]!=128 or n>keys.shape[2]:
        raise ValueError('Matching Qwen3 KV buffers required')
    if positions.shape!=(n,) or positions.dtype!=torch.long or keys.dtype!=torch.bfloat16 or values.dtype!=torch.bfloat16:
        raise ValueError('Int64 positions and BF16 KV storage required')
    if not qkv.is_cuda or any(t.device!=qkv.device or not t.is_contiguous() for t in (qkv,q_weight,k_weight,cos,sin,keys,values,positions)):
        raise ValueError('Contiguous CUDA buffers on one device required')
    # Preserve the original transpose strides used by SDPA, including padding.
    q=torch.empty((1,n,32,128),device=qkv.device,dtype=qkv.dtype)
    _prefill_qkv[(n,40)](qkv,q_weight,k_weight,cos,sin,q,keys,values,positions,keys.shape[2],eps,num_warps=4)
    return q.transpose(1,2)


def enable(llm):
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Install prefill fusion before graph capture')
    if getattr(llm,'prefill_qkv',None):raise ValueError('Prefill fusion is already installed')
    layers=llm.model.language_model.layers
    for layer in layers:
        a=layer.self_attn
        if a._qkv_sizes!=(4096,1024,1024) and list(a._qkv_sizes)!=[4096,1024,1024]:raise ValueError('Qwen3-8B QKV sizes required')
        if a.head_dim!=128 or a.q_norm.variance_epsilon!=a.k_norm.variance_epsilon or getattr(a,'_fp8_prefill',False):
            raise ValueError('Qualified BF16 prefill and matching Q/K normalization required')
        if any(w.dtype!=torch.bfloat16 for w in (a._qkv,a.q_norm.weight,a.k_norm.weight)):
            raise ValueError('BF16 projection and head norm weights required')
    for layer in layers:layer.self_attn._fused_prefill_qkv=True
    llm.prefill_qkv={'layers':len(layers),'query_heads':32,'kv_heads':8,'head_dim':128,'dtype':'bfloat16','codebooks':32}
    return dict(llm.prefill_qkv)
