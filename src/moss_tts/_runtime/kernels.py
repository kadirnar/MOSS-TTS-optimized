"""Inference-only Triton kernels. All launches use PyTorch's current CUDA stream."""
import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_sum(IDS, TEXT, AUDIO, OUT, H: tl.constexpr, V: tl.constexpr, Q: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tok = tl.load(IDS + t * (Q + 1))
    x = tl.load(TEXT + tok * H + d, d < H, 0)
    # Preserve upstream's BF16 rounding after each sequential addition.
    for q in range(Q):
        c = tl.load(IDS + t * (Q + 1) + q + 1)
        y = tl.load(AUDIO + (q * V + c) * H + d, d < H, 0)
        x = (x.to(tl.float32) + y.to(tl.float32)).to(OUT.dtype.element_ty)
    tl.store(OUT + t * H + d, x, d < H)


def embedding_sum(ids, text_weight, audio_weight):
    ids = ids.contiguous()
    out = torch.empty((*ids.shape[:-1], text_weight.shape[-1]), device=ids.device, dtype=text_weight.dtype)
    _embedding_sum[(ids.numel() // ids.shape[-1], triton.cdiv(out.shape[-1], 256))](
        ids, text_weight, audio_weight, out, out.shape[-1], audio_weight.shape[1], ids.shape[-1]-1, 256)
    return out


@triton.jit
def _codebook_sum(CODES, TABLE, OUT, T: tl.constexpr, Q: tl.constexpr, V: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.full((BLOCK,), 0, tl.float32)
    for q in range(Q):
        c = tl.load(CODES + q * T + t)
        x = x + tl.load(TABLE + (q * V + c) * D + d, d < D, 0)
    tl.store(OUT + d * T + t, x, d < D)


def codebook_sum(codes, table):
    if codes.ndim != 3 or codes.shape[1] != 1:
        raise ValueError("Optimized codec currently requires [Q, 1, T] codes")
    codes = codes.contiguous()
    q, _, t = codes.shape
    out = torch.empty((1, table.shape[-1], t), device=codes.device, dtype=torch.float32)
    _codebook_sum[(t, triton.cdiv(table.shape[-1], 128))](codes, table, out, t, q, table.shape[1], table.shape[-1], 128)
    return out


@triton.jit
def _rmsnorm(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + d, d < N, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x*x, 0) / N + EPS)
    # Qwen3 casts normalized activations before weight multiplication.
    z = (x*r).to(Y.dtype.element_ty).to(tl.float32)
    w = tl.load(W + d, d < N, 0).to(tl.float32)
    tl.store(Y + row * N + d, z*w, d < N)


def rmsnorm(x, weight, eps):
    x = x.contiguous()
    y = torch.empty_like(x)
    _rmsnorm[(x.numel()//x.shape[-1],)](x, weight, y, x.shape[-1], eps, triton.next_power_of_2(x.shape[-1]))
    return y


@triton.jit
def _add_rmsnorm(X,R,W,S,Y,N:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    d=tl.arange(0,B)
    x=tl.load(X+row*N+d,d<N,0).to(tl.float32)
    r=tl.load(R+row*N+d,d<N,0).to(tl.float32)
    # Preserve the BF16 residual addition and normalized-activation rounding.
    summed=(x+r).to(S.dtype.element_ty)
    tl.store(S+row*N+d,summed,d<N)
    f=summed.to(tl.float32)
    inv=tl.rsqrt(tl.sum(f*f,0)/N+EPS)
    normalized=(f*inv).to(Y.dtype.element_ty).to(tl.float32)
    weight=tl.load(W+d,d<N,0).to(tl.float32)
    tl.store(Y+row*N+d,normalized*weight,d<N)


def add_rmsnorm(x,residual,weight,eps):
    summed=torch.empty_like(x)
    normalized=torch.empty_like(x)
    _add_rmsnorm[(x.numel()//x.shape[-1],)](x,residual,weight,summed,normalized,
        x.shape[-1],eps,triton.next_power_of_2(x.shape[-1]))
    return summed,normalized


@triton.jit
def _gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)
    k=tl.arange(0,BK)
    x=tl.load(X+k,k<K,0).to(tl.float32)
    w=tl.load(W+row*K+k,k<K,0.0).to(tl.float32)
    tl.store(Y+row,tl.sum(w*x,0))


def linear_decode(x,w):
    if x.numel()!=x.shape[-1]:
        return torch.nn.functional.linear(x,w)
    n,k=w.shape
    y=torch.empty((*x.shape[:-1],n),dtype=x.dtype,device=x.device)
    _gemv[(n,)](x,w,y,n,k,triton.next_power_of_2(k),num_warps=4 if k<=4096 else 8)
    return y














@triton.jit
def _silu_mul(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    a=tl.load(X+i,i<N,0).to(tl.float32)
    b=tl.load(X+N+i,i<N,0).to(tl.float32)
    s=(a/(1+tl.exp(-a))).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,s*b,i<N)


def silu_mul(x):
    n=x.shape[-1]//2
    if x.numel()!=2*n:
        gate,up=x.chunk(2,-1)
        return torch.nn.functional.silu(gate)*up
    y=torch.empty((*x.shape[:-1],n),dtype=x.dtype,device=x.device)
    _silu_mul[(triton.cdiv(n,256),)](x,y,n,256)
    return y


@triton.jit
def _qk_rope_cache(X, QW, KW, COS, SIN, Q, KC, VC, POS, L: tl.constexpr, EPS: tl.constexpr):
    h=tl.program_id(0)
    d=tl.arange(0,128)
    swap=(d+64)%128
    p=tl.load(POS)
    if h<32:
        base=h*128
        w=tl.load(QW+d).to(tl.float32)
        ws=tl.load(QW+swap).to(tl.float32)
    else:
        base=4096+(h-32)*128
        w=tl.load(KW+d).to(tl.float32)
        ws=tl.load(KW+swap).to(tl.float32)
    x=tl.load(X+base+d).to(tl.float32)
    xs=tl.load(X+base+swap).to(tl.float32)
    r=tl.rsqrt(tl.sum(x*x,0)/128+EPS)
    x=((x*r).to(tl.bfloat16).to(tl.float32)*w).to(tl.bfloat16).to(tl.float32)
    xs=((xs*r).to(tl.bfloat16).to(tl.float32)*ws).to(tl.bfloat16).to(tl.float32)
    c=tl.load(COS+d).to(tl.float32)
    s=tl.load(SIN+d).to(tl.float32)
    rotated=(x*c).to(tl.bfloat16).to(tl.float32)+(tl.where(d<64,-xs,xs)*s).to(tl.bfloat16).to(tl.float32)
    if h<32:
        tl.store(Q+h*128+d,rotated)
    else:
        kh=h-32
        tl.store(KC+(kh*L+p)*128+d,rotated)
        v=tl.load(X+5120+kh*128+d)
        tl.store(VC+(kh*L+p)*128+d,v)


@triton.jit
def _decode_attn(Q,K,V,POS,PART,LSE,L:tl.constexpr,SPLITS:tl.constexpr,BLOCK:tl.constexpr):
    h=tl.program_id(0)
    split=tl.program_id(1)
    p=tl.load(POS)
    t=split*BLOCK+tl.arange(0,BLOCK)
    d=tl.arange(0,128)
    if split*BLOCK<=p:
        q=tl.load(Q+h*128+d).to(tl.float32)
        k=tl.load(K+((h//4)*L+t[:,None])*128+d[None,:],t[:,None]<=p,0).to(tl.float32)
        logits=tl.sum(k*q[None,:],1)*0.08838834764831845
        logits=tl.where(t<=p,logits,float('-inf'))
        m=tl.max(logits,0)
        prob=tl.exp(logits-m)
        denom=tl.sum(prob,0)
        v=tl.load(V+((h//4)*L+t[:,None])*128+d[None,:],t[:,None]<=p,0).to(tl.float32)
        result=tl.sum(prob[:,None]*v,0)/denom
        logsum=m+tl.log(denom)
    else:
        result=tl.full((128,),0,tl.float32)
        logsum=float('-inf')
    tl.store(PART+(h*SPLITS+split)*128+d,result)
    tl.store(LSE+h*SPLITS+split,logsum)


@triton.jit
def _decode_attn_reduce(PART,LSE,OUT,SPLITS:tl.constexpr,BS:tl.constexpr):
    h=tl.program_id(0)
    s=tl.arange(0,BS)
    d=tl.arange(0,128)
    lse=tl.load(LSE+h*SPLITS+s,s<SPLITS,float('-inf'))
    a=tl.exp(lse-tl.max(lse,0))
    a=a/tl.sum(a,0)
    val=tl.load(PART+(h*SPLITS+s[:,None])*128+d[None,:],s[:,None]<SPLITS,0)
    out=tl.sum(a[:,None]*val,0)
    tl.store(OUT+h*128+d,out)


def qk_rope_decode(qkv, q_weight, k_weight, cos, sin, cache, layer_idx, position, eps, backend=None, block=128, warps=4,context_capacity=None,quantize_output=False,native_attention=False,pdl=None):
    kc,vc=cache.layers[layer_idx].keys,cache.layers[layer_idx].values
    length=kc.shape[-2]
    q=torch.empty((32,128),device=qkv.device,dtype=qkv.dtype)
    if pdl:
        if not native_attention or quantize_output!=True or backend is not None or (block,warps)!=(32,4):
            raise ValueError('Attention PDL requires native B32/W4 and G32 quantization')
        from .qk_rope_pdl import launch
        launch(qkv,q_weight,k_weight,cos,sin,q,kc,vc,position,eps,
               pdl=True,trigger=pdl['qk'],preload=pdl['preload'])
    else:
        _qk_rope_cache[(40,)](qkv,q_weight,k_weight,cos,sin,q,kc,vc,position,length,eps,enable_fp_fusion=False)
    if backend is not None:
        if native_attention:raise ValueError('Native attention conflicts with another backend')
        if quantize_output:raise ValueError('Fused output quantization requires custom attention')
        return backend(q,kc,vc,position)
    capacity=length if context_capacity is None else context_capacity
    if not 0<capacity<=length or capacity%block:raise ValueError('Invalid decode attention capacity')
    splits=triton.cdiv(capacity,block)
    partial=torch.empty((32,splits,128),device=q.device,dtype=torch.float32)
    lse=torch.empty((32,splits),device=q.device,dtype=torch.float32)
    if native_attention:
        if (block,warps)!=(32,4):raise ValueError('Native attention requires B32/W4')
        if pdl:
            from .attention_pdl import launch
            launch(q,kc,vc,position,partial,lse,pdl=True,trigger=pdl['attention'])
        else:
            from .attention_native import launch
            launch(q,kc,vc,position,partial,lse)
    else:
        _decode_attn[(32,splits)](q,kc,vc,position,partial,lse,length,splits,block,num_warps=warps)
    if quantize_output:
        if pdl:
            from .attention_quant_pdl import reduce_quant
            return reduce_quant(partial,lse,pdl=True,trigger=pdl['reduce'])
        else:
            from .attention_quant import reduce_quant
            return reduce_quant(partial,lse)
    out=torch.empty((1,1,4096),device=q.device,dtype=q.dtype)
    _decode_attn_reduce[(32,)](partial,lse,out,splits,triton.next_power_of_2(splits),num_warps=4)
    return out


@triton.jit
def _codec_rope_cache(X,CS,Q,KC,VC,OFFSET,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr):
    h=tl.program_id(0)
    t=tl.program_id(1)
    d=tl.arange(0,64)
    pair=d//2
    swap=d^1
    offset=tl.load(OFFSET)
    cos=tl.load(CS+(offset+t)*64+pair)
    sin=tl.load(CS+(offset+t)*64+32+pair)
    b=t*3*H*64+h*64
    qr=tl.load(X+b+d).to(tl.float32)
    qi=tl.load(X+b+swap).to(tl.float32)
    kr=tl.load(X+b+H*64+d).to(tl.float32)
    ki=tl.load(X+b+H*64+swap).to(tl.float32)
    qo=qr*cos+tl.where(d%2==0,-qi,qi)*sin
    ko=kr*cos+tl.where(d%2==0,-ki,ki)*sin
    v=tl.load(X+b+2*H*64+d)
    tl.store(Q+(h*T+t)*64+d,qo)
    cache_pos=(offset+t)%C
    tl.store(KC+(h*C+cache_pos)*64+d,ko)
    tl.store(VC+(h*C+cache_pos)*64+d,v)


@triton.jit
def _codec_attention(Q,K,V,OFFSET,OUT,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,BC:tl.constexpr):
    h=tl.program_id(0)
    t=tl.program_id(1)
    i=tl.arange(0,BC)
    d=tl.arange(0,64)
    offset=tl.load(OFFSET)
    end=offset+T-1
    delta=i-end%C
    pos=end+delta-tl.where(delta<=0,0,C)
    valid=(i<C)&(pos>=0)&(pos<=offset+t)&(offset+t-pos<C)
    q=tl.load(Q+(h*T+t)*64+d).to(tl.float32)
    k=tl.load(K+(h*C+i[:,None])*64+d[None,:],valid[:,None],0).to(tl.float32)
    a=tl.sum(k*q[None,:],1)*0.125
    a=tl.where(valid,a,float('-inf'))
    p=tl.exp(a-tl.max(a,0))
    p=p/tl.sum(p,0)
    v=tl.load(V+(h*C+i[:,None])*64+d[None,:],valid[:,None],0).to(tl.float32)
    out=tl.sum(p[:,None]*v,0)
    tl.store(OUT+(t*H+h)*64+d,out)


def codec_attention(projected,cos_sin,state,heads):
    _,t,_=projected.shape
    capacity=state.kv_cache.capacity
    q=torch.empty((heads,t,64),device=projected.device,dtype=projected.dtype)
    out=torch.empty((1,t,heads*64),device=projected.device,dtype=projected.dtype)
    k,v=state.kv_cache.cache[0],state.kv_cache.cache[1]
    _codec_rope_cache[(heads,t)](projected,cos_sin,q,k,v,state.offset,t,heads,capacity,enable_fp_fusion=False)
    _codec_attention[(heads,t)](q,k,v,state.offset,out,t,heads,capacity,triton.next_power_of_2(capacity),num_warps=8)
    state.offset.add_(t)
    return out
