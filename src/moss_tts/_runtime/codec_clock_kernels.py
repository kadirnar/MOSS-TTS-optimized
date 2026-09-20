"""FP32 codec attention with a single immutable frame counter per decode."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope(X,CS,Q,KC,VC,OFFSET,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,FIRST:tl.constexpr,KV_ONLY:tl.constexpr):
    h=tl.program_id(0)
    t=tl.program_id(1)
    d=tl.arange(0,64)
    pair=d//2
    swap=d^1
    offset=0 if FIRST else tl.load(OFFSET)*T
    cos=tl.load(CS+(offset+t)*64+pair)
    sin=tl.load(CS+(offset+t)*64+32+pair)
    b=t*(2 if KV_ONLY else 3)*H*64+h*64
    kr=tl.load(X+b+(0 if KV_ONLY else H*64)+d).to(tl.float32)
    ki=tl.load(X+b+(0 if KV_ONLY else H*64)+swap).to(tl.float32)
    ko=kr*cos+tl.where(d%2==0,-ki,ki)*sin
    v=tl.load(X+b+(1 if KV_ONLY else 2)*H*64+d)
    if FIRST and T==1:
        # Softmax of the sole valid position is one: attention output is V.
        tl.store(Q+h*64+d,v)
    else:
        qr=tl.load(X+b+d).to(tl.float32)
        qi=tl.load(X+b+swap).to(tl.float32)
        qo=qr*cos+tl.where(d%2==0,-qi,qi)*sin
        tl.store(Q+(h*T+t)*64+d,qo)
    cache_pos=(offset+t)%C
    tl.store(KC+(h*C+cache_pos)*64+d,ko)
    tl.store(VC+(h*C+cache_pos)*64+d,v)


@triton.jit
def _attention(Q,K,V,OFFSET,OUT,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,BC:tl.constexpr,FIRST:tl.constexpr):
    h=tl.program_id(0)
    t=tl.program_id(1)
    i=tl.arange(0,BC)
    d=tl.arange(0,64)
    offset=0 if FIRST else tl.load(OFFSET)*T
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



def attention(projected,cos_sin,state,heads,clock,*,first=False,kv_only=False,compact_first=True):
    _,t,_=projected.shape
    if kv_only and not (first and t==1):raise ValueError('KV-only projection requires the initial single-token stage')
    capacity=state.kv_cache.capacity
    q=torch.empty((heads,t,64),device=projected.device,dtype=projected.dtype)
    out=torch.empty((1,t,heads*64),device=projected.device,dtype=projected.dtype)
    k,v=state.kv_cache.cache[0],state.kv_cache.cache[1]
    destination=out if first and t==1 else q
    _rope[(heads,t)](projected,cos_sin,destination,k,v,clock,t,heads,capacity,first,kv_only,enable_fp_fusion=False)
    if not (first and t==1):
        _attention[(heads,t)](q,k,v,clock,out,t,heads,capacity,triton.next_power_of_2(t if first and compact_first else capacity),first,num_warps=8)
    return out


def stage_attention(projected,cos_sin,state,heads,clock,*,first=False,kv_only=False,compact_first=False):
    """Reuse original steady-state kernels with a shared per-stage offset."""
    from .kernels import _codec_rope_cache,_codec_attention
    _,t,_=projected.shape
    capacity=state.kv_cache.capacity
    q=torch.empty((heads,t,64),device=projected.device,dtype=projected.dtype)
    out=torch.empty((1,t,heads*64),device=projected.device,dtype=projected.dtype)
    k,v=state.kv_cache.cache[0],state.kv_cache.cache[1]
    if first and t==1:
        _rope[(heads,t)](projected,cos_sin,out,k,v,clock,t,heads,capacity,True,kv_only,enable_fp_fusion=False)
    else:
        if kv_only:raise ValueError('KV-only projection requires the first single-token stage')
        _codec_rope_cache[(heads,t)](projected,cos_sin,q,k,v,clock,t,heads,capacity,enable_fp_fusion=False)
        _codec_attention[(heads,t)](q,k,v,clock,out,t,heads,capacity,triton.next_power_of_2(t if first and compact_first else capacity),num_warps=8)
    return out
