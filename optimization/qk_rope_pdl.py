"""Exact QK normalization/rotary cache update with optional independent norm loads."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait,gdc_launch_dependents


@triton.jit
def _qk_rope_cache(X, QW, KW, COS, SIN, Q, KC, VC, POS, L: tl.constexpr, EPS: tl.constexpr,PDL:tl.constexpr,TRIGGER:tl.constexpr,PRE:tl.constexpr):
    if PDL and not PRE:gdc_wait()
    if PDL and not PRE and TRIGGER==1:gdc_launch_dependents()
    h=tl.program_id(0)
    d=tl.arange(0,128)
    swap=(d+64)%128
    if h<32:
        base=h*128
        w=tl.load(QW+d).to(tl.float32)
        ws=tl.load(QW+swap).to(tl.float32)
    else:
        base=4096+(h-32)*128
        w=tl.load(KW+d).to(tl.float32)
        ws=tl.load(KW+swap).to(tl.float32)
    if PDL and PRE:gdc_wait()
    if PDL and PRE and TRIGGER==1:gdc_launch_dependents()
    p=tl.load(POS)
    x=tl.load(X+base+d).to(tl.float32)
    xs=tl.load(X+base+swap).to(tl.float32)
    r=tl.rsqrt(tl.sum(x*x,0)/128+EPS)
    x=((x*r).to(tl.bfloat16).to(tl.float32)*w).to(tl.bfloat16).to(tl.float32)
    xs=((xs*r).to(tl.bfloat16).to(tl.float32)*ws).to(tl.bfloat16).to(tl.float32)
    if PDL and TRIGGER==2:gdc_launch_dependents()
    c=tl.load(COS+d).to(tl.float32)
    s=tl.load(SIN+d).to(tl.float32)
    rotated=(x*c).to(tl.bfloat16).to(tl.float32)+(tl.where(d<64,-xs,xs)*s).to(tl.bfloat16).to(tl.float32)
    if PDL and TRIGGER==3:gdc_launch_dependents()
    if h<32:
        tl.store(Q+h*128+d,rotated)
    else:
        kh=h-32
        tl.store(KC+(kh*L+p)*128+d,rotated)
        v=tl.load(X+5120+kh*128+d)
        tl.store(VC+(kh*L+p)*128+d,v)


def launch(qkv,q_weight,k_weight,cos,sin,q,kc,vc,position,eps,*,pdl=True,trigger=1,preload=False):
    if trigger not in (0,1,2,3):raise ValueError('Invalid PDL trigger')
    return _qk_rope_cache[(40,)](qkv,q_weight,k_weight,cos,sin,q,kc,vc,position,kc.shape[-2],eps,
                               pdl,trigger,preload,enable_fp_fusion=False,launch_pdl=pdl)
