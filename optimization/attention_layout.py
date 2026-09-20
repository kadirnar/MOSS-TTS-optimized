"""Split decode attention with explicit register layouts.

Keep the selected 32-token reduction partition while loading Q and storing
partials directly in their consumer/producer layouts. This removes otherwise
unnecessary shared-memory layout conversions. No codebooks are removed.
"""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _attention(Q,K,V,POS,PART,LSE,L:gl.constexpr,SPLITS:gl.constexpr,
               BLOCK:gl.constexpr,WARPS:gl.constexpr,Q_DIRECT:gl.constexpr,
               OUT_DIRECT:gl.constexpr):
    layout:gl.constexpr=gl.BlockedLayout([1,8],[2,16],[WARPS,1],[1,0])
    vector:gl.constexpr=gl.BlockedLayout([1],[32],[WARPS],[0])
    h=gl.program_id(0);split=gl.program_id(1);p=gl.load(POS)
    t=split*BLOCK+gl.arange(0,BLOCK,layout=gl.SliceLayout(1,layout))
    d=gl.arange(0,128,layout=gl.SliceLayout(0,layout))
    if split*BLOCK<=p:
        if Q_DIRECT:
            q=gl.load(Q+h*128+d).to(gl.float32)
        else:
            dq=gl.arange(0,128,layout=vector)
            q=gl.convert_layout(gl.load(Q+h*128+dq),gl.SliceLayout(0,layout)).to(gl.float32)
        k=gl.load(K+((h//4)*L+t[:,None])*128+d[None,:],t[:,None]<=p,0).to(gl.float32)
        logits=gl.sum(k*q[None,:],1)*0.08838834764831845
        logits=gl.where(t<=p,logits,float('-inf'))
        m=gl.max(logits,0);prob=gl.exp(logits-m);denom=gl.sum(prob,0)
        v=gl.load(V+((h//4)*L+t[:,None])*128+d[None,:],t[:,None]<=p,0).to(gl.float32)
        result=gl.sum(prob[:,None]*v,0)/denom
        logsum=m+gl.log(denom)
    else:
        result=gl.full((128,),0,gl.float32,layout=gl.SliceLayout(0,layout))
        logsum=float('-inf')
    if OUT_DIRECT:
        gl.store(PART+(h*SPLITS+split)*128+d,result)
    else:
        dout=gl.arange(0,128,layout=vector)
        gl.store(PART+(h*SPLITS+split)*128+dout,gl.convert_layout(result,vector))
    gl.store(LSE+h*SPLITS+split,logsum)


def launch(q,k,v,position,partial,lse,*,q_direct=True,out_direct=True,warps=4):
    splits=partial.shape[1]
    return _attention[(32,splits)](q,k,v,position,partial,lse,k.shape[-2],splits,32,
        warps,q_direct,out_direct,num_warps=warps)
