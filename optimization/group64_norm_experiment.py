"""Private-benchmark G64 norm/projection dispatch, retaining original buffers.

No serving entry point selects this changed-precision model. Each dispatch owns
its replacement buffers so control and candidate CUDA graphs can coexist.
"""
import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_group64_a64 import norm_linear


@torch.inference_mode()
def prepare(llm,stages=('up','qkv')):
    if not stages or not set(stages)<= {'up','qkv'}:raise ValueError('Norm projections only')
    if not getattr(llm,'bulk_prefetch',None):raise ValueError('Selected bulk/PDL preset required')
    original=llm._bulk_norm_linear;replacements={}
    for i,layer in enumerate(llm.model.language_model.layers):
        for name in stages:
            module=layer.mlp if name=='up' else layer.self_attn
            assert module._dp4a_group==32
            old_w=getattr(module,'_quant_'+name);old_s=getattr(module,'_quant_'+name+'_scale')
            saved=torch.load(RESULTS/f'gptq_v1_g64_diag_d10/{i:02d}_{name}.pt',map_location=old_w.device,weights_only=True)
            assert saved['group']==64 and saved['packed'].shape==old_w.shape
            w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16()
            assert s.shape==(old_s.shape[0],old_s.shape[1]//2)
            replacements[old_w.data_ptr()]=(w,s)
    def dispatch(x,res,nw,eps,w,s,**kwargs):
        pair=replacements.get(w.data_ptr())
        if pair is None:return original(x,res,nw,eps,w,s,**kwargs)
        return norm_linear(x,res,nw,eps,*pair,activation_group=32,output_group=32,factor=1,**kwargs)
    dispatch.replacements=replacements
    dispatch.metadata={'weight_group':64,'activation_group':32,'factor':1,'stages':list(stages),
        'replaced_projections':len(replacements),'codebooks':32,'experimental':True}
    return dispatch


@torch.inference_mode()
def enable(llm,stages=('up','qkv')):
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Install before graph capture')
    if getattr(llm,'group64_norm_experiment',None):raise ValueError('Install once')
    dispatch=prepare(llm,stages)
    llm._bulk_norm_linear=dispatch;llm.group64_norm_experiment=dispatch.metadata
    return dict(dispatch.metadata)
