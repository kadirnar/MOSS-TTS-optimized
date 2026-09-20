"""Opt-in exact G32 attention-output weight staging for the SM90 PDL path."""
from .dp4a_async_weights import configured


def make_dispatch(rows=8,register_preload=False):
    if rows not in (8,16):raise ValueError('Supported experiment row counts are 8 or 16')
    if register_preload:
        from .dp4a_layout_pdl_prefetch import linear as fn
    else:fn=configured(mode=1,swizzle=8)
    def dispatch(*args,**kwargs):
        return fn(*args,**{**kwargs,'rows':rows,'warps':4,**({'prefetch':3} if register_preload else {})})
    return dispatch


def enable(llm,*,register_preload=False):
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Install async output before graph capture')
    if getattr(llm,'async_output',None) or getattr(llm,'output_weight_prefetch',None):raise ValueError('Install one output staging strategy once')
    if not getattr(llm,'short_scales',None) or getattr(llm,'projection_pdl',None)!={'norm_trigger':1,'projection_trigger':3,'scale_prefetch':True}:
        raise ValueError('Async output requires selected G32 short scales and projection PDL')
    if getattr(llm,'attention_pdl',None)!={'pdl':True,'qk':1,'attention':2,'reduce':1,'preload':False}:
        raise ValueError('Selected attention PDL schedule required')
    modules=[layer.self_attn for layer in llm.model.language_model.layers]
    if any(getattr(m,'_dp4a_group',None)!=32 or getattr(m,'_short_scale_projection',None)!='out' for m in modules):
        raise ValueError('Selected G32 attention-output projections required')
    fn=make_dispatch(16 if register_preload else 8,register_preload=register_preload)
    for module in modules:module._async_out_linear=fn
    metadata={'copy':'register_load' if register_preload else 'cp.async','shared_swizzle':None if register_preload else 8,
              'rows':16 if register_preload else 8,'warps':4,'projections':len(modules),'weight_group':32}
    setattr(llm,'output_weight_prefetch' if register_preload else 'async_output',metadata)
    return dict(metadata)
