"""Install the selected exact projection PDL preset before graph capture."""
import torch


def enable(llm, *, prefetch=True):
    if llm.graph is not None or llm.prefill_graphs:
        raise RuntimeError('Install projection PDL before graph capture')
    if not getattr(llm,'short_scales',None) or set(getattr(llm,'norm_projection_fused',{}))!={'qkv','up'}:
        raise ValueError('Projection PDL requires the exact G32 short-scale norm-projection preset')
    if getattr(llm,'qkv_load_policy',False):
        raise ValueError('Projection PDL and the experimental QKV load policy are separate presets')
    modules=[module for layer in llm.model.language_model.layers for module in (layer.self_attn,layer.mlp)]
    for module in modules:
        if getattr(module,'_short_scale_projection',None) not in ('out','down'):
            raise ValueError('All output/down modules must have exact BF16 short scales')
    device=modules[0]._quant_out.device
    if torch.cuda.get_device_capability(device)!=(9,0):
        raise ValueError('This projection PDL preset has been validated only on SM90')
    config={'norm_trigger':1,'projection_trigger':3,'scale_prefetch':bool(prefetch)}
    llm.projection_pdl=config
    for module in modules:module._projection_pdl=dict(config)
    return dict(config)
