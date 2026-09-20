"""Opt-in eight-row down projection for the selected clustered decode path."""
from .short_scales import PLAN


def make_dispatch():
    from .bulk_prefetch import configured
    fn = configured('projection', divisor=16)
    tile = {**PLAN['down'], 'rows': 8, 'warps': 4, 'prefetch': 1}
    def dispatch(*args, **kwargs):
        return fn(*args, **{**kwargs, **tile})
    return dispatch


def enable(fast):
    if fast.graph is not None or fast.prefill_graphs or getattr(fast, 'down_tile8', None):
        raise RuntimeError('Install down tile once, before graph capture')
    if fast.cfg.n_vq != 32 or not getattr(fast, 'qkv_cluster', None):
        raise ValueError('The selected 32-codebook clustered QKV preset is required')
    if not getattr(fast, 'bulk_prefetch', None) or not getattr(fast, 'output_weight_prefetch', None):
        raise ValueError('Selected bulk and output register-prefetch paths required')
    if getattr(fast, 'projection_pdl', None) != {
            'norm_trigger': 1, 'projection_trigger': 3, 'scale_prefetch': True}:
        raise ValueError('The selected projection-PDL schedule is required')
    modules = [layer.mlp for layer in fast.model.language_model.layers]
    if any(getattr(m, '_dp4a_group', None) != 32 or
           getattr(m, '_short_scale_projection', None) != 'down' or
           not getattr(m, '_bulk_down_linear', None) for m in modules):
        raise ValueError('Selected G32 short-scale down projections required')
    fn = make_dispatch()
    for module in modules:
        module._bulk_down_linear = fn
    fast.down_tile8 = {'codebooks': 32, 'rows': 8, 'warps': 4,
        'integer_groups': 1, 'integer_rows': 1, 'prefetch': 1,
        'bulk_divisor': 16, 'trigger_mode': 3, 'projections': len(modules)}
    return dict(fast.down_tile8)
