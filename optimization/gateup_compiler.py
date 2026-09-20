"""Opt-in exact gate/up cubins compiled by isolated Triton 3.8.

Uses one CTA, IG1/IR2 and explicit legacy reduction order. Other kernels and
the host remain on the selected runtime. Install before all graph capture.
"""
import hashlib
import json
from .common import RESULTS


def make_dispatch(original):
    from .gateup_cluster_binary import load_bundle
    folder = RESULTS/'gateup_exact_bundle_v1'
    choices, launchers = load_bundle(folder)
    name = 'c1_ig1ir2'; options = choices[name]; launch = launchers[name]
    def dispatch(*args, **kwargs):
        if not kwargs.get('fused', False):
            return original(*args, **kwargs)
        for key, value in (('rows', 32), ('integer_groups', 2), ('integer_rows', 4), ('pdl', True), ('trigger_mode', 1)):
            if kwargs.pop(key, value) != value:
                raise ValueError('Selected gate/up producer configuration required: '+key)
        kwargs.pop('fused')
        return launch(*args, **options, **kwargs)
    return dispatch


def enable(fast):
    import triton
    if fast.graph is not None or fast.prefill_graphs or getattr(fast, 'gateup_compiler', None):
        raise RuntimeError('Install gate/up compiler option once, before graph capture')
    if fast.cfg.n_vq != 32 or not getattr(fast, 'down_tile8', None) or not getattr(fast, 'qkv_cluster', None):
        raise ValueError('Selected 32-codebook clustered-QKV and eight-row-down preset required')
    if triton.__version__ != '3.7.1':
        raise ValueError('Only the qualified Triton 3.7.1 host is supported')
    if getattr(fast, 'projection_pdl', None) != {'norm_trigger': 1, 'projection_trigger': 3, 'scale_prefetch': True}:
        raise ValueError('Selected projection-PDL schedule required')
    modules = [layer.mlp for layer in fast.model.language_model.layers]
    if any(getattr(m, '_dp4a_group', None) != 32 for m in modules):
        raise ValueError('Selected G32 gate/up weights required')
    original = getattr(fast, '_bulk_norm_linear', None)
    if original is None:
        raise ValueError('Selected bulk norm/projection path required')
    fast._bulk_norm_linear = make_dispatch(original)
    folder = RESULTS/'gateup_exact_bundle_v1'
    manifest = json.loads((folder/'manifest.json').read_text())
    fast.gateup_compiler = {'codebooks': 32, 'compiler': manifest['triton'], 'host_triton': triton.__version__,
        'ctas': 1, 'rows': 32, 'integer_groups': 1, 'integer_rows': 2, 'bulk_divisor': 16,
        'trigger_mode': 1, 'projections': len(modules), 'bundle': str(folder),
        'manifest_sha256': hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
        'cubins': {n: r['cubin_sha256'] for n, r in manifest['binaries'].items()}}
    return dict(fast.gateup_compiler)
