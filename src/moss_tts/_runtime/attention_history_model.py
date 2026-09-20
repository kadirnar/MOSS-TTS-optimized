"""Opt-in per-model historical-KV preloads, leaving global functions intact."""
import types
from .qkv_cluster_model import Hidden as BaseHidden
from .attention_history import launch


def enable(fast):
    import hashlib
    import json
    from pathlib import Path
    import triton
    from .paths import ASSETS
    from .qkv_cluster_binary import load_bundle
    from .attention_history import library
    if fast.graph is not None or fast.prefill_graphs or getattr(fast, 'attention_history', None):
        raise RuntimeError('Install historical preloads once before graph capture')
    if fast.cfg.n_vq != 32 or not getattr(fast, 'gateup_compiler', None) or not getattr(fast, 'down_tile8', None):
        raise ValueError('Selected 32-codebook gate/up compiler and eight-row-down preset required')
    if triton.__version__ != '3.7.1' or not getattr(fast, 'qkv_cluster', None):
        raise ValueError('Selected Triton 3.7.1 clustered-QKV host required')
    if fast.attention_pdl != {'pdl': True, 'qk': 1, 'attention': 2, 'reduce': 1, 'preload': False}:
        raise ValueError('Selected attention-PDL baseline required')
    folder = ASSETS / 'qkv_cluster_bundle_history_v1'
    options, launchers = load_bundle(folder); key = 'c8_exact'
    expected = {'ctas': 8, 'divisor': 16, 'legacy_projection': True, 'legacy_norm': True}
    if options[key] != expected:
        raise ValueError('Unexpected early-producer specialization')
    library()
    config = {'producer': 1, 'trigger': 1, 'mode': 3, 'packed': False}
    fast.hidden = Hidden(fast, launchers[key], options[key], config)
    manifest = json.loads((folder / 'manifest.json').read_text())
    source = Path(__file__).with_name('attention_history.cu')
    native_hash = hashlib.sha256(source.read_bytes() + source.with_name('attention_pdl.cu').read_bytes()).hexdigest()
    fast.attention_history = {'codebooks': 32, **config, 'native_source_pair_sha256': native_hash,
        'native_build': native_hash[:16], 'producer_bundle': str(folder),
        'producer_manifest_sha256': hashlib.sha256((folder / 'manifest.json').read_bytes()).hexdigest(),
        'producer_cubins': {n: r['cubin_sha256'] for n, r in manifest['binaries'].items()},
        'scope': 'Historical t<position K/V preloads only; Q and current-token reads retain dependency wait. '
                 'Prefill and the sequential single-request cache ownership contract remain unchanged.'}
    return dict(fast.attention_history)


class Hidden(BaseHidden):
    def __init__(self, fast, launcher, options, config):
        super().__init__(fast, launcher, options)
        self.config = None if config is None else dict(config)
        if config is None:
            self.call = super().__call__
        else:
            def attention(*args, pdl=True, trigger=2):
                if not pdl:
                    raise ValueError('Selected PDL model required')
                return launch(*args, mode=config['mode'], packed=config['packed'], trigger=config['trigger'])
            original = BaseHidden.__call__
            function = types.FunctionType(original.__code__, {**original.__globals__, 'attention': attention},
                original.__name__, original.__defaults__, original.__closure__)
            function.__kwdefaults__ = original.__kwdefaults__
            self.call = types.MethodType(function, self)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
