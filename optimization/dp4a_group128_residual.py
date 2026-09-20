"""Experimental G128 output/down projections with the selected G32 consumers.

Only residual-branch output projections change calibration. QKV, gate/up,
normalization, activation groups, attention and all 32 audio heads retain the
selected G32 implementation. Install after the G32 preset, before capture.
"""
import hashlib
import json
from pathlib import Path

import torch

from .common import TTS_REVISION
from .dp4a_packing import pack_interleaved


@torch.inference_mode()
def install(llm, folder, plan_path):
    folder, plan_path = Path(folder), Path(plan_path)
    if llm.graph is not None or llm.prefill_graphs:
        raise RuntimeError('Install residual G128 projections before capture')
    config = json.loads((folder / 'config.json').read_text())
    plan = json.loads(plan_path.read_text())
    if config['tts_revision'] != TTS_REVISION or config['group'] != 128:
        raise ValueError('Matching calibrated G128 export required')
    if plan.get('format') != 'group128_v1' or plan.get('activation_group') != 32 or plan.get('codebooks') != 32:
        raise ValueError('An all-32-codebook G128/A32 plan is required')
    if set(getattr(llm, 'norm_projection_fused', {})) != {'qkv', 'up'} or getattr(llm, 'group128_norm_fused', None):
        raise ValueError('Selected G32 norm/consumer preset required')
    targets = []
    for i, layer in enumerate(llm.model.language_model.layers):
        for module, name in ((layer.self_attn, 'out'), (layer.mlp, 'down')):
            if (not getattr(module, '_use_dp4a', False)
                    or getattr(module, '_dp4a_group', None) != 32
                    or not getattr(module, '_scaled_dp4a', None)
                    or getattr(module, '_group128_plan', None)):
                raise ValueError('Fresh scaled G32 modules required')
            cfg = plan['projections'][name]
            if (cfg['fused'] or cfg['rows'] not in (1,2,4,8,16,32,64,128)
                    or cfg['warps'] not in (1,2,4,8,16) or cfg['mode'] not in (1,2)):
                raise ValueError('Invalid residual projection configuration')
            path = folder / f'{i:02d}_{name}.pt'
            if not path.is_file():
                raise ValueError(f'Missing calibrated projection: {path}')
            targets.append((module, name, path, dict(cfg)))
    # Check CPU tensors before replacing any GPU inference buffers.
    staged = []
    for module, name, path, cfg in targets:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        n, k2 = getattr(module, '_quant_' + name).shape
        if (saved['group'] != 128 or saved['shape'] != [n, k2*2]
                or saved['packed'].shape != (n,k2) or saved['packed'].dtype != torch.uint8
                or saved['scales'].shape != (n,k2*2//128)
                or saved['scales'].dtype != torch.bfloat16
                or not torch.isfinite(saved['scales']).all() or not (saved['scales'] > 0).all()):
            raise ValueError(f'Invalid calibrated projection: {path}')
        staged.append((module, name, saved, cfg))
    for module, name, saved, cfg in staged:
        device = getattr(module, '_quant_' + name).device
        setattr(module, '_quant_' + name, pack_interleaved(saved['packed'].to(device)))
        setattr(module, '_quant_' + name + '_scale', saved['scales'].to(device))
        module._group128_plan = {name: cfg}
    return {'weight_groups': {'qkv':32,'out':128,'up':32,'down':128},
            'activation_group':32, 'codebooks':32, 'export':str(folder),
            'calibration_config':config, 'plan_path':str(plan_path),
            'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            'projections':{name:plan['projections'][name] for name in ('out','down')}}
