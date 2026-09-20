"""Load an explicitly selected GPTQ export into an already fused backbone."""
import json
from pathlib import Path
import torch
from ..models import TTS_REVISION


def unpack_signed(packed):
    lo=(packed&15).to(torch.int8)
    hi=(packed>>4).to(torch.int8)
    lo=torch.where(lo>=8,lo-16,lo)
    hi=torch.where(hi>=8,hi-16,hi)
    return torch.stack([lo,hi],dim=-1).reshape(packed.shape[0],-1)


def enable_dp4a_reciprocal(llm):
    """Select the separately validated reciprocal-rounding quantizer pre-capture."""
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Select quantizer before capture')
    modules=[m for layer in llm.model.language_model.layers for m in (layer.self_attn,layer.mlp)]
    if not all(getattr(m,'_use_dp4a',False) for m in modules):raise ValueError('DP4A backbone required')
    for m in modules:m._dp4a_reciprocal=True


def enable_grouped_activation(llm):
    enable_dp4a_reciprocal(llm)
    for layer in llm.model.language_model.layers:
        for m in (layer.self_attn,layer.mlp):m._dp4a_grouped_activation=True


@torch.inference_mode()
def install_calibrated(llm,folder,backend='dp4a'):
    folder=Path(folder)
    config=json.loads((folder/'config.json').read_text())
    if config['tts_revision']!=TTS_REVISION:raise ValueError('Calibrated checkpoint revision mismatch')
    if backend != 'dp4a':raise ValueError('Unsupported calibrated backend')
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Install calibrated projections before graph capture')
    # Validate completeness before changing the model's inference buffers.
    paths={(i,p):folder/f'{i:02d}_{p}.pt' for i in range(36) for p in ('qkv','out','up','down')}
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing:raise ValueError(f'Incomplete calibrated export: {len(missing)} projections missing')
    for i,layer in enumerate(llm.model.language_model.layers):
        targets=[(layer.self_attn,'qkv',layer.self_attn._qkv),
            (layer.self_attn,'out',layer.self_attn.o_proj.weight),
            (layer.mlp,'up',layer.mlp._gate_up),(layer.mlp,'down',layer.mlp.down_proj.weight)]
        for module,name,weight in targets:
            saved=torch.load(paths[i,name],map_location=weight.device,weights_only=True)
            if list(weight.shape)!=saved['shape'] or saved['group']!=config['group']:
                raise ValueError(f'Projection shape/group mismatch: {i} {name}')
            packed,scales=saved['packed'],saved['scales']
            module.register_buffer('_quant_'+name,packed)
            module.register_buffer('_quant_'+name+'_scale',scales.float())
            module._use_quantized=True
            module._use_dp4a=True
            module._dp4a_group=config['group']
    llm.weight_quantization='gptq_'+backend
    return config
