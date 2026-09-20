"""Install an environment-gated hook in the isolated vLLM-Omni environment."""
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.find_spec('vllm_omni')
target = Path(next(iter(spec.submodule_search_locations))) / 'model_executor/models/moss_tts/modeling_moss_tts_codec.py'
marker = '# MOSS_V1_OPTIMIZED_CODEC adapter hook'
hook = '''
# MOSS_V1_OPTIMIZED_CODEC adapter hook
import os as _moss_adapter_os
if _moss_adapter_os.environ.get("MOSS_V1_OPTIMIZED_CODEC") == "1":
    from optimization.vllm_codec_adapter import install as _install_moss_codec
    _install_moss_codec(MossTTSCodecDecoder)
'''
original = target.read_text()
if marker not in original:
    backup = target.with_suffix('.py.pre_moss_adapter')
    if backup.exists():
        raise RuntimeError(f'Refusing to overwrite existing backup {backup}')
    backup.write_text(original)
    target.write_text(original+hook)
print(json.dumps({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'enabled_only_with':'MOSS_V1_OPTIMIZED_CODEC=1'}))
target = Path(next(iter(spec.submodule_search_locations))) / 'model_executor/stage_input_processors/moss_tts.py'
original = target.read_text()
marker = '# MOSS_V1_OPTIMIZED_CODEC delay streaming hook'
hook = '''
# MOSS_V1_OPTIMIZED_CODEC delay streaming hook
import os as _moss_stream_os
if _moss_stream_os.environ.get("MOSS_V1_OPTIMIZED_CODEC") == "1":
    from optimization.vllm_delay_stream import talker2codec_delay_async_chunk
'''
if marker not in original:
    backup = target.with_suffix('.py.pre_moss_adapter')
    if backup.exists():
        raise RuntimeError(f'Refusing to overwrite existing backup {backup}')
    backup.write_text(original)
    target.write_text(original+hook)
print(json.dumps({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'enabled_only_with':'MOSS_V1_OPTIMIZED_CODEC=1'}))
