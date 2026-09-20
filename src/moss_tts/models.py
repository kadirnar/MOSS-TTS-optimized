"""Pinned 8B checkpoints loaded through bundled model and codec classes."""

from pathlib import Path

MODEL_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
CODEC_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
TTS_REVISION = "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"
CODEC_REVISION = "3cd226ba2947efa357ef453bcad111b6eafba782"


def load_models(*, cache_dir: str | Path | None = None, local_files_only: bool = False):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from ._codec.modeling_moss_audio_tokenizer import MossAudioTokenizerModel
    from ._model.modeling_moss_tts import MossTTSDelayModel
    from ._model.processing_moss_tts import MossTTSDelayProcessor

    options = {"cache_dir": cache_dir, "local_files_only": local_files_only}
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]
    model_path = snapshot_download(
        MODEL_ID, revision=TTS_REVISION, allow_patterns=patterns, **options
    )
    codec_path = snapshot_download(
        CODEC_ID, revision=CODEC_REVISION, allow_patterns=patterns, **options
    )
    model = MossTTSDelayModel.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    codec = MossAudioTokenizerModel.from_pretrained(
        codec_path, dtype=torch.float32, device_map="cuda"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    processor = MossTTSDelayProcessor(
        tokenizer=tokenizer, audio_tokenizer=codec, model_config=model.config
    )
    return model, codec, processor
