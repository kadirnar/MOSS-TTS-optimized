import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor
from moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerModel

ROOT = Path(__file__).resolve().parents[1]
TTS_REVISION = "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"
CODEC_REVISION = "3cd226ba2947efa357ef453bcad111b6eafba782"
RESULTS = ROOT / "optimization" / "results"
RESULTS.mkdir(exist_ok=True)


def load_models():
    torch.set_num_threads(4)
    transformers_logging.disable_progress_bar()
    path = snapshot_download("OpenMOSS-Team/MOSS-TTS-v1.5", revision=TTS_REVISION, local_files_only=True)
    codec_path = snapshot_download("OpenMOSS-Team/MOSS-Audio-Tokenizer", revision=CODEC_REVISION, local_files_only=True)
    model = MossTTSDelayModel.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    codec = MossAudioTokenizerModel.from_pretrained(codec_path, dtype=torch.float32, device_map="cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    processor = MossTTSDelayProcessor(tokenizer=tokenizer, audio_tokenizer=codec, model_config=model.config)
    return model, codec, processor


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def stats(values):
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)), "min_ms": float(min(values)), "samples_ms": values}


def save_json(name, data):
    (RESULTS / name).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2), flush=True)
