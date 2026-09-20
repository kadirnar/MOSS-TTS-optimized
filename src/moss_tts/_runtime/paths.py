"""Read-only package resources and a separate writable compilation cache."""

import os
from pathlib import Path
import shutil

ASSETS = Path(__file__).resolve().parent.parent / "_assets"


def build_directory(kernel: str, digest: str) -> Path:
    base = os.environ.get("MOSS_TTS_CACHE")
    directory = Path(base) if base else Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ) / "moss-tts"
    directory = directory / "kernels" / kernel / digest
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def nvcc() -> str:
    candidate = os.environ.get("MOSS_TTS_NVCC") or shutil.which("nvcc")
    if not candidate:
        candidate = str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
    if not Path(candidate).is_file():
        raise RuntimeError("The gptq preset requires nvcc; set CUDA_HOME or MOSS_TTS_NVCC.")
    return candidate
