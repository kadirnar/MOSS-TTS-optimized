"""Streaming speech synthesis with the MOSS-TTS v1.5 8B model."""

from .api import MossTTS
from .types import AudioChunk, Voice

__version__ = "0.3.0"
__all__ = ["MossTTS", "AudioChunk", "Voice", "__version__"]
