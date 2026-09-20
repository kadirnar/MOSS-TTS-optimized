"""Opt-in complete-model integration test using an external reference file."""

import os
from contextlib import closing

import pytest

from moss_tts import MossTTS

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not os.environ.get("MOSS_TTS_TEST_REFERENCE"),
        reason="Set MOSS_TTS_TEST_REFERENCE to run on GPU",
    ),
]


def test_full_codebook_stream_and_cancellation():
    import torch

    calibration = os.environ.get("MOSS_TTS_TEST_CALIBRATION")
    with MossTTS.from_pretrained(
        preset="gptq" if calibration else "bf16",
        calibration_path=calibration,
        local_files_only=True,
    ) as model:
        voice = model.clone_voice(os.environ["MOSS_TTS_TEST_REFERENCE"])

        def generate():
            return model.stream("Hello, this is a voice cloning test.", voice=voice, seed=1234)

        with closing(generate()) as stream:
            first = next(stream)
            saved = first.pcm.clone()
            with pytest.raises(RuntimeError, match="busy"):
                next(generate())
        with torch.cuda.stream(torch.cuda.Stream()):
            chunks = list(generate())
        assert chunks and all(chunk.sample_rate == 48000 for chunk in chunks)
        assert all(chunk.pcm.numel() == 3840 for chunk in chunks[1:-1])
        assert sum(chunk.pcm.numel() for chunk in chunks) == model.metrics["frames"] * 3840
        assert len(chunks[0].pcm16()) == chunks[0].pcm.numel() * 2
        assert torch.equal(first.pcm, saved)
        assert torch.equal(chunks[0].pcm, saved)
        assert model.codebooks == 32
