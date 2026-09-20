import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from moss_tts import AudioChunk, MossTTS, Voice


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_: nullcontext())
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda *_: None)

    class Engine:
        last_metrics = {}
        finalized = 0
        closed = False

        def stream(self, text, reference, language, budget):
            try:
                if text == "error":
                    raise ValueError("Generation failed")
                for frame in range(2):
                    yield AudioChunk(torch.full((1920,), 0.25), frame, 10.0)
                self.last_metrics = {"frames": 2, "ttfa_ms": 10.0, "step_ms": [1.0]}
            finally:
                self.finalized += 1

        def close(self):
            self.closed = True

    engine = Engine()
    engine.codec = engine
    encoder = SimpleNamespace(encode=lambda wave, rate: torch.zeros(4, 32, dtype=torch.long))
    tts = MossTTS(engine, encoder, device=torch.device("cuda:0"), preset="bf16")
    yield tts, engine
    tts.close()


def test_import_does_not_load_torch():
    subprocess.run(
        [sys.executable, "-c", "import moss_tts, sys; assert 'torch' not in sys.modules"],
        check=True,
    )


def test_cancel_busy_and_recovery(model):
    tts, engine = model
    stream = tts.stream("First")
    next(stream)
    with pytest.raises(RuntimeError, match="busy"):
        next(tts.stream("Second"))
    with pytest.raises(RuntimeError, match="active stream"):
        tts.close()
    stream.close()
    assert engine.finalized == 1
    assert len(list(tts.stream("Recovered"))) == 2
    assert tts.metrics["frames"] == 2
    metrics = tts.metrics
    metrics["step_ms"].clear()
    assert tts.metrics["step_ms"] == [1.0]


def test_generation_error_releases_lock(model):
    tts, engine = model
    with pytest.raises(ValueError, match="Generation failed"):
        list(tts.stream("error"))
    assert len(list(tts.stream("Recovered"))) == 2
    assert engine.finalized == 2


def test_close_rejects_requests(model):
    tts, engine = model
    tts.close()
    tts.close()
    assert engine.closed
    with pytest.raises(RuntimeError, match="closed"):
        next(tts.stream("Hello"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": " "},
        {"text": "Hello", "max_new_tokens": 32},
        {"text": "Hello", "seed": -1},
        {"text": "Hello", "language": ""},
    ],
)
def test_invalid_requests(model, kwargs):
    with pytest.raises(ValueError):
        model[0].stream(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preset": "4b"},
        {"preset": "gptq"},
        {"calibration_path": "unwanted"},
        {"max_length": 127},
        {"cpu_threads": 0},
        {"preset": "gptq", "calibration_path": "unused", "max_length": 512},
    ],
)
def test_configuration_rejected_before_loading(kwargs):
    with pytest.raises(ValueError):
        MossTTS.from_pretrained(**kwargs)


def test_reference_validation_and_reuse(model, tmp_path):
    tts, _ = model
    path = tmp_path / "reference.wav"
    sf.write(path, np.zeros(4800, dtype=np.float32), 24000)
    voice = tts.clone_voice(path)
    assert isinstance(voice, Voice) and voice.codes.shape == (4, 32)
    assert len(list(tts.stream("Hello", voice=voice))) == 2
    sf.write(path, np.zeros(100, dtype=np.float32), 24000)
    with pytest.raises(ValueError, match="0.2"):
        tts.clone_voice(path)


def test_pcm16_format():
    chunk = AudioChunk(torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0]), 0, 0.0)
    assert np.frombuffer(chunk.pcm16(), dtype="<i2").tolist() == [-32767, -16383, 0, 16383, 32767]
