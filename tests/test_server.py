import base64

import pytest
from fastapi.testclient import TestClient

from moss_tts import server


@pytest.fixture
def client(monkeypatch):
    class Chunk:
        def pcm16(self):
            return b"\x00\x00" * 1920

    class Model:
        def clone_voice(self, source):
            if source.read() != b"reference":
                raise ValueError("Invalid reference")
            return object()

        def stream(self, text, **kwargs):
            if text == "error":
                raise ValueError("Invalid synthesis request")
            if text != "empty":
                yield Chunk()
                yield Chunk()

        def close(self):
            pass

    monkeypatch.setattr(server.MossTTS, "from_pretrained", lambda **_: Model())
    with TestClient(server.create_app()) as connection:
        yield connection


def test_health_and_stream(client):
    assert client.get("/health").json()["codebooks"] == 32
    response = client.post("/v1/audio/speech", json={"input": "Hello"})
    assert response.status_code == 200
    assert response.headers["x-audio-format"] == "s16le"
    assert len(response.content) == 2 * 3840


def test_voice_lifecycle(client):
    encoded = base64.b64encode(b"reference").decode()
    voice = client.post("/v1/voices", json={"wav_base64": encoded}).json()["voice"]
    assert (
        client.post("/v1/audio/speech", json={"input": "Hello", "voice": voice}).status_code == 200
    )
    assert client.delete(f"/v1/voices/{voice}").status_code == 200
    assert (
        client.post("/v1/audio/speech", json={"input": "Hello", "voice": voice}).status_code == 404
    )


def test_errors_and_recovery(client):
    assert client.post("/v1/voices", json={"wav_base64": "not base64"}).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": "error"}).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": "empty"}).status_code == 422
    assert (
        client.post("/v1/audio/speech", json={"input": "Hello", "max_new_tokens": 32}).status_code
        == 422
    )
    assert client.post("/v1/audio/speech", json={"input": "Recovered"}).status_code == 200
