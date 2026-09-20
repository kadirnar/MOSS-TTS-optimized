"""Optional HTTP transport around the public MossTTS streaming API."""

import asyncio
import base64
import io
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager, closing

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .api import MossTTS
from .models import MODEL_ID


class VoiceRequest(BaseModel):
    wav_base64: str = Field(min_length=1, max_length=16_000_000)


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=800)
    voice: str | None = None
    language: str = "English"
    max_new_tokens: int = Field(default=400, ge=33, le=700)
    seed: int = Field(default=1234, ge=0, le=2**32 - 1)


def create_app(**model_options) -> FastAPI:
    """Create an app that owns one model and one serialized GPU worker."""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moss-tts")
    busy = threading.Lock()
    voices = {}
    state = {}
    cancellations = set()

    @asynccontextmanager
    async def lifespan(_app):
        loop = asyncio.get_running_loop()
        try:
            state["tts"] = await loop.run_in_executor(
                pool, lambda: MossTTS.from_pretrained(**model_options)
            )
            yield
        finally:
            for event in tuple(cancellations):
                event.set()
            if "tts" in state:
                await loop.run_in_executor(pool, state["tts"].close)
                state.clear()
            pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="MOSS-TTS 8B", version="0.3.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "ready": "tts" in state,
            "model": MODEL_ID,
            "preset": model_options.get("preset", "bf16"),
            "sample_rate": MossTTS.sample_rate,
            "channels": 1,
            "codebooks": 32,
            "pcm_format": "s16le",
            "concurrency": 1,
            "cached_voices": len(voices),
        }

    @app.post("/v1/voices")
    async def add_voice(request: VoiceRequest):
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "GPU is serving another request")

        def encode():
            try:
                if len(voices) >= 64:
                    raise ValueError("Voice cache is full; delete an unused voice first")
                raw = base64.b64decode(request.wav_base64, validate=True)
                voice = state["tts"].clone_voice(io.BytesIO(raw))
                name = uuid.uuid4().hex
                voices[name] = voice
                return {"voice": name}
            finally:
                busy.release()

        try:
            return await asyncio.get_running_loop().run_in_executor(pool, encode)
        except (ValueError, RuntimeError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @app.delete("/v1/voices/{voice_id}")
    async def delete_voice(voice_id: str):
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "GPU is serving another request")
        try:
            if voice_id not in voices:
                raise HTTPException(404, "Unknown voice")
            del voices[voice_id]
            return {"deleted": voice_id}
        finally:
            busy.release()

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest):
        if request.voice is not None and request.voice not in voices:
            raise HTTPException(404, "Unknown voice; register one at /v1/voices")
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "GPU is serving another request")
        reference = voices.get(request.voice)
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=8)
        cancelled = threading.Event()
        cancellations.add(cancelled)

        def send(item):
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not cancelled.is_set():
                try:
                    future.result(timeout=0.1)
                    return True
                except FutureTimeout:
                    continue
            future.cancel()
            return False

        def produce():
            try:
                with closing(
                    state["tts"].stream(
                        request.input,
                        voice=reference,
                        language=request.language,
                        max_new_tokens=request.max_new_tokens,
                        seed=request.seed,
                    )
                ) as chunks:
                    for chunk in chunks:
                        if not send(chunk.pcm16()):
                            break
            except Exception as error:
                send(error)
            finally:
                busy.release()
                if not cancelled.is_set():
                    send(None)

        pool.submit(produce)
        try:
            first = await queue.get()
            if isinstance(first, Exception):
                raise HTTPException(400, str(first)) from first
            if first is None:
                raise HTTPException(422, "No audio generated within the token budget")
        except BaseException:
            cancelled.set()
            cancellations.discard(cancelled)
            raise

        async def body():
            try:
                yield first
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    if isinstance(item, Exception):
                        raise RuntimeError("Synthesis failed after audio began") from item
                    yield item
            finally:
                cancelled.set()
                cancellations.discard(cancelled)

        return StreamingResponse(
            body(),
            media_type="audio/pcm",
            headers={
                "X-Audio-Sample-Rate": str(MossTTS.sample_rate),
                "X-Audio-Channels": "1",
                "X-Audio-Format": "s16le",
                "X-Audio-Codebooks": "32",
            },
        )

    return app
