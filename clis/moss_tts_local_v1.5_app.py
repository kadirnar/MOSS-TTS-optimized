# coding=utf-8
"""Realtime Web Audio app for MOSS-TTS Local Transformer v1.5."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import queue
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import orjson
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
STREAMING_MODULE_DIR = REPO_ROOT / "moss_tts_local_v1.5"
if str(STREAMING_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMING_MODULE_DIR))

from streaming import (
    DEFAULT_CODEC_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    StreamingRequest,
    StreamingRuntime,
    load_runtime,
    synthesize_stream,
)


torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

DEFAULT_UPLOAD_DIR = Path("outputs/moss_tts_local_v1_5_uploads")
DEFAULT_MAX_NEW_TOKENS = 7500
MODE_CLONE = "Clone"
MODE_CONTINUE = "Continuation"
MODE_CONTINUE_CLONE = "Continuation + Clone"
CONTINUATION_NOTICE = (
    "Continuation mode is active. Fill Reference Audio Transcript with the transcript of the reference audio."
)
ZH_TOKENS_PER_CHAR = 3.098411951313033
EN_TOKENS_PER_CHAR = 0.8673376262755219
REFERENCE_AUDIO_DIR = REPO_ROOT / "assets" / "audio"
EXAMPLE_TEXTS_JSONL_PATH = REPO_ROOT / "assets" / "text" / "moss_tts_example_texts.jsonl"
LANGUAGE_TAG_AUTO = "Auto (omit)"
LANGUAGE_TAG_CHOICES = [
    LANGUAGE_TAG_AUTO,
    "Chinese",
    "Cantonese",
    "English",
    "Arabic",
    "Czech",
    "Danish",
    "Dutch",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Italian",
    "Japanese",
    "Korean",
    "Macedonian",
    "Malay",
    "Persian (Farsi)",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tagalog",
    "Thai",
    "Turkish",
    "Vietnamese",
]


def _parse_example_id(example_id: str) -> tuple[str, int] | None:
    matched = re.fullmatch(r"(zh|en)/(\d+)", (example_id or "").strip())
    if matched is None:
        return None
    return matched.group(1), int(matched.group(2))


def _resolve_reference_audio_path(language: str, index: int) -> Path | None:
    stem = f"reference_{language}_{index}"
    for ext in (".wav", ".mp3", ".m4a"):
        audio_path = REFERENCE_AUDIO_DIR / f"{stem}{ext}"
        if audio_path.exists():
            return audio_path
    return None


def build_example_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not EXAMPLE_TEXTS_JSONL_PATH.exists():
        return rows
    with open(EXAMPLE_TEXTS_JSONL_PATH, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            sample = orjson.loads(line)
            parsed = _parse_example_id(str(sample.get("id", "")))
            if parsed is None:
                continue
            language, index = parsed
            audio_path = _resolve_reference_audio_path(language, index)
            if audio_path is None:
                continue
            rows.append(
                {
                    "role": str(sample.get("role", "")).strip(),
                    "audio_path": str(audio_path),
                    "text": str(sample.get("text", "")).strip(),
                    "language": "Chinese" if language == "zh" else "English",
                }
            )
    return rows


EXAMPLE_ROWS = build_example_rows()


def _normalize_language(language_tag: str | None) -> str:
    value = (language_tag or "").strip()
    return "" if value == LANGUAGE_TAG_AUTO else value


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int | None = None) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = int(default)
    parsed = max(int(minimum), parsed)
    if maximum is not None:
        parsed = min(int(maximum), parsed)
    return parsed


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    parsed = max(float(minimum), parsed)
    if maximum is not None:
        parsed = min(float(maximum), parsed)
    return parsed


def _decode_reference_path(path: str) -> str:
    decoded = str(path or "")
    for _ in range(2):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return decoded


def _pcm16le_bytes(waveform: torch.Tensor) -> bytes:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    pcm = waveform.detach().cpu().to(torch.float32).clamp(-1.0, 1.0)
    pcm = (pcm * 32767.0).round().to(torch.int16)
    return pcm.transpose(0, 1).contiguous().numpy().tobytes()


class RuntimeManager:
    def __init__(
        self,
        *,
        model_dir: str,
        codec_dir: str,
        device: str,
        tts_device: str,
        codec_device: str,
        dtype: str,
        attn_implementation: str,
        codec_weight_dtype: str,
        codec_compute_dtype: str,
        warmup: bool,
    ) -> None:
        self.model_dir = str(model_dir)
        self.codec_dir = str(codec_dir)
        self.device = device
        self.tts_device = tts_device
        self.codec_device = codec_device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.codec_weight_dtype = codec_weight_dtype
        self.codec_compute_dtype = codec_compute_dtype
        self.warmup = bool(warmup)
        self._lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._runtime: StreamingRuntime | None = None
        self._loader_thread: threading.Thread | None = None
        self._state = "not_loaded"
        self._error: str | None = None
        self._load_started_at: float | None = None
        self._ready_at: float | None = None

    def _set_status(self, *, state: str, error: str | None = None) -> None:
        with self._status_lock:
            self._state = state
            self._error = error
            if state == "loading":
                self._load_started_at = time.time()
                self._ready_at = None
            elif state == "ready":
                self._ready_at = time.time()

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            state = self._state
            error = self._error
            load_started_at = self._load_started_at
            ready_at = self._ready_at
        elapsed = None
        if load_started_at is not None:
            elapsed = max(0.0, (ready_at or time.time()) - load_started_at)
        return {
            "state": state,
            "error": error,
            "load_started_at": load_started_at,
            "ready_at": ready_at,
            "load_elapsed_seconds": elapsed,
            "model_dir": self.model_dir,
            "codec_dir": self.codec_dir,
            "device": self.device,
            "tts_device": self.tts_device,
            "codec_device": self.codec_device,
            "dtype": self.dtype,
            "requested_attn_implementation": self.attn_implementation,
            "attn_implementation": (
                self.attn_implementation
                if self._runtime is None
                else self._runtime.attn_implementation
            ),
            "codec_weight_dtype": (
                self.codec_weight_dtype
                if self._runtime is None
                else self._runtime.codec_weight_dtype
            ),
            "codec_compute_dtype": self.codec_compute_dtype,
            "n_vq": None if self._runtime is None else int(self._runtime.n_vq),
            "sample_rate": None if self._runtime is None else int(self._runtime.sample_rate),
        }

    def preload_async(self) -> None:
        with self._status_lock:
            if self._runtime is not None or self._state == "loading":
                return
            if self._loader_thread is not None and self._loader_thread.is_alive():
                return

        def _load() -> None:
            try:
                self.get()
            except Exception:
                logging.exception("failed to preload MOSS-TTS Local v1.5 streaming runtime")

        self._loader_thread = threading.Thread(target=_load, name="moss-tts-local-v1.5-runtime-loader", daemon=True)
        self._loader_thread.start()

    def get(self) -> StreamingRuntime:
        with self._lock:
            if self._runtime is None:
                self._set_status(state="loading")
                try:
                    self._runtime = load_runtime(
                        model_dir=self.model_dir,
                        codec_dir=self.codec_dir,
                        device=self.device,
                        tts_device=self.tts_device,
                        codec_device=self.codec_device,
                        dtype=self.dtype,
                        attn_implementation=self.attn_implementation,
                        codec_weight_dtype=self.codec_weight_dtype,
                        codec_compute_dtype=self.codec_compute_dtype,
                        warmup=self.warmup,
                    )
                except Exception as exc:
                    self._set_status(state="error", error=str(exc))
                    raise
                self._set_status(state="ready")
            return self._runtime


class StreamingJob:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=64)
        self.status_lock = threading.Lock()
        self.status: dict[str, Any] = {
            "job_id": job_id,
            "state": "queued",
            "created_at": time.time(),
            "started_at": None,
            "first_audio_at": None,
            "sample_rate": 48000,
            "channels": 2,
            "generated_frames": 0,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "generated_audio_seconds": 0.0,
            "emitted_audio_seconds": 0.0,
            "lead_seconds": 0.0,
            "error": None,
            "closed": False,
        }
        self.result: dict[str, Any] | None = None
        self.thread: threading.Thread | None = None
        self.is_closed = False

    def update(self, **kwargs: Any) -> None:
        with self.status_lock:
            self.status.update(kwargs)

    def snapshot(self) -> dict[str, Any]:
        with self.status_lock:
            return dict(self.status)


class StreamingJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, StreamingJob] = {}
        self._lock = threading.Lock()

    def create(self) -> StreamingJob:
        job = StreamingJob(uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> StreamingJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"stream job not found: {job_id}")
        return job

    def close(self, job_id: str) -> StreamingJob:
        job = self.get(job_id)
        with job.status_lock:
            job.is_closed = True
            job.status["closed"] = True
            if job.status.get("state") not in {"finished", "error"}:
                job.status["state"] = "closed"
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass
        return job


def create_app(
    *,
    model_dir: str,
    codec_dir: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    upload_dir: str | Path = DEFAULT_UPLOAD_DIR,
    device: str = "cuda",
    tts_device: str = "cuda:0",
    codec_device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    codec_weight_dtype: str = "fp32",
    codec_compute_dtype: str = "bf16",
    warmup: bool = True,
    preload: bool = True,
) -> FastAPI:
    runtime_manager = RuntimeManager(
        model_dir=str(model_dir),
        codec_dir=str(codec_dir),
        device=device,
        tts_device=tts_device,
        codec_device=codec_device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        codec_weight_dtype=codec_weight_dtype,
        codec_compute_dtype=codec_compute_dtype,
        warmup=warmup,
    )
    jobs = StreamingJobManager()
    output_dir = Path(output_dir)
    upload_dir = Path(upload_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if preload:
            runtime_manager.get()
        yield

    app = FastAPI(title="MOSS-TTS Local v1.5 Realtime Streaming", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        defaults = {
            "text": "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。",
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "seed": 1234,
        }
        return HTMLResponse(
            _html(
                defaults=defaults,
                examples=EXAMPLE_ROWS,
                languages=LANGUAGE_TAG_CHOICES,
                runtime=runtime_manager.status(),
            )
        )

    def _put_stream_audio(job: StreamingJob, pcm_bytes: bytes) -> None:
        while True:
            with job.status_lock:
                if job.is_closed:
                    return
            try:
                job.audio_queue.put(pcm_bytes, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run_job(job: StreamingJob, request: StreamingRequest, mode_name: str, streaming_generation: bool) -> None:
        try:
            job.update(
                state="loading_runtime",
                started_at=time.time(),
                max_new_tokens=int(request.max_new_frames),
                mode=mode_name,
                streaming_generation=streaming_generation,
            )
            runtime = runtime_manager.get()
            job.update(state="running", sample_rate=runtime.sample_rate, channels=2, n_vq=runtime.n_vq)
            for event in synthesize_stream(runtime, request, output_dir=output_dir):
                with job.status_lock:
                    if job.is_closed:
                        break
                if event.type == "metadata":
                    job.update(**event.data)
                elif event.type == "progress":
                    job.update(**event.data)
                elif event.type == "audio":
                    waveform = event.data["waveform"]
                    channels = 1 if waveform.ndim == 1 else int(min(2, waveform.shape[0]))
                    with job.status_lock:
                        if job.status.get("first_audio_at") is None:
                            job.status["first_audio_at"] = time.time()
                    if streaming_generation:
                        _put_stream_audio(job, _pcm16le_bytes(waveform))
                    job.update(
                        generated_frames=event.data.get("generated_frames", job.snapshot().get("generated_frames", 0)),
                        emitted_audio_seconds=event.data.get("emitted_audio_seconds", 0.0),
                        generated_audio_seconds=event.data.get("generated_audio_seconds", 0.0),
                        sample_rate=event.data.get("sample_rate", runtime.sample_rate),
                        channels=channels,
                        lead_seconds=event.data.get("lead_seconds", 0.0),
                        generation_lead_seconds=event.data.get("generation_lead_seconds", 0.0),
                        playback_lead_seconds=event.data.get("playback_lead_seconds"),
                        generation_realtime_factor=event.data.get("generation_realtime_factor", 0.0),
                        post_first_generation_realtime_factor=event.data.get(
                            "post_first_generation_realtime_factor"
                        ),
                        first_audio_latency_seconds=event.data.get("first_audio_latency_seconds"),
                        decode_chunks_submitted=event.data.get("decode_chunks_submitted", 0),
                        decode_queue_depth=event.data.get("decode_queue_depth", 0),
                        pending_decode_frames=event.data.get("pending_decode_frames", 0),
                        chunk_frames=event.data.get("chunk_frames", 0),
                    )
                elif event.type == "result":
                    metadata = dict(event.data["metadata"])
                    job.result = {
                        "audio_path": event.data["audio_path"],
                        "tokens_path": event.data["tokens_path"],
                        "metadata_path": event.data["metadata_path"],
                        "metadata": metadata,
                    }
                    job.update(
                        state="finished",
                        generated_frames=metadata.get("generated_frames", 0),
                        emitted_audio_seconds=metadata.get("duration_seconds", 0.0),
                        audio_path=event.data["audio_path"],
                    )
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass
        except Exception as exc:  # noqa: BLE001
            job.update(state="error", error=str(exc))
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass

    @app.post("/api/generate-stream/start")
    async def generate_stream_start(
        mode: str = Form("voice_clone"),
        language: str = Form(""),
        text: str = Form(...),
        prompt_text: str = Form(""),
        max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
        codec_chunk_frames: int = Form(8),
        seed: int = Form(1234),
        tokens_control: int = Form(0),
        tokens: int = Form(0),
        temperature: float = Form(1.7),
        top_p: float = Form(0.8),
        top_k: int = Form(25),
        repetition_penalty: float = Form(1.0),
        streaming_generation: int = Form(1),
        example_audio_path: str = Form(""),
        prompt_audio: UploadFile | None = File(None),
    ) -> JSONResponse:
        text = (text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty")

        mode = (mode or "").strip().lower()
        if mode not in {"voice_clone", "continuation", "continuation_clone"}:
            mode = "voice_clone"
        mode_name = {
            "voice_clone": MODE_CLONE,
            "continuation": MODE_CONTINUE,
            "continuation_clone": MODE_CONTINUE_CLONE,
        }[mode]

        prompt_audio_path = ""
        if prompt_audio is not None and prompt_audio.filename:
            suffix = Path(prompt_audio.filename).suffix or ".wav"
            prompt_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
            prompt_path.write_bytes(await prompt_audio.read())
            prompt_audio_path = str(prompt_path)
        elif example_audio_path:
            candidate = Path(_decode_reference_path(example_audio_path))
            if candidate.exists() and REFERENCE_AUDIO_DIR in candidate.resolve().parents:
                prompt_audio_path = str(candidate)

        if not prompt_audio_path:
            mode_name = "Direct Generation"

        if mode in {"continuation", "continuation_clone"} and prompt_audio_path:
            if not text:
                raise HTTPException(status_code=400, detail="continuation mode requires text")
            if not (prompt_text or "").strip():
                raise HTTPException(
                    status_code=400,
                    detail="continuation mode requires reference audio transcript",
                )

        max_new_tokens = _safe_int(
            max_new_tokens,
            default=DEFAULT_MAX_NEW_TOKENS,
            minimum=1,
            maximum=DEFAULT_MAX_NEW_TOKENS,
        )
        codec_chunk_frames = _safe_int(codec_chunk_frames, default=8, minimum=0, maximum=32)
        streaming_generation_enabled = bool(_safe_int(streaming_generation, default=1, minimum=0, maximum=1))
        request = StreamingRequest(
            text=text,
            mode="continuation" if not prompt_audio_path or mode in {"continuation", "continuation_clone"} else "voice_clone",
            prompt_text=prompt_text or "",
            prompt_audio_path=prompt_audio_path or None,
            language=_normalize_language(language),
            tokens_control=bool(int(tokens_control)),
            tokens=_safe_int(tokens, default=0, minimum=0),
            max_new_frames=max_new_tokens,
            do_sample=True,
            temperature=_safe_float(temperature, default=1.7, minimum=0.1, maximum=3.0),
            top_p=_safe_float(top_p, default=0.8, minimum=0.1, maximum=1.0),
            top_k=_safe_int(top_k, default=25, minimum=1, maximum=200),
            repetition_penalty=_safe_float(repetition_penalty, default=1.0, minimum=0.8, maximum=2.0),
            seed=None if int(seed) < 0 else int(seed),
            codec_chunk_frames=codec_chunk_frames,
        )
        job = jobs.create()
        thread = threading.Thread(target=_run_job, args=(job, request, mode_name, streaming_generation_enabled), daemon=True)
        job.thread = thread
        thread.start()
        return JSONResponse(
            {
                "job_id": job.job_id,
                "audio_url": f"/api/generate-stream/{job.job_id}/audio",
                "status_url": f"/api/generate-stream/{job.job_id}/status",
                "result_url": f"/api/generate-stream/{job.job_id}/result",
                "sample_rate": runtime_manager.status().get("sample_rate") or 48000,
                "channels": 2,
            }
        )

    @app.get("/api/reference-audio")
    async def reference_audio(path: str) -> FileResponse:
        try:
            reference_root = REFERENCE_AUDIO_DIR.resolve(strict=True)
            candidate = Path(_decode_reference_path(path)).resolve(strict=True)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="reference audio not found") from exc
        if candidate != reference_root and reference_root not in candidate.parents:
            raise HTTPException(status_code=403, detail="reference audio path is not allowed")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="reference audio not found")
        media_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        return FileResponse(str(candidate), media_type=media_type, filename=candidate.name)

    @app.get("/api/generate-stream/{job_id}/audio")
    async def generate_stream_audio(job_id: str) -> StreamingResponse:
        job = jobs.get(job_id)

        def iterator():
            while True:
                item = job.audio_queue.get()
                if item is None:
                    break
                yield item

        snapshot = job.snapshot()
        return StreamingResponse(
            iterator(),
            media_type="application/octet-stream",
            headers={
                "X-Audio-Codec": "pcm_s16le",
                "X-Audio-Sample-Rate": str(snapshot.get("sample_rate", 48000)),
                "X-Audio-Channels": str(snapshot.get("channels", 2)),
                "X-Stream-Id": job_id,
            },
        )

    @app.get("/api/generate-stream/{job_id}/status")
    async def generate_stream_status(job_id: str) -> JSONResponse:
        return JSONResponse(jobs.get(job_id).snapshot())

    @app.get("/api/generate-stream/{job_id}/result")
    async def generate_stream_result(job_id: str) -> JSONResponse:
        job = jobs.get(job_id)
        if job.result is None:
            raise HTTPException(status_code=404, detail="result is not ready")
        return JSONResponse(job.result)

    @app.get("/api/generate-stream/{job_id}/result-audio")
    async def generate_stream_result_audio(job_id: str) -> FileResponse:
        job = jobs.get(job_id)
        if job.result is None:
            raise HTTPException(status_code=404, detail="result is not ready")
        return FileResponse(job.result["audio_path"], media_type="audio/wav", filename="generated.wav")

    @app.post("/api/generate-stream/{job_id}/close")
    async def generate_stream_close(job_id: str) -> JSONResponse:
        jobs.close(job_id)
        return JSONResponse({"ok": True})

    @app.get("/api/runtime")
    async def runtime_info() -> JSONResponse:
        return JSONResponse(
            {
                "model_dir": str(model_dir),
                "codec_dir": str(codec_dir),
                "output_dir": str(output_dir),
                "upload_dir": str(upload_dir),
                "device": device,
                "tts_device": tts_device,
                "codec_device": codec_device,
                "dtype": dtype,
                "attn_implementation": attn_implementation,
                "codec_weight_dtype": codec_weight_dtype,
                "codec_compute_dtype": codec_compute_dtype,
                "runtime": runtime_manager.status(),
            }
        )

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse(runtime_manager.status())

    return app


def _html(*, defaults: dict[str, Any], examples: list[dict[str, str]], languages: list[str], runtime: dict[str, Any]) -> str:
    replacements = {
        "__DEFAULT_TEXT__": json.dumps(defaults["text"], ensure_ascii=False),
        "__DEFAULT_MAX_NEW_TOKENS__": str(defaults["max_new_tokens"]),
        "__DEFAULT_SEED__": str(defaults["seed"]),
        "__EXAMPLES_JSON__": json.dumps(examples, ensure_ascii=False),
        "__LANGUAGES_JSON__": json.dumps(languages, ensure_ascii=False),
        "__RUNTIME_JSON__": json.dumps(runtime, ensure_ascii=False),
    }
    html = INDEX_HTML
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MOSS-TTS Local v1.5 Realtime Streaming</title>
  <style>
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #111418;
      --muted: #4d5562;
      --line: #e5e7eb;
      --accent: #0f766e;
      --orange: #f97316;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #f7f8fa 0%, #f3f5f7 100%);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .page { max-width: 1840px; margin: 0 auto; padding: 22px 58px 48px; }
    .app-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      padding: 14px;
      margin-bottom: 16px;
    }
    .app-title { font-size: 22px; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.2px; }
    .app-subtitle { color: var(--muted); font-size: 14px; }
    .layout { display: grid; grid-template-columns: minmax(0, 3fr) minmax(360px, 2fr); gap: 16px; align-items: start; }
    .stack { display: flex; flex-direction: column; gap: 16px; }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 4px;
      padding: 12px;
    }
    label { display: block; color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    textarea, select, input[type="number"], input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      padding: 10px 12px;
    }
    textarea { min-height: 190px; resize: vertical; }
    .small-textarea { min-height: 76px; }
    .hint { color: var(--muted); font-size: 12px; margin-top: -3px; }
    .drop-zone {
      border: 1px solid var(--line);
      border-radius: 4px;
      min-height: 158px;
      display: grid;
      place-items: center;
      color: #6b7280;
      background: #fff;
      position: relative;
      overflow: hidden;
    }
    .drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; z-index: 1; }
    .drop-zone.hidden { display: none; }
    .drop-zone.has-reference { min-height: 0; display: block; padding: 0; overflow: visible; }
    .drop-zone.has-reference input { display: none; }
    .drop-zone.has-reference .drop-copy { display: none; }
    .reference-preview { display: block; width: 100%; position: relative; z-index: 2; pointer-events: auto; }
    audio[disabled] { opacity: 0.55; pointer-events: none; }
    .drop-copy { text-align: center; pointer-events: none; }
    .selected-reference { margin-top: 8px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; text-align: center; }
    .reference-action-row { display: flex; justify-content: center; align-items: center; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
    .reference-source-row { display: flex; justify-content: center; margin-top: 0; }
    .reference-source-toggle { display: inline-flex; gap: 4px; border: 1px solid var(--line); border-radius: 8px; padding: 4px; background: #fff; }
    .reference-source-button { display: inline-flex; align-items: center; gap: 7px; border-radius: 6px; padding: 8px 12px; background: transparent; color: var(--muted); }
    .reference-source-button.active { background: var(--accent); color: #fff; }
    .reference-source-button svg { width: 18px; height: 18px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .reference-record-controls { display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
    .reference-record-controls.hidden { display: none; }
    .record-button { display: inline-flex; align-items: center; gap: 8px; background: var(--accent); color: #fff; }
    .record-button.recording { background: #fee2e2; color: #b91c1c; }
    .record-dot { width: 10px; height: 10px; border-radius: 999px; background: currentColor; }
    .record-status { color: var(--muted); font-size: 12px; }
    .radio-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .radio-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 8px 12px;
      cursor: pointer;
      background: #fff;
    }
    .radio-pill input { accent-color: var(--orange); }
    .mode-hint { margin-top: 12px; color: var(--ink); }
    .accordion {
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      padding: 0;
    }
    .accordion summary {
      cursor: pointer;
      list-style: none;
      padding: 12px;
      color: var(--ink);
      border-bottom: 1px solid var(--line);
    }
    .accordion summary::-webkit-details-marker { display: none; }
    .accordion summary::after { content: "▾"; float: right; }
    .accordion[open] summary::after { content: "▴"; }
    .accordion-body { padding: 12px; display: grid; gap: 14px; }
    .control-row { display: grid; grid-template-columns: 1fr 92px; gap: 12px; align-items: center; }
    .control-row input[type="range"] { width: 100%; accent-color: var(--orange); }
    .range-label { color: var(--muted); font-size: 13px; margin-bottom: 4px; }
    .range-minmax { display: flex; justify-content: space-between; color: #9ca3af; font-size: 11px; margin-top: 2px; }
    .button-row { display: grid; grid-template-columns: 1fr 150px 120px; gap: 10px; }
    button {
      border: none;
      border-radius: 4px;
      padding: 12px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    .primary { background: var(--accent); color: #fff; }
    .secondary { background: #e5e7eb; color: var(--ink); }
    .small-button { padding: 7px 10px; font-size: 12px; font-weight: 600; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    audio { width: 100%; }
    .audio-panel { min-height: 118px; display: flex; flex-direction: column; gap: 8px; justify-content: center; }
    .status-box {
      min-height: 110px;
      max-height: 260px;
      overflow: auto;
      white-space: pre-wrap;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 10px;
      background: #fff;
      color: var(--ink);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .summary { color: var(--muted); margin-bottom: 8px; }
    .meter { height: 7px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-bottom: 8px; }
    .meter > div { height: 100%; width: 0%; background: var(--orange); transition: width 0.2s ease; }
    .examples-wrap { overflow: auto; max-height: 500px; border: 1px solid var(--line); border-radius: 4px; }
    table { width: 100%; border-collapse: collapse; background: #fff; font-size: 13px; }
    th, td { border-bottom: 1px solid #eef0f3; padding: 10px; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: #fff; z-index: 1; font-weight: 700; }
    tr { cursor: pointer; }
    tr:hover td { background: #f8fafc; }
    .role-cell { width: 160px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .download { display: none; margin-top: 8px; color: var(--accent); font-weight: 700; text-decoration: none; }
    .hidden { display: none; }
    @media (max-width: 1100px) {
      .page { padding: 16px; }
      .layout { grid-template-columns: 1fr; }
      .button-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="app-card">
      <div class="app-title">MOSS-TTS Local v1.5 Realtime Streaming</div>
      <div class="app-subtitle">Realtime streaming decode with Direct Generation, Clone, Continuation, Continuation + Clone, and language tags</div>
    </div>

    <div class="layout">
      <div class="stack">
        <div class="panel">
          <label for="text">Text</label>
          <textarea id="text" placeholder="Enter text to synthesize or continue after the reference audio."></textarea>
        </div>

        <div class="panel">
          <label>Reference Audio (Optional)</label>
          <div id="reference-drop-zone" class="drop-zone">
            <input id="prompt-audio" type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.opus,.aac">
            <div class="drop-copy">Drop audio here<br>or<br>click to upload</div>
            <audio id="reference-audio-preview" class="reference-preview hidden" controls></audio>
          </div>
          <input id="example-audio-path" type="hidden" value="">
          <div id="reference-record-controls" class="reference-record-controls hidden">
            <button id="reference-record-button" class="record-button" type="button"><span class="record-dot"></span><span id="reference-record-button-label">Start Recording</span></button>
            <span id="reference-record-status" class="record-status">Ready to record.</span>
          </div>
          <div class="reference-action-row">
            <div class="reference-source-row">
              <div class="reference-source-toggle" role="group" aria-label="Reference audio source">
                <button id="reference-source-upload" class="reference-source-button active" type="button" aria-pressed="true" title="Upload">
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4"></path><path d="M7 9l5-5 5 5"></path><path d="M5 20h14"></path></svg>
                  <span>Upload</span>
                </button>
                <button id="reference-source-record" class="reference-source-button" type="button" aria-pressed="false" title="Record">
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3z"></path><path d="M19 11a7 7 0 0 1-14 0"></path><path d="M12 18v3"></path><path d="M8 21h8"></path></svg>
                  <span>Record</span>
                </button>
              </div>
            </div>
            <button id="clear-reference" class="secondary small-button" type="button">Clear Reference Audio</button>
          </div>
          <div id="selected-reference" class="selected-reference">No reference selected.</div>
        </div>

        <div class="panel">
          <label>Mode with Reference Audio</label>
          <div class="hint">If no reference audio is uploaded, Direct Generation will be used automatically.</div>
          <div class="radio-row" style="margin-top: 10px;">
            <label class="radio-pill"><input type="radio" name="mode" value="voice_clone" checked> Clone</label>
            <label class="radio-pill"><input type="radio" name="mode" value="continuation"> Continuation</label>
            <label class="radio-pill"><input type="radio" name="mode" value="continuation_clone"> Continuation + Clone</label>
          </div>
        </div>
        <div id="mode-hint" class="mode-hint"></div>

        <div id="reference-transcript-panel" class="panel hidden">
          <label for="prompt-text">Reference Audio Transcript</label>
          <div class="hint">Required for Continuation modes. Enter the transcript corresponding to the reference audio.</div>
          <textarea id="prompt-text" class="small-textarea" placeholder="Transcript of the reference audio."></textarea>
        </div>

        <div class="panel">
          <label for="language">Language Tag</label>
          <div class="hint">Optional for v1.5. Set this when the input language is known, especially outside Chinese and English.</div>
          <select id="language"></select>
          <label style="margin-top: 14px;"><input id="tokens-control" type="checkbox"> Enable Duration Control (Expected Audio Tokens)</label>
          <div id="tokens-wrap" class="hidden" style="margin-top: 10px;">
            <label for="tokens">expected_tokens</label>
            <input id="tokens" type="number" min="1" step="1" value="1">
          </div>
        </div>
        <div id="duration-hint" class="hint">Duration control is disabled.</div>

        <details class="accordion" open>
          <summary>Sampling Parameters (Audio)</summary>
          <div class="accordion-body">
            <div class="control-row" data-pair="temperature">
              <div>
                <div class="range-label">temperature</div>
                <input id="temperature-range" type="range" min="0.1" max="3" step="0.05" value="1.7">
                <div class="range-minmax"><span>0.1</span><span>3</span></div>
              </div>
              <input id="temperature" type="number" min="0.1" max="3" step="0.05" value="1.7">
            </div>
            <div class="control-row" data-pair="top-p">
              <div>
                <div class="range-label">top_p</div>
                <input id="top-p-range" type="range" min="0.1" max="1" step="0.01" value="0.8">
                <div class="range-minmax"><span>0.1</span><span>1</span></div>
              </div>
              <input id="top-p" type="number" min="0.1" max="1" step="0.01" value="0.8">
            </div>
            <div class="control-row" data-pair="top-k">
              <div>
                <div class="range-label">top_k</div>
                <input id="top-k-range" type="range" min="1" max="200" step="1" value="25">
                <div class="range-minmax"><span>1</span><span>200</span></div>
              </div>
              <input id="top-k" type="number" min="1" max="200" step="1" value="25">
            </div>
            <div class="control-row" data-pair="repetition-penalty">
              <div>
                <div class="range-label">repetition_penalty</div>
                <input id="repetition-penalty-range" type="range" min="0.8" max="2" step="0.05" value="1.0">
                <div class="range-minmax"><span>0.8</span><span>2</span></div>
              </div>
              <input id="repetition-penalty" type="number" min="0.8" max="2" step="0.05" value="1.0">
            </div>
            <div class="control-row" data-pair="max-new-tokens">
              <div>
                <div class="range-label">max_new_tokens</div>
                <input id="max-new-tokens-range" type="range" min="1" max="7500" step="1" value="__DEFAULT_MAX_NEW_TOKENS__">
                <div class="range-minmax"><span>1</span><span>7500</span></div>
              </div>
              <input id="max-new-tokens" type="number" min="1" max="7500" step="1" value="__DEFAULT_MAX_NEW_TOKENS__">
            </div>
            <div class="control-row" data-pair="codec-chunk-frames">
              <div>
                <div class="range-label">Codec Chunk Frames (0=auto)</div>
                <input id="codec-chunk-frames-range" type="range" min="0" max="32" step="1" value="8">
                <div class="range-minmax"><span>0</span><span>32</span></div>
              </div>
              <input id="codec-chunk-frames" type="number" min="0" max="32" step="1" value="8">
            </div>
            <div class="control-row">
              <label for="initial-playback-delay">Initial Playback Delay (s)</label>
              <input id="initial-playback-delay" type="number" min="0" max="2" step="0.01" value="0.08">
            </div>
            <div class="control-row" data-pair="seed">
              <div>
                <div class="range-label">seed (-1=random)</div>
                <input id="seed-range" type="range" min="-1" max="999999" step="1" value="__DEFAULT_SEED__">
                <div class="range-minmax"><span>-1</span><span>999999</span></div>
              </div>
              <input id="seed" type="number" min="-1" step="1" value="__DEFAULT_SEED__">
            </div>
            <label style="margin-top: 14px;"><input id="streaming-generation" type="checkbox" checked> Enable Streaming Generation</label>
          </div>
        </details>

        <div class="button-row">
          <button id="start" class="primary" type="button">Generate Speech</button>
          <button id="pause" class="secondary" type="button" disabled>Pause Playback</button>
          <button id="stop" class="secondary" type="button">Close Job</button>
        </div>
      </div>

      <div class="stack">
        <div class="panel audio-panel">
          <label>Output Audio</label>
          <audio id="audio-output" controls disabled></audio>
          <a id="download" class="download" href="#">Download final wav</a>
        </div>
        <div class="panel">
          <label>Status</label>
          <div id="runtime-summary" class="summary">Runtime: checking...</div>
          <div id="summary" class="summary"></div>
          <div class="meter"><div id="bar"></div></div>
          <div id="status" class="status-box">idle</div>
        </div>
        <div class="panel">
          <label>Examples (click a row to fill inputs)</label>
          <div class="examples-wrap">
            <table>
              <thead><tr><th>Reference Speech</th><th>Example Text</th></tr></thead>
              <tbody id="examples-body"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
const EXAMPLES = __EXAMPLES_JSON__;
const LANGUAGES = __LANGUAGES_JSON__;
const INITIAL_RUNTIME = __RUNTIME_JSON__;
const DEFAULT_TEXT = __DEFAULT_TEXT__;
const CONTINUATION_NOTICE = "Continuation mode is active. Fill Reference Audio Transcript with the transcript of the reference audio.";

let currentJob = null;
let audioContext = null;
let nextPlaybackTime = 0;
let statusTimer = null;
let runtimeReady = false;
let generationActive = false;
let currentStreamAbortController = null;
let currentStreamingGenerationEnabled = true;
let playbackPaused = false;
let playbackCompletionTimer = null;
let currentInitialPlaybackDelaySeconds = 0.08;
const MIN_INITIAL_PLAYBACK_BUFFER_SECONDS = 0.32;
let pendingRealtimePcmChunks = [];
let pendingRealtimePcmSeconds = 0;
let realtimePlaybackStarted = false;
let currentReferenceObjectUrl = null;
let referenceSourceMode = "upload";
let recordedReferenceFile = null;
let referenceRecordingActive = false;
let referenceRecordingStream = null;
let referenceRecordingAudioContext = null;
let referenceRecordingSource = null;
let referenceRecordingProcessor = null;
let referenceRecordingChunks = [];
let referenceRecordingSampleRate = 48000;
let referenceRecordingStartedAt = 0;
let referenceRecordingTimer = null;

function field(id) { return document.getElementById(id); }
function apiUrl(path) {
  const cleanPath = String(path || "").replace(/^\/+/, "");
  const pagePath = window.location.pathname.endsWith("/") ? window.location.pathname : window.location.pathname + "/";
  return new URL(cleanPath, window.location.origin + pagePath).toString();
}
function referenceAudioUrl(path) {
  const url = new URL(apiUrl("api/reference-audio"));
  url.searchParams.set("path", path);
  return url.toString();
}
function selectedMode() {
  const selected = document.querySelector("input[name='mode']:checked");
  return selected ? selected.value : "voice_clone";
}
function selectedModeName() {
  const mode = selectedMode();
  if (mode === "continuation") return "Continuation";
  if (mode === "continuation_clone") return "Continuation + Clone";
  return "Clone";
}
function hasReference() {
  return Boolean(field("prompt-audio").files[0] || recordedReferenceFile || field("example-audio-path").value);
}
function setStatus(obj) {
  field("status").textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  if (obj && (obj.generated_frames || obj.generated_frames === 0) && obj.max_new_tokens) {
    const pct = Math.min(100, 100 * Number(obj.generated_frames || 0) / Number(obj.max_new_tokens || 1));
    field("bar").style.width = pct.toFixed(1) + "%";
  }
}
function fetchJson(url, options) {
  return fetch(url, options).then(async (response) => {
    const text = await response.text();
    if (!response.ok) throw new Error(text || `HTTP ${response.status}`);
    return text ? JSON.parse(text) : {};
  });
}
function renderRuntime(status) {
  runtimeReady = status && status.state === "ready";
  const elapsed = status && status.load_elapsed_seconds != null ? ` | load=${Number(status.load_elapsed_seconds).toFixed(1)}s` : "";
  const extra = runtimeReady ? ` | n_vq=${status.n_vq} | sr=${status.sample_rate}` : "";
  const error = status && status.error ? ` | error=${status.error}` : "";
  field("runtime-summary").textContent = `Runtime: ${(status && status.state) || "unknown"}${elapsed}${extra}${error}`;
}
async function pollRuntime() {
  try {
    renderRuntime(await fetch(apiUrl("api/health")).then(r => r.json()));
  } catch (err) {
    runtimeReady = false;
    field("runtime-summary").textContent = `Runtime: unreachable (${err})`;
  }
}
function updateReferenceLabel() {
  const file = field("prompt-audio").files[0];
  const examplePath = field("example-audio-path").value;
  if (file) {
    field("selected-reference").textContent = `Uploaded reference: ${file.name}`;
  } else if (recordedReferenceFile) {
    field("selected-reference").textContent = `Recorded reference: ${recordedReferenceFile.name}`;
  } else if (examplePath) {
    field("selected-reference").textContent = `Example reference: ${examplePath}`;
  } else {
    field("selected-reference").textContent = "No reference selected.";
  }
  syncReferenceSourceControls();
  updateModeHint();
}
function clearReferencePreview() {
  if (currentReferenceObjectUrl) {
    URL.revokeObjectURL(currentReferenceObjectUrl);
    currentReferenceObjectUrl = null;
  }
  const preview = field("reference-audio-preview");
  preview.pause();
  preview.removeAttribute("src");
  preview.load();
  preview.classList.add("hidden");
  field("reference-drop-zone").classList.remove("has-reference");
  syncReferenceSourceControls();
}
function showReferencePreview(src, objectUrl = null) {
  if (currentReferenceObjectUrl) {
    URL.revokeObjectURL(currentReferenceObjectUrl);
    currentReferenceObjectUrl = null;
  }
  currentReferenceObjectUrl = objectUrl;
  const preview = field("reference-audio-preview");
  preview.src = src;
  preview.classList.remove("hidden");
  field("reference-drop-zone").classList.remove("hidden");
  field("reference-drop-zone").classList.add("has-reference");
  preview.load();
}
function updateReferencePreview() {
  const file = field("prompt-audio").files[0];
  const examplePath = field("example-audio-path").value;
  if (file) {
    const objectUrl = URL.createObjectURL(file);
    showReferencePreview(objectUrl, objectUrl);
    return;
  }
  if (recordedReferenceFile) {
    const objectUrl = URL.createObjectURL(recordedReferenceFile);
    showReferencePreview(objectUrl, objectUrl);
    return;
  }
  if (examplePath) {
    showReferencePreview(referenceAudioUrl(examplePath));
    return;
  }
  clearReferencePreview();
}
function syncReferenceSourceControls() {
  const uploadMode = referenceSourceMode === "upload";
  field("reference-source-upload").classList.toggle("active", uploadMode);
  field("reference-source-record").classList.toggle("active", !uploadMode);
  field("reference-source-upload").setAttribute("aria-pressed", uploadMode ? "true" : "false");
  field("reference-source-record").setAttribute("aria-pressed", uploadMode ? "false" : "true");
  field("reference-record-controls").classList.toggle("hidden", uploadMode);
  field("reference-drop-zone").classList.toggle("hidden", !uploadMode && !hasReference());
}
function setReferenceSourceMode(mode) {
  const nextMode = mode === "record" ? "record" : "upload";
  if (referenceRecordingActive && nextMode !== "record") stopReferenceRecording(true);
  referenceSourceMode = nextMode;
  if (nextMode === "upload") {
    recordedReferenceFile = null;
    field("reference-record-status").textContent = "Ready to record.";
  } else {
    field("prompt-audio").value = "";
    field("example-audio-path").value = "";
  }
  updateReferencePreview();
  updateReferenceLabel();
}
function formatRecordingDuration(seconds) {
  const safeSeconds = Math.max(0, Math.floor(seconds || 0));
  const minutes = Math.floor(safeSeconds / 60);
  const rest = safeSeconds % 60;
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}
function updateRecordButtonState() {
  const button = field("reference-record-button");
  button.classList.toggle("recording", referenceRecordingActive);
  field("reference-record-button-label").textContent = referenceRecordingActive ? "Stop Recording" : "Start Recording";
}
function updateRecordingStatus() {
  if (!referenceRecordingActive) return;
  const elapsed = (Date.now() - referenceRecordingStartedAt) / 1000;
  field("reference-record-status").textContent = `Recording ${formatRecordingDuration(elapsed)}`;
}
function flattenRecordingChunks(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}
function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}
function encodeWavPcm16(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return buffer;
}
function cleanupReferenceRecordingNodes() {
  if (referenceRecordingTimer) {
    window.clearInterval(referenceRecordingTimer);
    referenceRecordingTimer = null;
  }
  if (referenceRecordingProcessor) {
    try { referenceRecordingProcessor.disconnect(); } catch (err) {}
    referenceRecordingProcessor.onaudioprocess = null;
    referenceRecordingProcessor = null;
  }
  if (referenceRecordingSource) {
    try { referenceRecordingSource.disconnect(); } catch (err) {}
    referenceRecordingSource = null;
  }
  if (referenceRecordingStream) {
    for (const track of referenceRecordingStream.getTracks()) track.stop();
    referenceRecordingStream = null;
  }
  if (referenceRecordingAudioContext) {
    referenceRecordingAudioContext.close().catch(() => {});
    referenceRecordingAudioContext = null;
  }
}
async function startReferenceRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Microphone recording is not supported in this browser.");
  }
  field("prompt-audio").value = "";
  field("example-audio-path").value = "";
  recordedReferenceFile = null;
  clearReferencePreview();
  referenceRecordingChunks = [];
  referenceRecordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) throw new Error("Web Audio recording is not supported in this browser.");
  referenceRecordingAudioContext = new AudioContextCtor();
  referenceRecordingSampleRate = referenceRecordingAudioContext.sampleRate || 48000;
  referenceRecordingSource = referenceRecordingAudioContext.createMediaStreamSource(referenceRecordingStream);
  referenceRecordingProcessor = referenceRecordingAudioContext.createScriptProcessor(4096, 1, 1);
  referenceRecordingActive = true;
  referenceRecordingStartedAt = Date.now();
  referenceRecordingProcessor.onaudioprocess = (event) => {
    if (!referenceRecordingActive) return;
    const input = event.inputBuffer.getChannelData(0);
    referenceRecordingChunks.push(new Float32Array(input));
    event.outputBuffer.getChannelData(0).fill(0);
  };
  referenceRecordingSource.connect(referenceRecordingProcessor);
  referenceRecordingProcessor.connect(referenceRecordingAudioContext.destination);
  referenceRecordingTimer = window.setInterval(updateRecordingStatus, 200);
  updateRecordingStatus();
  updateRecordButtonState();
}
function stopReferenceRecording(discard = false) {
  if (!referenceRecordingActive) return;
  referenceRecordingActive = false;
  const chunks = referenceRecordingChunks;
  referenceRecordingChunks = [];
  cleanupReferenceRecordingNodes();
  updateRecordButtonState();
  if (discard) {
    field("reference-record-status").textContent = "Ready to record.";
    return;
  }
  if (!chunks.length) {
    field("reference-record-status").textContent = "No audio captured.";
    return;
  }
  const samples = flattenRecordingChunks(chunks);
  const wavBuffer = encodeWavPcm16(samples, referenceRecordingSampleRate);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  recordedReferenceFile = new File([wavBuffer], `recorded_reference_${stamp}.wav`, { type: "audio/wav" });
  field("reference-record-status").textContent = `Recorded ${formatRecordingDuration(samples.length / referenceRecordingSampleRate)}`;
  updateReferencePreview();
  updateReferenceLabel();
}
async function toggleReferenceRecording() {
  if (referenceRecordingActive) {
    stopReferenceRecording(false);
    return;
  }
  try {
    await startReferenceRecording();
  } catch (err) {
    cleanupReferenceRecordingNodes();
    referenceRecordingActive = false;
    updateRecordButtonState();
    field("reference-record-status").textContent = String(err);
  }
}
function clearReferenceAudio() {
  if (referenceRecordingActive) stopReferenceRecording(true);
  recordedReferenceFile = null;
  field("prompt-audio").value = "";
  field("example-audio-path").value = "";
  clearReferencePreview();
  updateReferenceLabel();
}
function updateModeHint() {
  const continuationMode = selectedMode() === "continuation" || selectedMode() === "continuation_clone";
  field("reference-transcript-panel").classList.toggle("hidden", !continuationMode);
  if (!hasReference()) {
    field("mode-hint").innerHTML = "Current mode: <b>Direct Generation</b> (no reference audio uploaded)";
  } else if (selectedMode() === "voice_clone") {
    field("mode-hint").innerHTML = "Current mode: <b>Clone</b> (speaker timbre will be cloned from the reference audio)";
  } else {
    field("mode-hint").innerHTML = `Current mode: <b>${selectedModeName()}</b><br><span class="hint">${CONTINUATION_NOTICE}</span>`;
  }
  updateDurationControls();
}
function detectTextLanguage(text) {
  const zh = (text.match(/[\u4e00-\u9fff]/g) || []).length;
  const en = (text.match(/[A-Za-z]/g) || []).length;
  if (zh === 0 && en === 0) return "en";
  return zh >= en ? "zh" : "en";
}
function supportsDurationControl() {
  return selectedMode() === "voice_clone";
}
function updateDurationControls() {
  const checkbox = field("tokens-control");
  const wrap = field("tokens-wrap");
  if (!supportsDurationControl()) {
    checkbox.checked = false;
    checkbox.disabled = true;
    wrap.classList.add("hidden");
    field("duration-hint").textContent = "Duration control is disabled for Continuation modes.";
    return;
  }
  checkbox.disabled = false;
  if (!checkbox.checked) {
    wrap.classList.add("hidden");
    field("duration-hint").textContent = "Duration control is disabled.";
    return;
  }
  const text = field("text").value || "";
  const lang = detectTextLanguage(text);
  const factor = lang === "zh" ? 3.098411951313033 : 0.8673376262755219;
  const defaultTokens = Math.max(1, Math.round(Math.max(text.length, 1) * factor));
  const minTokens = Math.max(1, Math.round(defaultTokens * 0.5));
  const maxTokens = Math.max(minTokens, Math.round(defaultTokens * 1.5));
  const current = Math.max(minTokens, Math.min(maxTokens, Number(field("tokens").value || defaultTokens)));
  field("tokens").min = String(minTokens);
  field("tokens").max = String(maxTokens);
  field("tokens").value = String(current);
  wrap.classList.remove("hidden");
  field("duration-hint").textContent = `Duration control enabled | detected language: ${lang === "zh" ? "Chinese" : "English"} | default=${defaultTokens}, range=[${minTokens}, ${maxTokens}]`;
}
function renderExamples() {
  const tbody = field("examples-body");
  tbody.innerHTML = "";
  for (const [index, example] of EXAMPLES.entries()) {
    const tr = document.createElement("tr");
    const role = document.createElement("td");
    role.className = "role-cell";
    role.textContent = example.role;
    const text = document.createElement("td");
    text.textContent = example.text;
    tr.append(role, text);
    tr.onclick = () => {
      field("text").value = example.text;
      field("example-audio-path").value = example.audio_path;
      if (LANGUAGES.includes(example.language)) field("language").value = example.language;
      if (referenceRecordingActive) stopReferenceRecording(true);
      recordedReferenceFile = null;
      referenceSourceMode = "upload";
      field("prompt-audio").value = "";
      syncReferenceSourceControls();
      updateReferencePreview();
      updateReferenceLabel();
      updateDurationControls();
      setStatus(`Example selected: ${example.role}`);
    };
    tbody.appendChild(tr);
  }
}
function setupLanguages() {
  const select = field("language");
  select.innerHTML = "";
  for (const language of LANGUAGES) {
    const option = document.createElement("option");
    option.value = language === "Auto (omit)" ? "" : language;
    option.textContent = language;
    select.appendChild(option);
  }
}
function setupRangePair(id, integer = false) {
  const range = field(`${id}-range`);
  const input = field(id);
  range.oninput = () => { input.value = range.value; };
  input.oninput = () => {
    let value = Number(input.value);
    const min = Number(input.min || range.min || 0);
    const max = Number(input.max || range.max || value);
    if (!Number.isFinite(value)) value = min;
    value = Math.max(min, Math.min(max, value));
    if (integer) value = Math.round(value);
    input.value = String(value);
    range.value = String(value);
  };
}
function mergeUint8Arrays(a, b) {
  const merged = new Uint8Array(a.length + b.length);
  merged.set(a, 0);
  merged.set(b, a.length);
  return merged;
}
function clearPlaybackCompletionTimer() {
  if (playbackCompletionTimer) {
    window.clearTimeout(playbackCompletionTimer);
    playbackCompletionTimer = null;
  }
}
function updatePauseButtonState() {
  const pauseBtn = field("pause");
  if (audioContext) {
    pauseBtn.disabled = false;
    pauseBtn.textContent = playbackPaused ? "Resume Playback" : "Pause Playback";
    return;
  }
  pauseBtn.disabled = true;
  pauseBtn.textContent = "Pause Playback";
}
function resolveInitialPlaybackDelaySeconds() {
  const raw = Number(field("initial-playback-delay").value || 0.08);
  return Number.isFinite(raw) ? Math.max(0.0, raw) : 0.08;
}
function setGenerationActive(active) {
  generationActive = Boolean(active);
  field("start").disabled = generationActive;
}
function resetRealtimePlaybackBuffer() {
  pendingRealtimePcmChunks = [];
  pendingRealtimePcmSeconds = 0;
  realtimePlaybackStarted = false;
}
function pcmChunkDurationSeconds(bytes, sampleRate, channels) {
  const bytesPerFrame = Math.max(1, Number(channels || 2) * 2);
  const frames = Math.floor(bytes.byteLength / bytesPerFrame);
  const resolvedSampleRate = Math.max(1, Number(sampleRate || 48000));
  return frames / resolvedSampleRate;
}
function flushPendingRealtimePcmChunks() {
  if (pendingRealtimePcmChunks.length === 0) return;
  const chunks = pendingRealtimePcmChunks;
  pendingRealtimePcmChunks = [];
  pendingRealtimePcmSeconds = 0;
  realtimePlaybackStarted = true;
  for (const chunk of chunks) {
    schedulePcmChunk(chunk.bytes, chunk.sampleRate, chunk.channels);
  }
}
function enqueueRealtimePcmChunk(bytes, sampleRate, channels) {
  if (realtimePlaybackStarted) {
    schedulePcmChunk(bytes, sampleRate, channels);
    return;
  }
  pendingRealtimePcmChunks.push({ bytes, sampleRate, channels });
  pendingRealtimePcmSeconds += pcmChunkDurationSeconds(bytes, sampleRate, channels);
  if (pendingRealtimePcmSeconds >= MIN_INITIAL_PLAYBACK_BUFFER_SECONDS) {
    flushPendingRealtimePcmChunks();
  }
}
function pcm16ToAudioBuffer(bytes, sampleRate, channels) {
  const bytesPerFrame = channels * 2;
  const frames = Math.floor(bytes.byteLength / bytesPerFrame);
  const buffer = audioContext.createBuffer(channels, frames, sampleRate);
  const view = new DataView(bytes.buffer, bytes.byteOffset, frames * bytesPerFrame);
  for (let channelIndex = 0; channelIndex < channels; channelIndex += 1) {
    const channelData = buffer.getChannelData(channelIndex);
    for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
      const byteOffset = (frameIndex * channels + channelIndex) * 2;
      channelData[frameIndex] = view.getInt16(byteOffset, true) / 32768.0;
    }
  }
  return buffer;
}
function schedulePcmChunk(bytes, sampleRate, channels) {
  if (!audioContext) {
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) throw new Error("This browser does not support Web Audio streaming playback.");
    audioContext = new AudioContextCtor({ sampleRate });
    nextPlaybackTime = 0;
    playbackPaused = false;
    updatePauseButtonState();
  }
  const buffer = pcm16ToAudioBuffer(bytes, sampleRate, channels);
  if (buffer.length === 0) return;
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  const startAt = Math.max(nextPlaybackTime || (audioContext.currentTime + currentInitialPlaybackDelaySeconds), audioContext.currentTime + 0.02);
  source.start(startAt);
  nextPlaybackTime = startAt + buffer.duration;
}
async function prepareRealtimePlayback(sampleRate) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) throw new Error("This browser does not support Web Audio streaming playback.");
  audioContext = new AudioContextCtor({ sampleRate });
  await audioContext.resume();
  currentInitialPlaybackDelaySeconds = resolveInitialPlaybackDelaySeconds();
  nextPlaybackTime = 0;
  resetRealtimePlaybackBuffer();
  playbackPaused = false;
  updatePauseButtonState();
}
async function closeRealtimeStream() {
  clearPlaybackCompletionTimer();
  if (statusTimer) {
    window.clearInterval(statusTimer);
    statusTimer = null;
  }
  if (currentStreamAbortController) {
    currentStreamAbortController.abort();
    currentStreamAbortController = null;
  }
  if (currentJob) {
    fetch(apiUrl(`api/generate-stream/${currentJob}/close`), { method: "POST" }).catch(() => {});
    currentJob = null;
  }
  if (audioContext) {
    try { await audioContext.close(); } catch (err) {}
    audioContext = null;
  }
  playbackPaused = false;
  nextPlaybackTime = 0;
  resetRealtimePlaybackBuffer();
  updatePauseButtonState();
  setGenerationActive(false);
}
function monitorPlaybackCompletion() {
  clearPlaybackCompletionTimer();
  if (!audioContext) return;
  const poll = async () => {
    if (!audioContext) return;
    if (playbackPaused || nextPlaybackTime - audioContext.currentTime > 0.05) {
      playbackCompletionTimer = window.setTimeout(() => poll().catch(() => {}), 120);
      return;
    }
    try { await audioContext.close(); } catch (err) {}
    audioContext = null;
    playbackPaused = false;
    nextPlaybackTime = 0;
    updatePauseButtonState();
  };
  playbackCompletionTimer = window.setTimeout(() => poll().catch(() => {}), 120);
}
async function streamAudio(jobId, sampleRate, channels) {
  const response = await fetch(apiUrl(`api/generate-stream/${jobId}/audio`), {
    signal: currentStreamAbortController ? currentStreamAbortController.signal : undefined,
  });
  if (!response.ok) throw new Error(await response.text());
  if (!response.body) throw new Error("ReadableStream is not available on this response.");
  const reader = response.body.getReader();
  const resolvedChannels = Number(channels || response.headers.get("X-Audio-Channels") || 2);
  const resolvedSampleRate = Number(sampleRate || response.headers.get("X-Audio-Sample-Rate") || 48000);
  const bytesPerFrame = resolvedChannels * 2;
  let remainder = new Uint8Array(0);
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (value && value.byteLength > 0) {
      const merged = mergeUint8Arrays(remainder, value);
      const alignedLength = Math.floor(merged.length / bytesPerFrame) * bytesPerFrame;
      if (alignedLength <= 0) {
        remainder = merged;
        continue;
      }
      enqueueRealtimePcmChunk(merged.subarray(0, alignedLength), resolvedSampleRate, resolvedChannels);
      remainder = merged.subarray(alignedLength);
    }
  }
  flushPendingRealtimePcmChunks();
  monitorPlaybackCompletion();
}
async function pollStatus(jobId) {
  const status = await fetchJson(apiUrl(`api/generate-stream/${jobId}/status`));
  setStatus(status);
  field("summary").textContent = `${status.state} | mode=${status.mode || selectedModeName()} | frames=${status.generated_frames || 0} | emitted=${Number(status.emitted_audio_seconds || 0).toFixed(2)}s | lead=${Number(status.lead_seconds || 0).toFixed(2)}s | playback_delay=${currentInitialPlaybackDelaySeconds.toFixed(2)}s`;
  if (status.state === "finished") {
    clearInterval(statusTimer);
    statusTimer = null;
    const result = await fetchJson(apiUrl(`api/generate-stream/${jobId}/result`));
    field("download").href = apiUrl(`api/generate-stream/${jobId}/result-audio`);
    field("download").style.display = "inline";
    const outputAudio = field("audio-output");
    outputAudio.src = apiUrl(`api/generate-stream/${jobId}/result-audio`);
    outputAudio.removeAttribute("disabled");
    outputAudio.load();
    setStatus({ ...status, result });
    if (!currentStreamingGenerationEnabled) {
      outputAudio.play().catch(err => {
        setStatus({ ...status, result, autoplay_error: String(err) });
      });
    }
    setGenerationActive(false);
  }
  if (status.state === "error" || status.state === "closed") {
    clearInterval(statusTimer);
    statusTimer = null;
    setGenerationActive(false);
  }
}
field("start").onclick = async () => {
  await closeRealtimeStream();
  setGenerationActive(true);
  field("download").style.display = "none";
  field("bar").style.width = "0%";
  field("summary").textContent = "";
  field("audio-output").pause();
  field("audio-output").setAttribute("disabled", "disabled");
  field("audio-output").removeAttribute("src");
  field("audio-output").load();
  const form = new FormData();
  form.append("mode", selectedMode());
  form.append("language", field("language").value);
  form.append("text", field("text").value);
  form.append("prompt_text", field("prompt-text").value);
  form.append("max_new_tokens", field("max-new-tokens").value);
  form.append("codec_chunk_frames", field("codec-chunk-frames").value);
  form.append("seed", field("seed").value);
  form.append("tokens_control", field("tokens-control").checked ? "1" : "0");
  form.append("tokens", field("tokens").value);
  form.append("temperature", field("temperature").value);
  form.append("top_p", field("top-p").value);
  form.append("top_k", field("top-k").value);
  form.append("repetition_penalty", field("repetition-penalty").value);
  currentStreamingGenerationEnabled = field("streaming-generation").checked;
  form.append("streaming_generation", currentStreamingGenerationEnabled ? "1" : "0");
  form.append("example_audio_path", field("example-audio-path").value);
  const file = field("prompt-audio").files[0] || recordedReferenceFile;
  if (file) form.append("prompt_audio", file);
  try {
    if ((selectedMode() === "continuation" || selectedMode() === "continuation_clone") && hasReference() && !field("prompt-text").value.trim()) {
      throw new Error("Reference Audio Transcript is required for Continuation modes.");
    }
    setStatus("starting...");
    const response = await fetch(apiUrl("api/generate-stream/start"), { method: "POST", body: form });
    if (!response.ok) throw new Error(await response.text());
    const start = await response.json();
    currentJob = start.job_id;
    currentInitialPlaybackDelaySeconds = resolveInitialPlaybackDelaySeconds();
    if (currentStreamingGenerationEnabled) {
      currentStreamAbortController = new AbortController();
      await prepareRealtimePlayback(start.sample_rate || 48000);
      streamAudio(currentJob, start.sample_rate || 48000, start.channels || 2).catch(err => {
        if (!(err && String(err).includes("AbortError"))) setStatus(String(err));
      });
    } else {
      currentStreamAbortController = null;
      resetRealtimePlaybackBuffer();
      updatePauseButtonState();
    }
    if (statusTimer) clearInterval(statusTimer);
    statusTimer = setInterval(() => pollStatus(currentJob), 500);
    pollStatus(currentJob);
  } catch (err) {
    setGenerationActive(false);
    setStatus(String(err));
  }
};
field("stop").onclick = () => closeRealtimeStream();
field("pause").onclick = async () => {
  if (!audioContext) return;
  if (playbackPaused) {
    await audioContext.resume();
    playbackPaused = false;
  } else {
    await audioContext.suspend();
    playbackPaused = true;
  }
  updatePauseButtonState();
};

field("text").value = DEFAULT_TEXT;
setupLanguages();
renderExamples();
renderRuntime(INITIAL_RUNTIME);
for (const id of ["temperature", "top-p", "top-k", "repetition-penalty", "max-new-tokens", "codec-chunk-frames", "seed"]) {
  setupRangePair(id, ["top-k", "max-new-tokens", "codec-chunk-frames", "seed"].includes(id));
}
field("tokens-control").onchange = updateDurationControls;
field("text").oninput = updateDurationControls;
field("prompt-audio").onchange = () => {
  if (field("prompt-audio").files[0]) {
    if (referenceRecordingActive) stopReferenceRecording(true);
    recordedReferenceFile = null;
    referenceSourceMode = "upload";
    field("example-audio-path").value = "";
    syncReferenceSourceControls();
  }
  updateReferencePreview();
  updateReferenceLabel();
};
field("reference-source-upload").onclick = () => setReferenceSourceMode("upload");
field("reference-source-record").onclick = () => setReferenceSourceMode("record");
field("reference-record-button").onclick = () => toggleReferenceRecording();
field("clear-reference").onclick = clearReferenceAudio;
for (const radio of document.querySelectorAll("input[name='mode']")) {
  radio.onchange = updateModeHint;
}
syncReferenceSourceControls();
updateRecordButtonState();
updateReferenceLabel();
setInterval(pollRuntime, 1500);
pollRuntime();
</script>
</body>
</html>
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MOSS-TTS Local v1.5 realtime streaming app.")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7861")))
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    parser.add_argument("--codec-dir", default=os.environ.get("CODEC_DIR", str(DEFAULT_CODEC_DIR)))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--upload-dir", default=os.environ.get("UPLOAD_DIR", str(DEFAULT_UPLOAD_DIR)))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--tts-device", default=os.environ.get("TTS_DEVICE", "cuda:0"))
    parser.add_argument("--codec-device", default=os.environ.get("CODEC_DEVICE", os.environ.get("TTS_DEVICE", "cuda:0")))
    parser.add_argument("--dtype", default=os.environ.get("TTS_DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
    parser.add_argument(
        "--codec-weight-dtype",
        default=os.environ.get("CODEC_WEIGHT_DTYPE", "fp32"),
        choices=["bf16", "bfloat16", "fp32", "float32"],
        help="Codec encoder/decoder parameter dtype. Defaults to fp32; pass bf16 to reduce memory. The quantizer stays fp32.",
    )
    parser.add_argument(
        "--codec-compute-dtype",
        default=os.environ.get("CODEC_COMPUTE_DTYPE", "bf16"),
        choices=["bf16", "fp32"],
        help="Codec non-quantizer autocast compute dtype.",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = create_app(
        model_dir=args.model_dir,
        codec_dir=args.codec_dir,
        output_dir=args.output_dir,
        upload_dir=args.upload_dir,
        device=args.device,
        tts_device=args.tts_device or args.device,
        codec_device=args.codec_device or args.tts_device or args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        codec_weight_dtype=args.codec_weight_dtype,
        codec_compute_dtype=args.codec_compute_dtype,
        warmup=not args.no_warmup,
        preload=not args.no_preload,
    )
    uvicorn.run(app, host=args.host, port=int(args.port))


if __name__ == "__main__":
    main()
