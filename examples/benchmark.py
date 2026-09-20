"""Compare upstream and optimized TTFA at 48 kHz with fresh/cached voices.

Each backend runs in a separate process. The baseline observes upstream
model.generate through a forward pre-hook and decodes its first complete
32-codebook frame, without changing the generation loop or model weights.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path


class FirstAudioReady(Exception):
    pass


def measure_backend(args):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import transformers
    import triton

    from moss_tts import AudioChunk, MossTTS
    from moss_tts.models import CODEC_REVISION, TTS_REVISION, load_models
    from moss_tts.resampling import StreamingResampler, output_resampler

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.backend == "base":
        from moss_tts._model import modeling_moss_tts
        from moss_tts._runtime.codec import StreamingCodec

        modeling_moss_tts.tqdm = lambda iterable, **_: iterable
        model, codec, processor = load_models(local_files_only=True)
        decoder = StreamingCodec(codec, graph=False, cached_codebooks=False, triton_attention=False)
        kernel = output_resampler()

        def clone_voice():
            data, rate = sf.read(args.reference, dtype="float32", always_2d=True)
            return processor.encode_audios_from_wav(
                [torch.from_numpy(data.T.copy())], rate, n_vq=32
            )[0]

        def first_pcm(voice):
            decoder.reset()
            inputs = processor(
                [
                    [
                        processor.build_user_message(
                            text=args.text, reference=[voice], language="English"
                        )
                    ]
                ],
                mode="generation",
            ).to("cuda")
            history = []
            first_audio_row = None
            result = None

            def observe(_module, _positional, kwargs):
                nonlocal first_audio_row, result
                ids = kwargs["input_ids"]
                if ids.shape[1] != 1:
                    return
                history.append(ids[0, 0])
                token = int(ids[0, 0, 0].item())
                if token == model.config.audio_start_token_id:
                    first_audio_row = len(history)
                if first_audio_row is not None and len(history) >= first_audio_row + 32:
                    channel = torch.arange(32, device="cuda")
                    codes = torch.stack(history)[first_audio_row + channel, channel + 1]
                    assert bool((codes < 1024).all())
                    native = decoder.decode(codes.reshape(32, 1, 1)).float().flatten().cpu()
                    pcm = StreamingResampler(kernel).push(native)
                    result = (AudioChunk(pcm, 0, 0.0).pcm16(), time.perf_counter())
                    raise FirstAudioReady

            handle = model.register_forward_pre_hook(observe, with_kwargs=True)
            try:
                model.generate(**inputs, max_new_tokens=400)
            except FirstAudioReady:
                pass
            finally:
                handle.remove()
            if result is None:
                raise RuntimeError("The base model did not produce a complete audio frame")
            return result

        cleanup = decoder.close
    else:
        options = {"preset": args.backend, "local_files_only": True}
        if args.backend == "gptq":
            options["calibration_path"] = args.calibration
        tts = MossTTS.from_pretrained(**options)

        def clone_voice():
            return tts.clone_voice(args.reference)

        def first_pcm(voice):
            with closing(tts.stream(args.text, voice=voice, seed=args.seed)) as stream:
                chunk = next(stream)
                assert chunk.sample_rate == 48000
                return chunk.pcm16(), time.perf_counter()

        cleanup = tts.close

    rows = {"cached_voice": [], "fresh_voice": []}
    pcm_hashes = {key: set() for key in rows}
    first_samples = set()
    with torch.inference_mode():
        cached = clone_voice()
        codes = cached if args.backend == "base" else cached.codes
        voice_hash = hashlib.sha256(codes.numpy().tobytes()).hexdigest()
        for index in range(args.warmups + args.rounds):
            order = list(rows) if index % 2 == 0 else list(reversed(rows))
            for workload in order:
                torch.cuda.synchronize()
                torch.cuda.manual_seed(args.seed)
                started = time.perf_counter()
                voice = cached if workload == "cached_voice" else clone_voice()
                encoded_ms = (time.perf_counter() - started) * 1000
                pcm, ready_at = first_pcm(voice)
                elapsed = (ready_at - started) * 1000
                assert pcm and len(pcm) % 2 == 0
                first_samples.add(len(pcm) // 2)
                if index >= args.warmups:
                    rows[workload].append({"ttfa_ms": elapsed, "reference_ms": encoded_ms})
                    pcm_hashes[workload].add(hashlib.sha256(pcm).hexdigest())
        memory_gib = torch.cuda.max_memory_allocated() / 1024**3
        cleanup()
    info = sf.info(args.reference)
    result = {
        "backend": args.backend,
        "hardware": torch.cuda.get_device_name(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "transformers": transformers.__version__,
            "triton": triton.__version__,
        },
        "model_revision": TTS_REVISION,
        "codec_revision": CODEC_REVISION,
        "sample_rate": 48000,
        "native_sample_rate": 24000,
        "codebooks": 32,
        "first_chunk_samples": sorted(first_samples),
        "reference": {
            "sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
            "sample_rate": info.samplerate,
            "seconds": info.duration,
            "codes_sha256": voice_hash,
        },
        "peak_memory_gib": memory_gib,
        "workloads": {},
    }
    for name, samples in rows.items():
        times = [sample["ttfa_ms"] for sample in samples]
        result["workloads"][name] = {
            "median_ms": statistics.median(times),
            "p95_ms": float(np.percentile(times, 95)),
            "samples": samples,
            "pcm_sha256": sorted(pcm_hashes[name]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--text", default="Hello, this is a voice cloning test.")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("base", "bf16", "gptq"))
    args = parser.parse_args()
    if args.rounds < 1 or args.warmups < 1:
        parser.error("rounds and warmups must be positive")
    if args.backend:
        result = measure_backend(args)
    else:
        result = {
            "text": args.text,
            "seed": args.seed,
            "rounds": args.rounds,
            "warmups": args.warmups,
            "boundary": "Warm model, complete text to first playable 48-kHz s16le PCM; "
            "includes text processing, 32-codebook generation, codec decoding, "
            "continuous resampling and PCM conversion. Fresh voice also includes "
            "reading and encoding the reference WAV. Excludes loading, graph warmup, "
            "network transport, cancellation cleanup and subsequent audio. Baseline forward observer overhead included.",
            "backends": {},
        }
        for name in ("base", "bf16", "gptq"):
            part = args.output.with_name(args.output.stem + f".{name}.json")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--backend",
                name,
                "--reference",
                str(args.reference.resolve()),
                "--calibration",
                str(args.calibration.resolve()),
                "--text",
                args.text,
                "--rounds",
                str(args.rounds),
                "--warmups",
                str(args.warmups),
                "--seed",
                str(args.seed),
                "--output",
                str(part.resolve()),
            ]
            subprocess.run(command, check=True, env=os.environ.copy())
            result["backends"][name] = json.loads(part.read_text())
            part.unlink()
        for name, backend in result["backends"].items():
            for workload, data in backend["workloads"].items():
                baseline = result["backends"]["base"]["workloads"][workload]["median_ms"]
                data["speedup"] = baseline / data["median_ms"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"backend": args.backend or "all", "output": str(args.output)}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
