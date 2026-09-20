"""Command line entry point; help works without loading Torch or model weights."""

import argparse
import json
from contextlib import closing
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="MOSS-TTS v1.5 8B streaming and voice cloning")
    commands = root.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("synthesize", help="Stream speech into a WAV file")
    serve = commands.add_parser("serve", help="Run the optional streaming HTTP server")
    for command in (synth, serve):
        command.add_argument("--preset", choices=("bf16", "gptq"), default="bf16")
        command.add_argument("--calibration-path", type=Path)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--cache-dir", type=Path)
        command.add_argument("--local-files-only", action="store_true")
    synth.add_argument("--text", required=True)
    synth.add_argument("--reference", type=Path, help="0.2–15 seconds of reference audio")
    synth.add_argument("--language", default="English")
    synth.add_argument("--output", type=Path, default=Path("output.wav"))
    synth.add_argument("--max-new-tokens", type=int, default=400)
    synth.add_argument("--seed", type=int, default=1234)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    calibrate = commands.add_parser(
        "calibrate", help="Export G32 weights from a JSONL voice dataset"
    )
    calibrate.add_argument("--dataset", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--local-files-only", action="store_true")
    return root


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.command == "calibrate":
        from .calibration import export

        export(args.dataset, args.output, local_files_only=args.local_files_only)
        return
    options = {
        name: getattr(args, name)
        for name in ("preset", "calibration_path", "device", "cache_dir", "local_files_only")
    }
    if args.command == "serve":
        try:
            import uvicorn

            from .server import create_app
        except ImportError as error:
            raise SystemExit('Install the server extra: pip install ".[server]"') from error
        uvicorn.run(create_app(**options), host=args.host, port=args.port, workers=1)
        return
    import soundfile as sf

    from .api import MossTTS

    with MossTTS.from_pretrained(**options) as tts:
        voice = tts.clone_voice(args.reference) if args.reference else None
        stream = tts.stream(
            args.text,
            voice=voice,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        with (
            closing(stream),
            sf.SoundFile(
                args.output, mode="w", samplerate=tts.sample_rate, channels=1, subtype="PCM_16"
            ) as output,
        ):
            for chunk in stream:
                output.write(chunk.pcm.numpy())
        if not tts.metrics["frames"]:
            raise SystemExit("No audio generated; increase --max-new-tokens")
        print(json.dumps({"output": str(args.output), **tts.metrics}, indent=2))


if __name__ == "__main__":
    main()
