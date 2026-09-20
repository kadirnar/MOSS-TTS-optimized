"""Gradio demo for the moss_soundeffect_v2 subproject."""

import argparse
import functools
import os
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from moss_soundeffect_v2 import MossSoundEffectPipeline


DEFAULT_MODEL_DIR = os.environ.get("SOUNDEFFECT_MODEL_DIR", "/path/to/SoundEffect-v2-hf")
DEFAULT_DEVICE = os.environ.get("SOUNDEFFECT_DEVICE", "cuda")
MAX_INFERENCE_SECONDS = 30


@functools.lru_cache(maxsize=1)
def load_backend(model_dir: str, device_str: str):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    pipe = MossSoundEffectPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device=str(device),
    )
    return pipe, device


def run_inference(
    prompt: str,
    seconds: float,
    steps: int,
    cfg_scale: float,
    sigma_shift: float,
    seed: int,
    model_dir: str,
    device: str,
):
    if not (prompt or "").strip():
        raise ValueError("Please enter a prompt describing the audio you want to generate.")

    seconds = round(float(seconds), 1)
    if seconds <= 0:
        raise ValueError("Duration must be greater than 0.")
    if seconds > MAX_INFERENCE_SECONDS:
        raise ValueError(f"Duration must be no greater than {MAX_INFERENCE_SECONDS}s.")

    started_at = time.monotonic()
    pipe, torch_device = load_backend(model_dir=model_dir, device_str=device)

    audio = pipe(
        prompt=prompt,
        seconds=seconds,
        num_inference_steps=int(steps),
        cfg_scale=float(cfg_scale),
        sigma_shift=float(sigma_shift),
        seed=int(seed),
    )

    audio_np = audio[0].detach().float().cpu().numpy()
    if audio_np.ndim > 1 and audio_np.shape[0] == 1:
        audio_np = audio_np.squeeze(0)
    elif audio_np.ndim > 1:
        audio_np = audio_np.T
    audio_np = audio_np.astype(np.float32, copy=False)

    elapsed = time.monotonic() - started_at
    status = (
        f"Done | elapsed: {elapsed:.2f}s | "
        f"duration={seconds:.1f}s, steps={int(steps)}, "
        f"cfg_scale={float(cfg_scale):.2f}, sigma_shift={float(sigma_shift):.2f}, "
        f"seed={int(seed)}"
    )
    return (pipe.sample_rate, audio_np), status


def build_demo(args: argparse.Namespace):
    custom_css = """
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #111418;
      --muted: #4d5562;
      --line: #e5e7eb;
      --accent: #0f766e;
    }
    .gradio-container {
      background: linear-gradient(180deg, #f7f8fa 0%, #f3f5f7 100%);
      color: var(--ink);
    }
    .app-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      padding: 14px;
    }
    .app-title {
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .app-subtitle {
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 8px;
    }
    #run-btn {
      background: var(--accent);
      border: none;
    }
    """

    with gr.Blocks(title="MOSS-SoundEffect v2.0", css=custom_css) as demo:
        gr.Markdown(
            """
            <div class="app-card">
              <div class="app-title">MOSS-SoundEffect v2.0</div>
              <div class="app-subtitle">Text-to-audio diffusion (MossSoundEffectPipeline).</div>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=8,
                    value="The crisp, rhythmic click-clack of fast typing on a mechanical keyboard.",
                )
                seconds = gr.Slider(
                    minimum=1, maximum=MAX_INFERENCE_SECONDS, step=0.1, value=10,
                    label="Duration (seconds)",
                )
                with gr.Accordion("Sampling Parameters", open=True):
                    steps = gr.Slider(minimum=10, maximum=150, step=1, value=100, label="num_inference_steps")
                    cfg_scale = gr.Slider(minimum=1.0, maximum=8.0, step=0.1, value=4.0, label="cfg_scale")
                    sigma_shift = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=5.0, label="sigma_shift")
                    seed = gr.Number(value=0, label="seed", precision=0)
                run_btn = gr.Button("Generate Sound Effect", variant="primary", elem_id="run-btn")
            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Output Audio", type="numpy")
                status = gr.Textbox(label="Status", lines=4, interactive=False)

        run_btn.click(
            fn=lambda prompt, seconds, steps, cfg_scale, sigma_shift, seed: run_inference(
                prompt=prompt,
                seconds=seconds,
                steps=steps,
                cfg_scale=cfg_scale,
                sigma_shift=sigma_shift,
                seed=seed,
                model_dir=args.model_dir,
                device=args.device,
            ),
            inputs=[prompt, seconds, steps, cfg_scale, sigma_shift, seed],
            outputs=[output_audio, status],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="MossSoundEffect Gradio Demo")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--root_path",
        type=str,
        default=os.environ.get("GRADIO_ROOT_PATH"),
        help="Mount path when serving behind a reverse proxy.",
    )
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"[Startup] Preloading backend: model_dir={args.model_dir}, device={args.device}", flush=True)
    started = time.monotonic()
    load_backend(model_dir=args.model_dir, device_str=args.device)
    print(f"[Startup] Backend ready in {time.monotonic() - started:.2f}s", flush=True)

    demo = build_demo(args)
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        root_path=args.root_path,
        share=args.share,
    )


if __name__ == "__main__":
    main()
