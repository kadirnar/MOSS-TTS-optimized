"""Inference entry point for MossSoundEffectPipeline."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio

_HERE = Path(__file__).resolve().parent
_PROJECT_DIR = _HERE.parent
sys.path.insert(0, str(_PROJECT_DIR))

from moss_soundeffect_v2 import MossSoundEffectPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--prompt", default="The crisp, rhythmic click-clack of fast typing on a mechanical keyboard.")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--output", default="output_pipeline.wav")
    args = parser.parse_args()

    torch_dtype = getattr(torch, args.torch_dtype)

    print(f"Loading pipeline from {args.model_dir} ...")
    pipe = MossSoundEffectPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device=args.device,
    )

    audio = pipe(
        prompt=args.prompt,
        seconds=args.seconds,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
    )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torchaudio.save(output_path, audio[0].detach().cpu(), pipe.sample_rate)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
