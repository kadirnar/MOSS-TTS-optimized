"""Export a fine-tuned DiT checkpoint to the public HF layout."""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_DIR = _HERE.parent
_PROJECT_DIR = _PKG_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from moss_soundeffect_v2.hf_export import export_finetuned_to_hf


def main():
    parser = argparse.ArgumentParser(
        description="Convert a fine-tuned DiT .safetensors checkpoint into a "
                    "public HF directory loadable by MossSoundEffectPipeline.from_pretrained."
    )
    parser.add_argument(
        "--ckpt_path", required=True, type=Path,
        help="Path to the fine-tuned DiT .safetensors checkpoint.",
    )
    parser.add_argument(
        "--source_hf_dir", required=True, type=Path,
        help="Existing public HF dir whose VAE / text_encoder / tokenizer / "
             "scheduler / package sources will be reused.",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Destination directory for the exported HF model.",
    )
    args = parser.parse_args()

    export_finetuned_to_hf(
        ckpt_path=args.ckpt_path,
        source_hf_dir=args.source_hf_dir,
        dst_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
