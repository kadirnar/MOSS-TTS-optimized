#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT_PATH=${CKPT_PATH:-"/path/to/output/finetune/epoch-0.safetensors"}
SOURCE_HF_DIR=${SOURCE_HF_DIR:-"/path/to/SoundEffect-v2-hf"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/finetune/hf_format"}

python finetuning/export_to_hf.py \
  --ckpt_path "$CKPT_PATH" \
  --source_hf_dir "$SOURCE_HF_DIR" \
  --output_dir "$OUTPUT_DIR"
