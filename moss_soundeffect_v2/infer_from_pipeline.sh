#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL_DIR=${MODEL_DIR:-"${SOUNDEFFECT_MODEL_DIR:-/path/to/SoundEffect-v2-hf}"}
PROMPT=${PROMPT:-"The crisp, rhythmic click-clack of fast typing on a mechanical keyboard."}
SECONDS_=${SECONDS_:-10.0}
STEPS=${STEPS:-100}
CFG_SCALE=${CFG_SCALE:-4.0}
SIGMA_SHIFT=${SIGMA_SHIFT:-5.0}
SEED=${SEED:-0}
DEVICE=${DEVICE:-"cuda"}
TORCH_DTYPE=${TORCH_DTYPE:-"bfloat16"}
OUTPUT=${OUTPUT:-"output/output_pipeline.wav"}

mkdir -p "$(dirname "$OUTPUT")"

TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1} \
  python infer_from_pipeline.py \
    --model_dir "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --seconds "$SECONDS_" \
    --steps "$STEPS" \
    --cfg_scale "$CFG_SCALE" \
    --sigma_shift "$SIGMA_SHIFT" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --torch_dtype "$TORCH_DTYPE" \
    --output "$OUTPUT"
