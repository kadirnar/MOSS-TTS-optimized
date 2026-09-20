#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

EXP_NAME=$(basename "$0" .sh)

HF_MODEL_DIR=${HF_MODEL_DIR:-"/path/to/SoundEffect-v2-hf"}
METADATA_PATH=${METADATA_PATH:-"/path/to/captions.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-"./output/${EXP_NAME}"}
CACHE_FOLDER=${CACHE_FOLDER:-"${OUTPUT_PATH}/data_cache"}

SAMPLE_RATE=${SAMPLE_RATE:-48000}
NUM_AUDIO_SAMPLES=${NUM_AUDIO_SAMPLES:-1440000}
MIN_NUM_AUDIO_SAMPLES=${MIN_NUM_AUDIO_SAMPLES:-960}
MAX_NUM_AUDIO_SAMPLES=${MAX_NUM_AUDIO_SAMPLES:-1440000}
DATASET_REPEAT=${DATASET_REPEAT:-1}
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-4}
DROP_PROMPT_PROB=${DROP_PROMPT_PROB:-0.1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_EPOCHS=${NUM_EPOCHS:-5}
CLIP_GRAD_NORM=${CLIP_GRAD_NORM:-0.1}
TRAINABLE_MODELS=${TRAINABLE_MODELS:-"dit"}
REMOVE_PREFIX_IN_CKPT=${REMOVE_PREFIX_IN_CKPT:-"pipe.dit."}

mkdir -p "$OUTPUT_PATH"

# Set NO_CACHE=1 to disable the one-shot VAE+text-encoder cache.
CACHE_ARGS=()
if [ -z "${NO_CACHE:-}" ]; then
  CACHE_ARGS+=(--cache_folder "$CACHE_FOLDER" --cache_first)
fi

accelerate launch \
  --mixed_precision bf16 \
  finetuning/finetuning.py \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dataset_base_path / \
  --dataset_metadata_path "$METADATA_PATH" \
  --sample_rate "$SAMPLE_RATE" \
  --num_audio_samples "$NUM_AUDIO_SAMPLES" \
  --min_num_audio_samples "$MIN_NUM_AUDIO_SAMPLES" \
  --max_num_audio_samples "$MAX_NUM_AUDIO_SAMPLES" \
  --mono \
  --data_file_keys "audio" \
  --dataset_repeat "$DATASET_REPEAT" \
  --dataset_num_workers "$DATASET_NUM_WORKERS" \
  --drop_prompt_prob "$DROP_PROMPT_PROB" \
  --append_duration_suffix \
  --duration_precision 1 \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_epochs "$NUM_EPOCHS" \
  --clip_grad_norm "$CLIP_GRAD_NORM" \
  --trainable_models "$TRAINABLE_MODELS" \
  --remove_prefix_in_ckpt "$REMOVE_PREFIX_IN_CKPT" \
  --output_path "$OUTPUT_PATH" \
  --log_dir "$OUTPUT_PATH" \
  "${CACHE_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_PATH/finetuning.log"
