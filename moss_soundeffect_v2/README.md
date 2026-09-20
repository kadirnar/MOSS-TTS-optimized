# MOSS-SoundEffect v2

**MOSS-SoundEffect v2.0** is a text-to-audio model with a Diffusion
Transformer (DiT) backbone trained with the Flow Matching objective,
paired with a DAC VAE and a Qwen3 text encoder.

## News

* **MOSS-SoundEffect v2.0** *(this directory)* — A new attempt at text-to-audio
  generation using a **DiT** backbone trained with the **Flow Matching**
  objective, replacing the discrete-token autoregressive backbone used in v1.
  Targets higher audio fidelity and more natural long-form environmental
  sound. Released on HuggingFace at
  [`OpenMOSS-Team/MOSS-SoundEffect-v2.0`](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0).
* **MOSS-SoundEffect v1.0** — The first release, built on the `MossTTSDelay`
  discrete-token autoregressive architecture shared with the rest of the
  MOSS-TTS family. See the
  [v1 model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_sound_effect_model_card.md)
  for architecture and usage details.

> **Note.** This subdirectory uses its **own** Python environment (Python
> 3.12, pinned `numpy==1.26`, `transformers==4.57`, `torch==2.9`) and is
> **not compatible** with the top-level MOSS-TTS environment. Install it in
> a clean, isolated environment as shown in
> [Environment Setup](#environment-setup) below.

## Environment Setup

We recommend a clean, isolated Python 3.12 environment to avoid dependency
conflicts with the top-level MOSS-TTS environment.

### Using Conda

```bash
conda create -n moss-soundeffect-v2 python=3.12 -y
conda activate moss-soundeffect-v2
```

Clone the repository and install all required dependencies:

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS/moss_soundeffect_v2
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e ".[torch-cu128,finetune]"
```

For a minimal **inference-only** install (still ships the Gradio demo;
skips the fine-tuning extras `accelerate` / `peft` / `pandas` / `torchcodec`):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e ".[torch-cu128]"
```

## Inference

```python
import torch
from moss_soundeffect_v2 import MossSoundEffectPipeline

pipe = MossSoundEffectPipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-SoundEffect-v2.0",   # HF hub repo id, or a local dir
    torch_dtype=torch.bfloat16,
    device="cuda",
)

audio = pipe(
    prompt="The crisp, rhythmic click-clack of fast typing on a mechanical keyboard.",
    seconds=10,
    num_inference_steps=100,
    cfg_scale=4.0,
)                                        # (B, C, T) waveform tensor
pipe.save_audio(audio, "out.wav")
```

Command-line: `bash infer_from_pipeline.sh`

> The bundled shell scripts accept either a HF hub repo id or a
> local directory; weights are auto-downloaded into the HuggingFace cache
> on first use.

> The underlying DiT is wrapped with `torch.compile` + Triton CUDA Graph for
> acceleration. The first call may take a few minutes to compile. If you hit
> `TorchDynamo` / Triton compile errors, set `TORCHDYNAMO_DISABLE=1` before
> launching Python — the bundled shell scripts already do this.

## Gradio demo

```bash
SOUNDEFFECT_MODEL_DIR=OpenMOSS-Team/MOSS-SoundEffect-v2.0 \
  python ../clis/moss_sound_effect_app.py
```

## Fine-tuning

Full-parameter DiT fine-tune from an existing HF directory:

```bash
HF_MODEL_DIR=OpenMOSS-Team/MOSS-SoundEffect-v2.0 \
METADATA_PATH=/path/to/captions.jsonl \
OUTPUT_PATH=./output/my_finetune \
  bash finetuning/finetuning.sh
```

### Metadata format

`METADATA_PATH` is a JSON Lines file with two required fields per line: `audio`
(path to the audio file, relative to `--dataset_base_path`) and `prompt`
(caption text in English or Chinese).

```jsonl
{"audio": "wavs/birdsong.wav", "prompt": "Birds chirp in the early morning, their calls clear and pleasant."}
{"audio": "wavs/brushing_teeth.wav", "prompt": "The soft scraping sound of toothbrush bristles brushing teeth."}
{"audio": "wavs/pooring_water.wav", "prompt": "Pouring water into a glass, clear liquid flowing sound, pitch rising as the glass fills up, refreshing."}
```

## Export a fine-tuned checkpoint

Training auto-exports the latest checkpoint to `<OUTPUT_PATH>/hf_format/`.
To convert any other fine-tuned DiT `.safetensors` checkpoint (e.g. an
intermediate `epoch-0.safetensors`) into a HF directory without re-running
training:

```bash
CKPT_PATH=/path/to/output/finetune/epoch-0.safetensors \
SOURCE_HF_DIR=OpenMOSS-Team/MOSS-SoundEffect-v2.0 \
OUTPUT_DIR=./output/finetune/hf_format_epoch0 \
  bash finetuning/export_to_hf.sh
```

`SOURCE_HF_DIR` is the HF directory (or hub repo id) you fine-tuned from.
Its frozen sub-modules (VAE / text encoder / tokenizer / scheduler) are
copied unchanged into the output, so you do **not** need to re-download
Qwen3 or the DAC VAE. The resulting directory can be loaded by
`MossSoundEffectPipeline.from_pretrained(OUTPUT_DIR)`.

