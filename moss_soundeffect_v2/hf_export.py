"""Utilities for exporting fine-tuned DiT checkpoints to the public HF layout."""

import os
import shutil
from pathlib import Path
from typing import Union

from safetensors.torch import load_file, save_file


# DiT key mapping: custom WanAudioModel naming → diffusers naming.
_BLOCK_RENAME = {
    "self_attn.norm_k.weight":  "attn1.norm_k.weight",
    "self_attn.norm_q.weight":  "attn1.norm_q.weight",
    "self_attn.k.bias":         "attn1.to_k.bias",
    "self_attn.k.weight":       "attn1.to_k.weight",
    "self_attn.o.bias":         "attn1.to_out.0.bias",
    "self_attn.o.weight":       "attn1.to_out.0.weight",
    "self_attn.q.bias":         "attn1.to_q.bias",
    "self_attn.q.weight":       "attn1.to_q.weight",
    "self_attn.v.bias":         "attn1.to_v.bias",
    "self_attn.v.weight":       "attn1.to_v.weight",
    "cross_attn.norm_k.weight": "attn2.norm_k.weight",
    "cross_attn.norm_q.weight": "attn2.norm_q.weight",
    "cross_attn.k.bias":        "attn2.to_k.bias",
    "cross_attn.k.weight":      "attn2.to_k.weight",
    "cross_attn.o.bias":        "attn2.to_out.0.bias",
    "cross_attn.o.weight":      "attn2.to_out.0.weight",
    "cross_attn.q.bias":        "attn2.to_q.bias",
    "cross_attn.q.weight":      "attn2.to_q.weight",
    "cross_attn.v.bias":        "attn2.to_v.bias",
    "cross_attn.v.weight":      "attn2.to_v.weight",
    "ffn.0.bias":               "ffn.net.0.proj.bias",
    "ffn.0.weight":             "ffn.net.0.proj.weight",
    "ffn.2.bias":               "ffn.net.2.bias",
    "ffn.2.weight":             "ffn.net.2.weight",
    "norm3.bias":               "norm2.bias",
    "norm3.weight":             "norm2.weight",
    "modulation":               "scale_shift_table",
}

_GLOBAL_RENAME = {
    "text_embedding.0.bias":    "condition_embedder.text_embedder.linear_1.bias",
    "text_embedding.0.weight":  "condition_embedder.text_embedder.linear_1.weight",
    "text_embedding.2.bias":    "condition_embedder.text_embedder.linear_2.bias",
    "text_embedding.2.weight":  "condition_embedder.text_embedder.linear_2.weight",
    "time_embedding.0.bias":    "condition_embedder.time_embedder.linear_1.bias",
    "time_embedding.0.weight":  "condition_embedder.time_embedder.linear_1.weight",
    "time_embedding.2.bias":    "condition_embedder.time_embedder.linear_2.bias",
    "time_embedding.2.weight":  "condition_embedder.time_embedder.linear_2.weight",
    "time_projection.1.bias":   "condition_embedder.time_proj.bias",
    "time_projection.1.weight": "condition_embedder.time_proj.weight",
    "head.modulation":          "scale_shift_table",
    "head.head.bias":           "proj_out.bias",
    "head.head.weight":         "proj_out.weight",
    "patch_embedding.bias":     "patch_embedding.bias",
    "patch_embedding.weight":   "patch_embedding.weight",
}


def convert_dit_keys(state_dict: dict) -> dict:
    """Custom WanAudioModel state_dict → diffusers state_dict."""
    out = {}
    for key, param in state_dict.items():
        if key in _GLOBAL_RENAME:
            out[_GLOBAL_RENAME[key]] = param
        elif key.startswith("blocks."):
            _, block_idx, suffix = key.split(".", 2)
            if suffix in _BLOCK_RENAME:
                out[f"blocks.{block_idx}.{_BLOCK_RENAME[suffix]}"] = param
            else:
                out[key] = param
        else:
            out[key] = param
    return out


def _copy_tree(src: str, dst: str) -> None:
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _resolve_local_dir(path_or_repo_id: str) -> str:
    """Return a local dir for ``path_or_repo_id``.

    If it's already a local directory, return it unchanged. Otherwise treat
    it as a HuggingFace Hub repo id and ``snapshot_download`` it into the
    cache, returning the cached local path.
    """
    if os.path.isdir(path_or_repo_id):
        return path_or_repo_id
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=path_or_repo_id)


def export_finetuned_to_hf(
    ckpt_path: Union[str, os.PathLike],
    source_hf_dir: Union[str, os.PathLike],
    dst_dir: Union[str, os.PathLike],
) -> None:
    """Rebuild a public HF dir from a fine-tuned DiT checkpoint.

    Only the DiT is replaced — VAE / text_encoder / tokenizer / scheduler /
    package sources are copied from ``source_hf_dir`` unchanged, so no Qwen3
    or DAC original weights are required.

    Args:
        ckpt_path: Path to the fine-tuned DiT ``.safetensors`` checkpoint
            (custom WanAudioModel naming, e.g. ``epoch-0.safetensors``).
        source_hf_dir: An existing public HF dir (local path) **or** a
            HuggingFace Hub repo id; the latter is auto-downloaded to the
            local cache before copying.
        dst_dir: Output directory.
    """
    ckpt_path = str(ckpt_path)
    source_hf_dir = _resolve_local_dir(str(source_hf_dir))
    dst_dir = str(dst_dir)

    os.makedirs(dst_dir, exist_ok=True)
    print(f"\n[export] writing public HF model to: {dst_dir}")
    print(f"[export] source HF dir:    {source_hf_dir}")
    print(f"[export] checkpoint:       {ckpt_path}")

    transformer_dst = os.path.join(dst_dir, "transformer")
    os.makedirs(transformer_dst, exist_ok=True)
    custom_sd = load_file(ckpt_path)
    diffusers_sd = convert_dit_keys(custom_sd)
    save_file(
        diffusers_sd,
        os.path.join(transformer_dst, "diffusion_pytorch_model.safetensors"),
    )
    print(f"[export] transformer: {len(custom_sd)} → {len(diffusers_sd)} keys")

    src_cfg = os.path.join(source_hf_dir, "transformer", "config.json")
    if os.path.isfile(src_cfg):
        shutil.copy2(src_cfg, os.path.join(transformer_dst, "config.json"))

    for sub in ("vae", "text_encoder", "tokenizer", "scheduler"):
        src = os.path.join(source_hf_dir, sub)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Missing {sub} dir in source HF model: {src}")
        _copy_tree(src, os.path.join(dst_dir, sub))
        print(f"[export] {sub}: copied from source")

    src_index = os.path.join(source_hf_dir, "model_index.json")
    if os.path.isfile(src_index):
        shutil.copy2(src_index, os.path.join(dst_dir, "model_index.json"))
    print(f"[export] done.")
