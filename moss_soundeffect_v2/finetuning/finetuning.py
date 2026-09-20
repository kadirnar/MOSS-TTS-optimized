"""Fine-tune the DiT from a public MossSoundEffect HF directory."""

import argparse
import faulthandler
import json
import os
import signal
import sys
import time
from pathlib import Path

faulthandler.enable(all_threads=True)
faulthandler.register(signal.SIGUSR2, all_threads=True)

import torch

_HERE = Path(__file__).resolve().parent
_PKG_DIR = _HERE.parent
_PROJECT_DIR = _PKG_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from moss_soundeffect_v2.diffsynth.trainers.utils import (
    AudioDataset,
    DiffusionTrainingModule,
    ModelLogger,
    launch_data_process_task,
    launch_training_task,
)
from moss_soundeffect_v2.hf_export import export_finetuned_to_hf
from moss_soundeffect_v2.pipeline_moss_soundeffect import (
    MossSoundEffectPipeline,
    _DIT_PRESETS,
)


class MossSoundEffectFinetuneModule(DiffusionTrainingModule):
    model_input_keys = ["input_latents", "context"]

    def __init__(
        self,
        hf_model_dir: str,
        trainable_models: str = "dit",
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        extra_inputs=None,
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,
        drop_prompt_prob: float = 0.1,
    ):
        super().__init__()
        self.hf_model_dir = hf_model_dir

        # Load on CPU; ``accelerator.prepare`` will move it.
        moss_pipe = MossSoundEffectPipeline.from_pretrained(
            hf_model_dir, torch_dtype=torch.bfloat16, device="cpu"
        )
        with open(os.path.join(hf_model_dir, "model_index.json")) as f:
            index = json.load(f)
        self.dit_variant = index.get("dit_variant")
        if self.dit_variant not in _DIT_PRESETS:
            raise ValueError(
                f"dit_variant {self.dit_variant!r} in model_index.json is not "
                f"a supported preset {sorted(_DIT_PRESETS.keys())}."
            )

        # ``self.pipe`` is the underlying WanAudioPipeline; DiT params live at
        # ``pipe.dit.*``, hence ``remove_prefix_in_ckpt='pipe.dit.'`` in
        # ModelLogger writes bare keys expected by WanAudioModel.load_state_dict.
        self.pipe = moss_pipe.engine
        # VAE encode runs every step; compile for throughput.
        self.pipe.vae.encode = torch.compile(
            self.pipe.vae.encode, fullgraph=True, mode="max-autotune"
        )

        self.pipe.scheduler.set_timesteps(1000, training=True)
        names = [] if trainable_models is None else trainable_models.split(",")
        self.pipe.freeze_except(names)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.drop_prompt_prob = drop_prompt_prob
        self.empty_embeddings = None

    @torch.no_grad()
    def forward_preprocess(self, data):
        # Cached branch: latents and text context precomputed by launch_data_process_task.
        if isinstance(data, dict) and "cached" in data:
            assert data["cached"].all(), "cached must be a Tensor bool, and all True"
            assert "input_latents" in data and "context" in data

            if self.empty_embeddings is None:
                self.empty_embeddings = self.pipe.prompter.encode_prompt(
                    "", positive=True, device=self.pipe.device
                )

            inputs = {
                "input_latents": data["input_latents"].squeeze(1).to(
                    dtype=self.pipe.torch_dtype, device=self.pipe.device
                ),
                "context": data["context"].squeeze(1).to(
                    dtype=self.pipe.torch_dtype, device=self.pipe.device
                ),
            }
            drop_mask = torch.bernoulli(
                torch.full(
                    (data["input_latents"].size(0),),
                    self.drop_prompt_prob,
                    device=self.pipe.device,
                )
            ).view(-1, 1, 1).bool()
            inputs["context"] = torch.where(drop_mask, self.empty_embeddings, inputs["context"])
            inputs["noise"] = self.pipe.generate_noise(
                inputs["input_latents"].shape, rand_device=self.pipe.device
            )
            inputs.update({
                "cfg_scale": 1,
                "cfg_merge": False,
                "vace_scale": 1,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
            })
            return inputs

        # Non-cached branch: encode audio + prompt on the fly.
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_audio": data["audio"],
            "audio_latent": data.get("audio_latent", None),
            "num_channels": data["audio"].size(-2),
            "num_samples": data["audio"].size(-1),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        for extra_input in self.extra_inputs:
            inputs_shared[extra_input] = data[extra_input]

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        return self.pipe.training_loss(**models, **inputs)


class MyModelLogger(ModelLogger):
    def on_step_end(self, accelerator, model, save_steps=None, loss=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


def _find_latest_checkpoint(output_path: str):
    candidates = []
    for name in os.listdir(output_path):
        if not name.endswith(".safetensors"):
            continue
        full = os.path.join(output_path, name)
        if not os.path.isfile(full):
            continue
        candidates.append((os.path.getmtime(full), full))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune MossSoundEffect from a public HF directory.")
    parser.add_argument("--hf_model_dir", type=str, required=True)
    # Dataset
    parser.add_argument("--dataset_base_path", type=str, default="", required=True)
    parser.add_argument("--dataset_metadata_path", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--num_audio_samples", type=int, default=1440000,
                        help="Fixed audio length in samples. Default 30s @ 48kHz.")
    parser.add_argument("--min_num_audio_samples", type=int, default=960)
    parser.add_argument("--max_num_audio_samples", type=int, default=1440000)
    parser.add_argument("--mono", default=False, action="store_true")
    parser.add_argument("--data_file_keys", type=str, default="audio")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--dataset_num_workers", type=int, default=4)
    parser.add_argument("--drop_prompt_prob", type=float, default=0.1)
    parser.add_argument("--append_duration_suffix", default=False, action="store_true")
    parser.add_argument("--append_duration_suffix_prob", type=float, default=0.5)
    parser.add_argument("--duration_precision", type=int, default=1)
    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--clip_grad_norm", type=float, default=0.1)
    parser.add_argument("--trainable_models", type=str, default="dit")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--extra_inputs", default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    # Cache
    parser.add_argument("--cache_folder", type=str, default=None)
    parser.add_argument("--cache_first", default=False, action="store_true")
    parser.add_argument("--cache_num_shards", type=int, default=64)
    parser.add_argument("--skip_first_batches", type=int, default=0)
    # HF export
    parser.add_argument("--no_export_hf", default=False, action="store_true")
    parser.add_argument("--export_hf_dir", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()

    if args.cache_first and not args.cache_folder:
        raise ValueError("--cache_first requires --cache_folder.")

    dataset = AudioDataset(args=args, metadata_path=args.dataset_metadata_path)
    if args.cache_folder is not None:
        dataset.cache_folder = args.cache_folder

    model = MossSoundEffectFinetuneModule(
        hf_model_dir=args.hf_model_dir,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        drop_prompt_prob=args.drop_prompt_prob,
    )

    model_logger = MyModelLogger(
        args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    if args.cache_first:
        os.makedirs(args.cache_folder, exist_ok=True)
        from moss_soundeffect_v2.diffsynth.trainers.cache_shards import (
            DEFAULT_META_FILENAME,
        )
        meta_path = os.path.join(args.cache_folder, DEFAULT_META_FILENAME)
        if os.path.exists(meta_path):
            print(
                f"[cache] {meta_path} already exists; assuming cache is populated "
                f"and skipping cache generation. Delete the cache folder to force "
                f"a rebuild."
            )
        else:
            cache_dataset = AudioDataset(args=args, metadata_path=args.dataset_metadata_path)
            cache_dataset.cache_folder = None
            cache_dataset.drop_prompt_prob = 0.0
            launch_data_process_task(
                model, cache_dataset, args.cache_folder,
                num_shards=args.cache_num_shards,
                skip_first_batches=args.skip_first_batches,
                num_workers=args.dataset_num_workers,
            )

    if args.num_epochs == 0:
        print("num_epochs=0, skipping training.")
        return

    log_dir = None
    if args.log_dir is not None:
        log_dir = args.log_dir + f"/{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler=None,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        find_unused_parameters=args.find_unused_parameters,
        num_workers=args.dataset_num_workers,
        prefetch_factor=16,
        batch_size=args.batch_size,
        clip_grad_norm=args.clip_grad_norm,
        log_dir=log_dir,
        resume_from=args.resume_from,
    )

    if args.no_export_hf:
        return

    from accelerate import PartialState
    if not PartialState().is_main_process:
        return

    ckpt = _find_latest_checkpoint(args.output_path)
    if ckpt is None:
        print(f"[export] no .safetensors found in {args.output_path}; skipping HF export.")
        return

    export_dir = args.export_hf_dir or os.path.join(args.output_path, "hf_format")
    export_finetuned_to_hf(
        ckpt_path=ckpt,
        source_hf_dir=args.hf_model_dir,
        dst_dir=export_dir,
    )


if __name__ == "__main__":
    main()
