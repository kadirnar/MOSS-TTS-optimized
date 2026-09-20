"""Create an explicit G32 export from user-supplied voice-cloning examples."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .models import CODEC_REVISION, TTS_REVISION


def read_dataset(path: str | Path) -> list[dict]:
    """Read JSONL with text, reference, language and train/validation split."""
    path = Path(path)
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("text", "reference", "language")
        ):
            raise ValueError(f"Line {line_number}: nonempty text, reference and language required")
        split = row.get("split", "train")
        if split not in ("train", "validation"):
            raise ValueError(f"Line {line_number}: split must be train or validation")
        reference = path.parent / row["reference"]
        if not reference.is_file():
            raise FileNotFoundError(reference)
        records.append({**row, "reference": str(reference.resolve()), "split": split})
    if {row["split"] for row in records} != {"train", "validation"}:
        raise ValueError("Provide at least one training and one held-out validation example")
    return records


def export(dataset: str | Path, output: str | Path, *, local_files_only: bool = False) -> None:
    """Export 144 projections; original checkpoints are never overwritten.

    Calibration is offline GPU work. New datasets produce new quantized weights
    and require their own quality evaluation; historical latency/quality records
    do not qualify a new export automatically.
    """
    records = read_dataset(dataset)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new export directory; existing exports are preserved")
    import torch

    from ._runtime.calibrated_quant import dequantize, gptq
    from .api import MossTTS

    output.mkdir(parents=True)
    torch.set_float32_matmul_precision("highest")
    with MossTTS.from_pretrained(local_files_only=local_files_only) as tts, torch.inference_mode():
        engine = tts._engine
        fast = engine.llm
        original_step, original_full = fast.step, fast.full_step
        observed = {}

        def step(ids, position, *args):
            observed[position] = ids.detach().cpu().clone()
            return original_step(ids, position, *args)

        def full(ids, position):
            observed[position] = ids.detach().cpu().clone()
            return original_full(ids, position)

        fast.step, fast.full_step = step, full
        sequences = []
        try:
            for index, row in enumerate(records):
                voice = tts.clone_voice(row["reference"])
                observed.clear()
                processor = engine.processor
                ids = processor(
                    [
                        [
                            processor.build_user_message(
                                text=row["text"], reference=[voice.codes], language=row["language"]
                            )
                        ]
                    ],
                    mode="generation",
                ).input_ids
                frames = sum(
                    1
                    for _ in tts.stream(
                        row["text"], voice=voice, language=row["language"], seed=8800 + index
                    )
                )
                if not frames or tts.metrics["truncated"]:
                    raise ValueError("Calibration example did not finish; use shorter text")
                prompt = ids.shape[1]
                positions = sorted(observed)
                if positions != list(range(prompt, prompt + len(positions))):
                    raise RuntimeError("Calibration decoder positions are not contiguous")
                sequences.append(
                    (
                        torch.cat(
                            [ids.cpu()] + [observed[position] for position in positions], dim=1
                        ),
                        prompt,
                        row["split"],
                    )
                )
                print(f"Collected {index + 1}/{len(records)} calibration utterances", flush=True)
        finally:
            fast.step, fast.full_step = original_step, original_full

        activations = defaultdict(lambda: defaultdict(list))
        prompt_boundary = 0
        current_split = "train"

        def capture(name, tensor):
            value = tensor[:, prompt_boundary:].reshape(-1, tensor.shape[-1]).detach().cpu()
            if not value.numel() or not torch.isfinite(value).all():
                raise ValueError("Invalid calibration activation")
            activations[name][current_split].append(value.contiguous())

        hooks = []
        try:
            for index, layer in enumerate(fast.model.language_model.layers):
                attention, mlp = layer.self_attn, layer.mlp
                mlp._triton_gemv = False
                hooks.extend(
                    [
                        attention.register_forward_pre_hook(
                            lambda mod, args, kwargs, i=index: capture(
                                f"{i:02d}_qkv", kwargs["hidden_states"]
                            ),
                            with_kwargs=True,
                        ),
                        attention.o_proj.register_forward_pre_hook(
                            lambda mod, args, i=index: capture(f"{i:02d}_out", args[0])
                        ),
                        mlp.register_forward_pre_hook(
                            lambda mod, args, i=index: capture(f"{i:02d}_up", args[0])
                        ),
                        mlp.down_proj.register_forward_pre_hook(
                            lambda mod, args, i=index: capture(f"{i:02d}_down", args[0])
                        ),
                    ]
                )
            for sequence, prompt_boundary, current_split in sequences:
                position = torch.arange(sequence.shape[1], device="cuda")
                mask = (fast.kv_index[None, :] <= position[:, None]).view(
                    1, 1, sequence.shape[1], -1
                )
                fast.hidden(sequence.cuda(), position, mask)
        finally:
            for hook in hooks:
                hook.remove()
        if len(activations) != 144:
            raise RuntimeError("Expected all 144 8B projection activations")

        config = {
            "complete": False,
            "group": 32,
            "damping": 0.1,
            "actorder": True,
            "static_bf16_scales": True,
            "code_range": [-7, 7],
            "codebooks": 32,
            "tts_revision": TTS_REVISION,
            "codec_revision": CODEC_REVISION,
            "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "projections": {},
        }
        config_path = output / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        for index, layer in enumerate(fast.model.language_model.layers):
            weights = {
                "qkv": layer.self_attn._qkv,
                "out": layer.self_attn.o_proj.weight,
                "up": layer.mlp._gate_up,
                "down": layer.mlp.down_proj.weight,
            }
            for projection, weight in weights.items():
                name = f"{index:02d}_{projection}"
                samples = activations.pop(name)
                train = torch.cat(samples["train"]).cuda()
                valid = torch.cat(samples["validation"]).cuda()
                signed, scales = gptq(weight, train, group=32, damping=0.1)
                reference = torch.nn.functional.linear(valid.float(), weight.float())
                predicted = torch.nn.functional.linear(
                    valid.float(), dequantize(signed, scales, 32).float()
                )
                mse = ((predicted - reference).square().mean() / reference.square().mean()).item()
                unsigned = signed.to(torch.uint8) & 15
                packed = (unsigned[:, ::2] | (unsigned[:, 1::2] << 4)).contiguous()
                destination = output / f"{name}.pt"
                temporary = destination.with_suffix(".tmp")
                torch.save(
                    {
                        "packed": packed.cpu(),
                        "scales": scales.cpu(),
                        "shape": list(weight.shape),
                        "group": 32,
                    },
                    temporary,
                )
                temporary.replace(destination)
                config["projections"][name] = {
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "training_rows": len(train),
                    "heldout_rows": len(valid),
                    "heldout_mse": mse,
                }
                print(f"Exported {name}; held-out projection MSE={mse:.6f}", flush=True)
        config["complete"] = True
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Export complete: {output}", flush=True)
