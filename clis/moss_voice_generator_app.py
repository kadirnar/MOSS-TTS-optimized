import argparse
import functools
import importlib.util
import json
from pathlib import Path
import re
import time

import gradio as gr
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

MODEL_PATH = "OpenMOSS-Team/MOSS-VoiceGenerator"
DEFAULT_ATTN_IMPLEMENTATION = "auto"
DEFAULT_MAX_NEW_TOKENS = 4096
EXAMPLE_TEXTS_JSONL_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "text" / "moss_voice_generator_example_texts.jsonl"
)


def _parse_example_id(example_id: str) -> tuple[str, int] | None:
    matched = re.fullmatch(r"(zh|en)/(\d+)", (example_id or "").strip())
    if matched is None:
        return None
    return matched.group(1), int(matched.group(2))


def build_example_rows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, int, str, str]] = []
    with open(EXAMPLE_TEXTS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            parsed = _parse_example_id(sample.get("id", ""))
            if parsed is None:
                continue

            language, index = parsed
            instruction = str(sample.get("instruction", "")).strip()
            text = str(sample.get("text", "")).strip()
            rows.append((language, index, instruction, text))

    language_order = {"zh": 0, "en": 1}
    rows.sort(key=lambda item: (language_order.get(item[0], 99), item[1]))
    return [(f"{language}/{index}", instruction, text) for language, index, instruction, text in rows]


EXAMPLE_ROWS = build_example_rows()


def apply_example_selection(evt: gr.SelectData):
    if evt is None or evt.index is None:
        return gr.update(), gr.update()

    if isinstance(evt.index, (tuple, list)):
        row_idx = int(evt.index[0])
    else:
        row_idx = int(evt.index)

    if row_idx < 0 or row_idx >= len(EXAMPLE_ROWS):
        return gr.update(), gr.update()

    _, instruction_value, text_value = EXAMPLE_ROWS[row_idx]
    return instruction_value, text_value


def resolve_attn_implementation(requested: str, device: torch.device, dtype: torch.dtype) -> str | None:
    requested_norm = (requested or "").strip().lower()

    if requested_norm in {"none"}:
        return None

    if requested_norm not in {"", "auto"}:
        return requested

    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device.type == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


@functools.lru_cache(maxsize=1)
def load_backend(model_path: str, device_str: str, attn_implementation: str):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    resolved_attn_implementation = resolve_attn_implementation(
        requested=attn_implementation,
        device=device,
        dtype=dtype,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        normalize_inputs=True,
    )
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if resolved_attn_implementation:
        model_kwargs["attn_implementation"] = resolved_attn_implementation

    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    return model, processor, device, sample_rate


def build_conversation(text: str, instruction: str, processor):
    text = (text or "").strip()
    instruction = (instruction or "").strip()
    if not text:
        raise ValueError("Please enter text to synthesize.")
    if not instruction:
        raise ValueError("Please enter a voice instruction.")

    return [[processor.build_user_message(text=text, instruction=instruction)]]


def run_inference(
    text: str,
    instruction: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    model_path: str,
    device: str,
    attn_implementation: str,
):
    started_at = time.monotonic()
    model, processor, torch_device, sample_rate = load_backend(
        model_path=model_path,
        device_str=device,
        attn_implementation=attn_implementation,
    )

    conversations = build_conversation(
        text=text,
        instruction=instruction,
        processor=processor,
    )

    batch = processor(conversations, mode="generation")
    input_ids = batch["input_ids"].to(torch_device)
    attention_mask = batch["attention_mask"].to(torch_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            audio_temperature=float(temperature),
            audio_top_p=float(top_p),
            audio_top_k=int(top_k),
            audio_repetition_penalty=float(repetition_penalty),
        )

    messages = processor.decode(outputs)
    if not messages or messages[0] is None:
        raise RuntimeError("The model did not return a decodable audio result.")

    audio = messages[0].audio_codes_list[0]
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)
    audio_np = audio_np.astype(np.float32, copy=False)

    elapsed = time.monotonic() - started_at
    status = (
        f"Done | elapsed: {elapsed:.2f}s | "
        f"max_new_tokens={int(max_new_tokens)}, "
        f"audio_temperature={float(temperature):.2f}, audio_top_p={float(top_p):.2f}, "
        f"audio_top_k={int(top_k)}, audio_repetition_penalty={float(repetition_penalty):.2f}"
    )
    return (sample_rate, audio_np), status


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
      letter-spacing: 0.2px;
    }
    .app-subtitle {
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 8px;
    }
    #output_audio {
      padding-bottom: 12px;
      margin-bottom: 8px;
      overflow: hidden !important;
    }
    #output_audio > .wrap {
      overflow: hidden !important;
    }
    #output_audio audio {
      margin-bottom: 6px;
    }
    #run-btn {
      background: var(--accent);
      border: none;
    }
    """

    with gr.Blocks(title="MOSS-VoiceGenerator Demo", css=custom_css) as demo:
        gr.Markdown(
            """
            <div class="app-card">
              <div class="app-title">MOSS-VoiceGenerator</div>
              <div class="app-subtitle">Design expressive voices from instruction + text without reference audio.</div>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                instruction = gr.Textbox(
                    label="Voice Instruction",
                    lines=5,
                    placeholder="Example: Warm, gentle female narrator voice with calm pacing and clear articulation.",
                )
                text = gr.Textbox(
                    label="Text",
                    lines=8,
                    placeholder="Enter the text content to synthesize with the instruction-defined voice.",
                )

                with gr.Accordion("Sampling Parameters (Audio)", open=True):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        step=0.05,
                        value=1.5,
                        label="temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.01,
                        value=0.6,
                        label="top_p",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=50,
                        label="top_k",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.8,
                        maximum=2.0,
                        step=0.05,
                        value=1.1,
                        label="repetition_penalty",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=256,
                        maximum=8192,
                        step=128,
                        value=DEFAULT_MAX_NEW_TOKENS,
                        label="max_new_tokens",
                    )

                run_btn = gr.Button("Generate Voice", variant="primary", elem_id="run-btn")

            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Output Audio", type="numpy", elem_id="output_audio")
                status = gr.Textbox(label="Status", lines=4, interactive=False)
                examples_table = gr.Dataframe(
                    headers=["Voice Instruction", "Example Text"],
                    value=[[example_instruction, example_text] for _, example_instruction, example_text in EXAMPLE_ROWS],
                    datatype=["str", "str"],
                    row_count=(len(EXAMPLE_ROWS), "fixed"),
                    col_count=(2, "fixed"),
                    interactive=False,
                    wrap=True,
                    label="Examples (click a row to fill inputs)",
                )

        examples_table.select(
            fn=apply_example_selection,
            inputs=[],
            outputs=[instruction, text],
        )

        run_btn.click(
            fn=lambda text, instruction, temperature, top_p, top_k, repetition_penalty, max_new_tokens: run_inference(
                text=text,
                instruction=instruction,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                model_path=args.model_path,
                device=args.device,
                attn_implementation=args.attn_implementation,
            ),
            inputs=[
                text,
                instruction,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_new_tokens,
            ],
            outputs=[output_audio, status],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="MOSS-VoiceGenerator Gradio Demo")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default=DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    runtime_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime_dtype = torch.bfloat16 if runtime_device.type == "cuda" else torch.float32
    args.attn_implementation = resolve_attn_implementation(
        requested=args.attn_implementation,
        device=runtime_device,
        dtype=runtime_dtype,
    ) or "none"
    print(f"[INFO] Using attn_implementation={args.attn_implementation}", flush=True)

    preload_started_at = time.monotonic()
    print(
        f"[Startup] Preloading backend: model={args.model_path}, device={args.device}, attn={args.attn_implementation}",
        flush=True,
    )
    load_backend(
        model_path=args.model_path,
        device_str=args.device,
        attn_implementation=args.attn_implementation,
    )
    print(
        f"[Startup] Backend preload finished in {time.monotonic() - preload_started_at:.2f}s",
        flush=True,
    )

    demo = build_demo(args)
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
