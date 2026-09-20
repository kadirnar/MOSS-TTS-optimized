"""Upstream baseline. Run with python -m optimization.baseline."""
from contextlib import ExitStack
import time
import torch
import soundfile as sf
import torchaudio

from .common import ROOT, RESULTS, load_models, timed, stats, save_json
from moss_audio_tokenizer.modeling_moss_audio_tokenizer import StreamingModule


@torch.inference_mode()
def main():
    model, codec, processor = load_models()
    wave, sr = sf.read(ROOT / "assets/audio/reference_zh.wav", dtype="float32")
    wave = torch.from_numpy(wave).reshape(1, -1)
    text = "你好，这是一段用于测试流式语音合成速度的句子。"
    _, cold_encode = timed(lambda: processor.encode_audios_from_wav([wave], sr))
    encoded, encode_ms = timed(lambda: processor.encode_audios_from_wav([wave], sr))
    reference = encoded[0]
    inputs, prep_ms = timed(lambda: processor([[processor.build_user_message(text=text, reference=[reference], language="Chinese")]], mode="generation").to("cuda"))
    print("INPUT", inputs.input_ids.shape, processor.tokenizer.decode(inputs.input_ids[0, -12:, 0]), flush=True)
    # Warm prefill and decode kernels before collecting the baseline.
    warm = model(**inputs, use_cache=True)
    model(input_ids=inputs.input_ids[:, -1:], past_key_values=warm.past_key_values, use_cache=True)
    torch.cuda.synchronize()
    calls = []
    def before(*_):
        torch.cuda.synchronize()
        calls.append([time.perf_counter(), None])
    def after(*_):
        torch.cuda.synchronize()
        calls[-1][1] = (time.perf_counter() - calls[-1][0]) * 1000
    h1 = model.register_forward_pre_hook(before)
    h2 = model.register_forward_hook(after)
    torch.manual_seed(1234)
    output, generate_ms = timed(lambda: model.generate(**inputs, max_new_tokens=100))
    h1.remove(); h2.remove()
    start_length, ids = output[0]
    codes = processor.apply_de_delay_pattern(ids[:, 1:])
    valid = (codes < model.config.audio_pad_code).all(-1)
    codes = codes[valid].T[:, None].contiguous()
    print("CODES", codes.shape, flush=True)
    torch.save({"inputs": {k: v.cpu() for k,v in inputs.items()}, "reference": reference.cpu(), "codes": codes.cpu(), "upstream_ids": ids.cpu(), "text": text}, RESULTS / "fixture.pt")
    first = codes[..., :1]
    codec.decode(first)
    codec_ms = []
    for _ in range(5):
        _, elapsed = timed(lambda: codec.decode(first))
        codec_ms.append(elapsed)
    streaming_ms = []
    chunks = []
    with ExitStack() as stack:
        for module in codec.decoder:
            if isinstance(module, StreamingModule):
                stack.enter_context(module.streaming(1))
        for i in range(min(codes.shape[-1], 20)):
            chunk, elapsed = timed(lambda: codec.decode(codes[..., i:i+1]))
            chunks.append(chunk.audio)
            streaming_ms.append(elapsed)
    sf.write(RESULTS / "baseline_stream.wav", torch.cat(chunks,-1).flatten().cpu().numpy(), 24000)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        model(input_ids=inputs.input_ids[:, -1:], past_key_values=warm.past_key_values, use_cache=True)
        codec.decode(first)
    prof.export_chrome_trace(str(RESULTS / "baseline_trace.json"))
    (RESULTS / "baseline_profile.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=35))
    save_json("baseline.json", {
        "torch": torch.__version__, "gpu":torch.cuda.get_device_name(),
        "llm_parameters":sum(p.numel() for p in model.parameters()),
        "codec_parameters":sum(p.numel() for p in codec.parameters()),
        "prompt_tokens":inputs.input_ids.shape[1], "reference_seconds": wave.numel()/sr,
        "reference_encode_first_ms": cold_encode, "reference_encode_warm_ms": encode_ms,
        "cached_reference_prompt_ms": prep_ms, "prefill_ms": calls[0][1],
        "llm_decode":stats([x[1] for x in calls[1:]]),
        "generation_100_steps_ms":generate_ms,
        "codec_first_frame":stats(codec_ms), "codec_stream_frames":stats(streaming_ms),
        "generated_codec_frames":codes.shape[-1],
        "note":"Upstream generate is not incremental; streaming codec timings are measured separately. No end-to-end TTFA claim from this script."
    })


if __name__ == "__main__":
    main()
