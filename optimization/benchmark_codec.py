import gc
import torch
import soundfile as sf
from .common import RESULTS, load_models, timed, stats, save_json
from .codec import StreamingCodec, cache_codebooks


@torch.inference_mode()
def main():
    model, codec, processor = load_models()
    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    codes = torch.load(RESULTS / "fixture.pt", weights_only=True)["codes"].cuda()
    codes = codes[..., :20]
    quant_ref = codec.quantizer.decode_codes(codes).clone()
    measurements = {}
    reference = None
    for name, graph, cached, bf16, triton in [
        ("upstream_stream",False,False,False,False),
        ("cuda_graph_fp32",True,False,False,False),
        ("cached_codebooks_graph_fp32",True,True,False,False),
        ("triton_codec_graph_fp32",True,True,False,True),
        ("triton_codec_graph_bf16",True,True,True,True),
    ]:
        decoder = StreamingCodec(codec, graph=graph, bf16=bf16, cached_codebooks=cached, triton_attention=triton)
        decoder.warmup()
        runs, first_times, waves = [], [], []
        for r in range(3):
            decoder.reset()
            chunks=[]
            for i in range(codes.shape[-1]):
                out, ms = timed(lambda: decoder.decode(codes[..., i:i+1]))
                runs.append(ms)
                if i == 0: first_times.append(ms)
                chunks.append(out.float())
            waves.append(torch.cat(chunks,-1))
        wave = waves[0]
        if reference is None: reference = wave.clone()
        error = (wave - reference).float()
        measurements[name] = {"decode": stats(runs), "first_frame":stats(first_times),
            "max_abs_wave_error":error.abs().max().item(),
            "wave_snr_db":(10*torch.log10(reference.square().sum()/error.square().sum().clamp_min(1e-30))).item(),
            "reset_max_abs_error": (waves[0]-waves[1]).abs().max().item()}
        if cached:
            measurements[name]["quantizer_max_abs_error"] = (codec.quantizer.decode_codes(codes)-quant_ref).abs().max().item()
        sf.write(RESULTS / (name+".wav"), wave.flatten().cpu().numpy(),24000)
        print(name, measurements[name]["decode"]["median_ms"], "SNR", measurements[name]["wave_snr_db"],flush=True)
        save_json("codec_iterations.json", measurements)
        decoder.close()
        del decoder
        gc.collect(); torch.cuda.empty_cache()
    save_json("codec_iterations.json",measurements)

if __name__ == "__main__": main()
