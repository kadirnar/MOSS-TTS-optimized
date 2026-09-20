"""Supported configurations for the same 8B / 32-codebook model."""

from .paths import ASSETS


def install_gptq(engine, calibration_path):
    from .calibrated_backend import enable_grouped_activation, install_calibrated
    from .dp4a_fusions import enable_fusions
    from .dp4a_gateup import enable_gateup
    from .dp4a_packing import install_packing
    from .attention_quant import enable_attention_quant
    from .attention_native import enable_native_attention
    from .dp4a_gateup_quant import enable_gateup_quant
    from .dp4a_scaled import enable_scaled
    from .dp4a_norm_projection import enable_norm_projection
    from . import short_scales, projection_pdl, attention_pdl, bulk_prefetch
    from . import prefill_qkv, prefill_pointwise, async_output, qkv_cluster_model
    from . import down_tile, gateup_compiler, attention_history_model

    fast = engine.llm
    install_calibrated(fast, calibration_path, backend="dp4a")
    enable_grouped_activation(fast)
    enable_fusions(fast, layout_limit=8)
    enable_gateup(fast)
    install_packing(fast, ASSETS / "packing.json")
    enable_attention_quant(fast)
    enable_native_attention(fast)
    enable_gateup_quant(fast)
    enable_scaled(fast)
    enable_norm_projection(fast)
    short_scales.enable(fast)
    projection_pdl.enable(fast)
    attention_pdl.enable(fast)
    bulk_prefetch.enable(fast)
    prefill_qkv.enable(fast)
    prefill_pointwise.enable(fast)
    async_output.enable(fast, register_preload=True)
    qkv_cluster_model.enable(fast)
    down_tile.enable(fast)
    gateup_compiler.enable(fast)
    attention_history_model.enable(fast)


def warmup(engine, preset):
    engine.warmup((128, 160, 256, 512) if preset == "gptq" else (128, 256, 512))
    if preset == "gptq":
        from .decode_buckets import DecodeContextBuckets
        from .audio_head_buckets import DecodeAudioHeadBuckets
        from .first_audio_graph import enable

        manager = DecodeContextBuckets(engine.llm)
        manager.warmup()
        manager.install()
        heads = DecodeAudioHeadBuckets(engine.llm, manager)
        heads.warmup()
        heads.install()
        enable(engine)
