"""All-32-codebook weight-only experiments, measured against upstream logits."""
import argparse
import gc
import json

import soundfile as sf
import torch

from .common import RESULTS, load_models, stats, timed
from .streaming import StreamingTTS


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['none', 'fp8_mlp', 'fp8_all', 'int8_all', 'int4_all','marlin4_all','marlin4g32_all','marlin8_all','int4dp4a_all','int4dp4ag32_all'], required=True)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--max-new-tokens', type=int, default=400)
    parser.add_argument('--tiled', action='store_true')
    parser.add_argument('--attention-backend', choices=['triton','flashinfer','flashinfer_tc'], default='triton')
    parser.add_argument('--tag', default='', help='Suffix to retain results from different runtimes')
    parser.add_argument('--attention-block',type=int,default=128)
    parser.add_argument('--attention-warps',type=int,default=4)
    parser.add_argument('--fused-residual',action='store_true')
    parser.add_argument('--fused-gateup',action='store_true')
    parser.add_argument('--fp8-prefill',action='store_true')
    parser.add_argument('--calibration',help='Explicit folder of calibrated INT4 projections')
    parser.add_argument('--calibrated-backend',choices=('marlin','dp4a'),default='marlin')
    parser.add_argument('--dp4a-reciprocal',action='store_true')
    parser.add_argument('--fused-dp4a',action='store_true')
    parser.add_argument('--grouped-activation',action='store_true')
    parser.add_argument('--prefill-buckets',type=int,nargs='+',default=[128,256,512])
    parser.add_argument('--dp4a-gateup',action='store_true')
    parser.add_argument('--norm-layout-limit',type=int,choices=(0,8),default=0)
    parser.add_argument('--packing-plan')
    parser.add_argument('--g128-plan')
    parser.add_argument('--g128-residual-calibration')
    parser.add_argument('--g128-residual-plan')
    parser.add_argument('--decode-buckets',action='store_true')
    parser.add_argument('--attention-quant',action='store_true')
    parser.add_argument('--native-attention',action='store_true')
    parser.add_argument('--gateup-quant',action='store_true')
    parser.add_argument('--scaled-dp4a',action='store_true')
    parser.add_argument('--norm-projection',action='store_true')
    parser.add_argument('--qkv-load-policy',action='store_true')
    parser.add_argument('--compressed-scales',action='store_true',help='Lossless SM90 compression of selected FP32 scale allocations')
    parser.add_argument('--short-scales',action='store_true',help='Exact BF16 scale storage with explicit Gluon reduction layouts')
    parser.add_argument('--projection-pdl',action='store_true',help='SM90 projection dependency overlap with exact short scales')
    parser.add_argument('--attention-pdl',action='store_true',help='Extend SM90 dependency overlap through native attention and quantization')
    parser.add_argument('--codec-clock',action='store_true',help='Exact FP32 codec with shared stage counters and first-frame specialization')
    parser.add_argument('--first-audio-graph',action='store_true',help='Run the initial 32 audio steps in one CUDA graph')
    parser.add_argument('--bulk-prefetch',action='store_true',help='Prefix-only asynchronous bulk L2 hints for selected projections')
    parser.add_argument('--prefill-qkv',action='store_true',help='Fuse BF16 prefill Q/K normalization, rotary and KV writes')
    parser.add_argument('--prefill-pointwise',action='store_true',help='Fuse BF16 prefill activation/product and residual/normalization')
    parser.add_argument('--async-output',action='store_true',help='Stage exact G32 output weights with SM90 asynchronous copies')
    parser.add_argument('--output-weight-prefetch',action='store_true',help='Preload exact G32 attention-output weights into registers before PDL wait')
    parser.add_argument('--qkv-cluster',action='store_true',help='Fuse G32 QKV/head preparation using SM90 clustered cubins')
    parser.add_argument('--down-tile8',action='store_true',help='Use eight-row/four-warp G32 down projections with clustered QKV')
    parser.add_argument('--gateup-compiler',action='store_true',help='Use exact one-CTA gate/up cubins from isolated Triton 3.8')
    args = parser.parse_args()
    if args.async_output and args.output_weight_prefetch:raise ValueError('Choose one output staging strategy')
    if args.gateup_compiler and not args.down_tile8:raise ValueError('Gate/up compiler option requires the eight-row down preset')
    if args.down_tile8 and not args.qkv_cluster:raise ValueError('Eight-row down tile requires clustered QKV')
    if args.qkv_cluster and not (args.bulk_prefetch and args.output_weight_prefetch):raise ValueError('Clustered QKV requires the selected bulk and register-preload preset')
    if (args.async_output or args.output_weight_prefetch) and not (args.short_scales and args.projection_pdl and args.attention_pdl):raise ValueError('Async output requires G32 short scales and attention/projection PDL')
    if args.projection_pdl and (not args.short_scales or args.qkv_load_policy):raise ValueError('Projection PDL requires exact short scales without the experimental QKV load policy')
    if args.attention_pdl and not (args.projection_pdl and args.native_attention and args.attention_quant):raise ValueError('Attention PDL requires projection PDL and native quantized attention')
    if args.qkv_load_policy and not args.norm_projection:raise ValueError('QKV load policy requires G32 norm-projection fusion')
    if args.compressed_scales and (not args.norm_projection or args.g128_plan or args.g128_residual_calibration):raise ValueError('Compressed scales require the selected G32 norm-projection preset')
    if args.short_scales and (not args.norm_projection or args.compressed_scales or args.g128_plan or args.g128_residual_calibration):raise ValueError('Short scales require G32 norm-projection without compressed FP32 scales or G128')
    if bool(args.g128_residual_calibration)!=bool(args.g128_residual_plan) or (args.g128_residual_calibration and (args.g128_plan or not args.norm_projection)):
        raise ValueError('Residual G128 requires its export and plan, plus the G32 norm-projection preset')
    if args.g128_plan and (not args.calibration or args.packing_plan or args.scaled_dp4a or args.norm_projection or args.gateup_quant):raise ValueError('G128 kernels require a calibrated export and their separate plan')
    if args.norm_layout_limit and not args.fused_dp4a:raise ValueError('Norm layout requires fused DP4A')
    if args.calibration and (args.mode!='none' or args.fp8_prefill):
        raise ValueError('Calibrated weights require --mode none and BF16 prefill')
    model, codec, processor = load_models()
    fixture = torch.load(RESULTS / 'fixture.pt', weights_only=True)
    ids = fixture['inputs']['input_ids'].cuda()
    tokens = fixture['upstream_ids'].cuda()
    out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    reference = []
    for i in range(36):
        out = model(input_ids=tokens[i:i+1][None], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        reference.append(torch.cat([a[:, -1, :1024] for a in out.logits[1:]], 0).clone())
    del out, cache
    gc.collect()

    engine = StreamingTTS(model, codec, processor, weight_quantization=args.mode, tiled_quantization=args.tiled, attention_backend=args.attention_backend,attention_block=args.attention_block,attention_warps=args.attention_warps,fused_residual=args.fused_residual,fused_gateup=args.fused_gateup,fp8_prefill=args.fp8_prefill,codec_clock=args.codec_clock)
    calibration_config=None
    if args.calibration:
        from .calibrated_backend import install_calibrated
        calibration_config=install_calibrated(engine.llm,args.calibration,args.calibrated_backend)
    if args.dp4a_reciprocal:
        from .calibrated_backend import enable_dp4a_reciprocal
        enable_dp4a_reciprocal(engine.llm)
    if args.fused_dp4a:
        if not args.dp4a_reciprocal:raise ValueError('Fused DP4A requires explicit reciprocal quantization')
        from .dp4a_fusions import enable_fusions
        enable_fusions(engine.llm,layout_limit=args.norm_layout_limit)
    if args.grouped_activation:
        if not args.dp4a_reciprocal:raise ValueError('Grouped input requires explicit reciprocal quantization')
        from .calibrated_backend import enable_grouped_activation
        enable_grouped_activation(engine.llm)
    if args.dp4a_gateup:
        from .dp4a_gateup import enable_gateup
        enable_gateup(engine.llm)
    packing_config=None
    if args.packing_plan:
        from .dp4a_packing import install_packing
        packing_config=install_packing(engine.llm,args.packing_plan)
    group128_config=None
    if args.g128_plan:
        from .dp4a_group128 import install as install_group128
        group128_config=install_group128(engine.llm,args.g128_plan)
    if args.attention_quant:
        from .attention_quant import enable_attention_quant
        enable_attention_quant(engine.llm)
    if args.native_attention:
        from .attention_native import enable_native_attention
        enable_native_attention(engine.llm)
    if args.gateup_quant:
        from .dp4a_gateup_quant import enable_gateup_quant
        enable_gateup_quant(engine.llm)
    if args.scaled_dp4a:
        from .dp4a_scaled import enable_scaled
        enable_scaled(engine.llm)
    if args.norm_projection:
        from .dp4a_norm_projection import enable_norm_projection
        enable_norm_projection(engine.llm)
    if args.qkv_load_policy:
        from .dp4a_norm_memory import enable_qkv_load_policy
        enable_qkv_load_policy(engine.llm)
    compressed_scales_config=None
    if args.compressed_scales:
        from .compressed_alloc import enable_compressed_scales
        compressed_scales_config=enable_compressed_scales(engine.llm)
    short_scales_config=None
    if args.short_scales:
        from .short_scales import enable
        short_scales_config=enable(engine.llm)
    projection_pdl_config=None
    if args.projection_pdl:
        from .projection_pdl import enable
        projection_pdl_config=enable(engine.llm)
    attention_pdl_config=None
    if args.attention_pdl:
        from .attention_pdl import enable
        attention_pdl_config=enable(engine.llm)
    group128_residual_config=None
    if args.g128_residual_calibration:
        from .dp4a_group128_residual import install as install_residual_group128
        group128_residual_config=install_residual_group128(engine.llm,args.g128_residual_calibration,args.g128_residual_plan)
    bulk_prefetch_config=None
    if args.bulk_prefetch:
        from .bulk_prefetch import enable
        bulk_prefetch_config=enable(engine.llm)
    prefill_qkv_config=None
    if args.prefill_qkv:
        from .prefill_qkv import enable
        prefill_qkv_config=enable(engine.llm)
    prefill_pointwise_config=None
    if args.prefill_pointwise:
        from .prefill_pointwise import enable
        prefill_pointwise_config=enable(engine.llm)
    async_output_config=None
    if args.async_output:
        from .async_output import enable
        async_output_config=enable(engine.llm)
    output_weight_prefetch_config=None
    if args.output_weight_prefetch:
        from .async_output import enable
        output_weight_prefetch_config=enable(engine.llm,register_preload=True)
    qkv_cluster_config=None
    if args.qkv_cluster:
        from .qkv_cluster_model import enable
        qkv_cluster_config=enable(engine.llm)
    down_tile8_config=None
    if args.down_tile8:
        from .down_tile import enable
        down_tile8_config=enable(engine.llm)
    gateup_compiler_config=None
    if args.gateup_compiler:
        from .gateup_compiler import enable
        gateup_compiler_config=enable(engine.llm)
    engine.warmup(tuple(args.prefill_buckets))
    decode_buckets=None
    if args.decode_buckets:
        from .decode_buckets import DecodeContextBuckets
        decode_buckets=DecodeContextBuckets(engine.llm)
        decode_buckets.warmup();decode_buckets.install()
    first_audio_graph_config=None
    if args.first_audio_graph:
        from .first_audio_graph import enable
        first_audio_graph_config=enable(engine)
    fast = engine.llm
    fast.prefill(ids)
    errors, top1, kl = [], [], []
    for i, ref in enumerate(reference):
        fast.step(tokens[i:i+1][None], ids.shape[1]+i, i+1, -1)
        value = fast.audio_logits
        active = min(i+1, 32)
        errors.append(((value.float()-ref.float()).square().mean().sqrt()/ref.float().square().mean().sqrt()).item())
        top1.append((value[:active].argmax(-1) == ref[:active].argmax(-1)).float().mean().item())
        p = (ref[:active].float()/1.7).softmax(-1)
        kl.append((p*((ref[:active].float()/1.7).log_softmax(-1)-(value[:active].float()/1.7).log_softmax(-1))).sum(-1).mean().item())
        assert torch.isfinite(value).all(), 'Non-finite teacher-forced logits'
    validation = {
        'reference': 'Original upstream BF16 model; 36 teacher-forced steps on one Chinese cloned-voice prompt',
        'max_relative_rms_logit_error': max(errors),
        'active_codebook_mean_top1_agreement': sum(top1)/len(top1),
        'active_codebook_mean_kl_divergence': sum(kl)/len(kl),
        'per_step_relative_rms': errors,
        'quality_gate': 'Diagnostic only; no corpus-level quality or voice-identity acceptance implied',
    }
    decode = []
    for i in range(23):
        _, ms = timed(lambda: fast.step(tokens[10:11][None], ids.shape[1]+10, 11, -1))
        if i >= 3:
            decode.append(ms)
    mode='gptq_'+args.calibrated_backend if args.calibration else args.mode
    name = 'all32_' + mode + ('_tiled' if args.tiled else '')
    if args.attention_backend != 'triton':
        name += '_' + args.attention_backend
    if args.tag:
        if not all(c.isalnum() or c=='_' for c in args.tag):
            raise ValueError('tag must contain only letters, numbers and underscores')
        name += '_' + args.tag
    if (args.attention_block,args.attention_warps)!=(128,4):
        name += f'_b{args.attention_block}w{args.attention_warps}'
    if args.fused_residual:name+='_fr'
    if args.fused_gateup:name+='_fg'
    if args.fp8_prefill:name+='_pf8'
    if args.dp4a_reciprocal:name+='_recip'
    if args.fused_dp4a:name+='_qf'
    if args.grouped_activation:name+='_ag'
    if args.dp4a_gateup:name+='_gu'
    if args.norm_layout_limit:name+=f'_nl{args.norm_layout_limit}'
    if args.packing_plan:name+='_packed'
    if args.decode_buckets:name+='_db'
    if args.g128_plan:name+='_g128p'
    if args.g128_residual_calibration:name+='_g128r'
    if args.attention_quant:name+='_aq'
    if args.native_attention:name+='_na'
    if args.gateup_quant:name+='_guq'
    if args.scaled_dp4a:name+='_scaled'
    if args.norm_projection:name+='_np'
    if args.qkv_load_policy:name+='_qkvload'
    if args.compressed_scales:name+='_cscale'
    if args.short_scales:name+='_sscale'
    if args.projection_pdl:name+='_pdl'
    if args.attention_pdl:name+='_apdl'
    if args.codec_clock:name+='_cc'
    if args.first_audio_graph:name+='_fag'
    if args.bulk_prefetch:name+='_bulk'
    if args.prefill_qkv:name+='_pqkv'
    if args.prefill_pointwise:name+='_ppw'
    if args.async_output:name+='_acopy'
    if args.output_weight_prefetch:name+='_opre'
    if args.qkv_cluster:name+='_qcluster'
    if args.down_tile8:name+='_down8'
    if args.gateup_compiler:name+='_gup38'
    if args.prefill_buckets!=[128,256,512]:name+='_pb'+'_'.join(map(str,args.prefill_buckets))
    runs = []
    for i in range(args.runs+1):
        torch.manual_seed(1234+i)
        chunks = list(engine.stream(fixture['text'], fixture['reference'], max_new_tokens=args.max_new_tokens))
        assert chunks and all(c.pcm.numel() == 1920 and torch.isfinite(c.pcm).all() for c in chunks)
        metrics = engine.last_metrics.copy()
        print(f"RUN {i}: TTFA={metrics['ttfa_ms']:.3f}ms frames={metrics['frames']} truncated={metrics['truncated']}", flush=True)
        if i:
            runs.append(metrics)
        if i == 1:
            sf.write(RESULTS / (name+'.wav'), torch.cat([c.pcm for c in chunks]).numpy(), 24000)
    result = {
        'mode': mode, 'tiled': args.tiled, 'attention_backend': args.attention_backend, 'codebooks': 32, 'codec_dtype': 'float32',
        'prefill_dtype': 'FP8 W8A8 projections, BF16 other activations' if args.fp8_prefill else 'bfloat16', 'quantized_projections': 'MLP only' if args.mode == 'fp8_mlp' else ('none' if args.mode == 'none' else 'Attention QKV and output, MLP gate/up and down'),
        'embeddings_and_lm_heads': 'bfloat16', 'decode': stats(decode),
        'ttfa': stats([r['ttfa_ms'] for r in runs]), 'runs': runs, 'validation': validation,
        'workload': 'Warm batch one, complete Chinese text, cached 3.112 s voice reference, first 80 ms PCM chunk; processor, prefill, 32-codebook generation, FP32 codec and CPU PCM copy included; network excluded.',
        'deployed': False,
        'torch_version':torch.__version__,
        'attention_block':args.attention_block,'attention_warps':args.attention_warps,
        'fused_residual':args.fused_residual,
        'fused_gateup':args.fused_gateup,
        'fp8_prefill':args.fp8_prefill,
        'calibration_config':calibration_config,
        'calibration_path':args.calibration,
        'dp4a_reciprocal':args.dp4a_reciprocal,
        'fused_dp4a':args.fused_dp4a,
        'grouped_activation':args.grouped_activation,
        'prefill_buckets':args.prefill_buckets,
        'dp4a_gateup':args.dp4a_gateup,
        'norm_layout_limit':args.norm_layout_limit,
        'packing_config':packing_config,
        'decode_buckets':decode_buckets.stats() if decode_buckets else None,
        'attention_quant':args.attention_quant,
        'native_attention':args.native_attention,
        'gateup_quant':args.gateup_quant,
        'scaled_dp4a':args.scaled_dp4a,
        'norm_projection':args.norm_projection,
        'qkv_load_policy':args.qkv_load_policy,
        'compressed_scales':compressed_scales_config,
        'short_scales':short_scales_config,
        'projection_pdl':projection_pdl_config,
        'attention_pdl':attention_pdl_config,
        'codec_clock':args.codec_clock,
        'first_audio_graph':first_audio_graph_config,
        'bulk_prefetch':bulk_prefetch_config,
        'prefill_qkv':prefill_qkv_config,
        'prefill_pointwise':prefill_pointwise_config,
        'async_output':async_output_config,
        'output_weight_prefetch':output_weight_prefetch_config,'qkv_cluster':qkv_cluster_config,'down_tile8':down_tile8_config,'gateup_compiler':gateup_compiler_config,
        'group128_config':group128_config,
        'group128_residual_config':group128_residual_config,
    }
    if args.calibration:result['quantized_projections']='Calibrated INT4 attention QKV/output and MLP gate/up/down'
    (RESULTS / (name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs', 'validation')}, indent=2), flush=True)
    print('VALIDATION', json.dumps({k:v for k,v in validation.items() if k != 'per_step_relative_rms'}), flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        list(engine.stream(fixture['text'], fixture['reference'], max_new_tokens=34))
    (RESULTS / (name+'_profile.txt')).write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
    engine.codec.close()


if __name__ == '__main__':
    main()
