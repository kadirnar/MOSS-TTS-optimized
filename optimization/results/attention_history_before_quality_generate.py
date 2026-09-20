"""Small bilingual, four-reference cloning suite; generated audio is never cached."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .common import ROOT,RESULTS,load_models
from .streaming import StreamingTTS
from .reference_encoder import ReferenceEncoder


PROMPTS = {
    'Chinese': [
        '你好，这是一个语音克隆测试。我们正在优化流式语音合成的速度。',
        '明天上午九点开会，请带上最新的报告。',
        '窗外下着小雨，屋里飘着刚泡好的茶香。',
        '请先确认订单编号，再检查收货地址是否正确。'],
    'English': [
        'Hello, this is a voice cloning test. We are improving the speed of streaming speech synthesis.',
        'The meeting starts at nine tomorrow morning. Please bring the updated report.',
        'Rain tapped against the window while the kettle quietly came to a boil.',
        'Please confirm the order number and check that the delivery address is correct.']}
VOICES = [('zh0','Chinese','reference_zh.wav'),('zh1','Chinese','reference_zh_1.wav'),
          ('en0','English','reference_en.m4a'),('en1','English','reference_en_1.mp3')]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('none','fp8_all'),required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--prompt-file',type=Path,help='Frozen JSON mapping Chinese/English to nonempty text lists')
    p.add_argument('--seed-base',type=int,default=1234)
    p.add_argument('--fused-residual',action='store_true')
    p.add_argument('--fused-gateup',action='store_true')
    p.add_argument('--attention-block',type=int,default=128)
    p.add_argument('--fp8-prefill',action='store_true')
    p.add_argument('--prefix-cache',type=int,default=0)
    p.add_argument('--calibration')
    p.add_argument('--calibrated-backend',choices=('marlin','dp4a'),default='marlin')
    p.add_argument('--dp4a-reciprocal',action='store_true')
    p.add_argument('--fused-dp4a',action='store_true')
    p.add_argument('--grouped-activation',action='store_true')
    p.add_argument('--prefill-buckets',type=int,nargs='+',default=[128,256,512])
    p.add_argument('--dp4a-gateup',action='store_true')
    p.add_argument('--norm-layout-limit',type=int,choices=(0,8),default=0)
    p.add_argument('--packing-plan')
    p.add_argument('--g128-plan')
    p.add_argument('--g128-residual-calibration')
    p.add_argument('--g128-residual-plan')
    p.add_argument('--decode-buckets',action='store_true')
    p.add_argument('--attention-quant',action='store_true')
    p.add_argument('--native-attention',action='store_true')
    p.add_argument('--gateup-quant',action='store_true')
    p.add_argument('--scaled-dp4a',action='store_true')
    p.add_argument('--norm-projection',action='store_true')
    p.add_argument('--qkv-load-policy',action='store_true')
    p.add_argument('--compressed-scales',action='store_true',help='Lossless SM90 compression of selected FP32 scale allocations')
    p.add_argument('--short-scales',action='store_true',help='Exact BF16 scale storage with explicit Gluon reduction layouts')
    p.add_argument('--projection-pdl',action='store_true',help='SM90 projection dependency overlap with exact short scales')
    p.add_argument('--attention-pdl',action='store_true',help='Extend SM90 dependency overlap through native attention and quantization')
    p.add_argument('--codec-clock',action='store_true',help='Exact FP32 codec with shared stage counters and first-frame specialization')
    p.add_argument('--first-audio-graph',action='store_true',help='Run the initial 32 audio steps in one CUDA graph')
    p.add_argument('--bulk-prefetch',action='store_true',help='Prefix-only asynchronous bulk L2 hints for selected projections')
    p.add_argument('--prefill-qkv',action='store_true',help='Fuse BF16 prefill Q/K normalization, rotary and KV writes')
    p.add_argument('--prefill-pointwise',action='store_true',help='Fuse BF16 prefill activation/product and residual/normalization')
    p.add_argument('--async-output',action='store_true',help='Stage exact G32 output weights with SM90 asynchronous copies')
    p.add_argument('--output-weight-prefetch',action='store_true',help='Preload exact G32 attention-output weights into registers before PDL wait')
    p.add_argument('--qkv-cluster',action='store_true',help='Fuse G32 QKV/head preparation using SM90 clustered cubins')
    p.add_argument('--down-tile8',action='store_true',help='Use eight-row/four-warp G32 down projections with clustered QKV')
    p.add_argument('--gateup-compiler',action='store_true',help='Use exact one-CTA gate/up cubins from isolated Triton 3.8')
    p.add_argument('--g64-norm-projections',choices=('up','up_qkv'),help='Experimental G64 weights/G32 activations; changes audio and requires quality review')
    p.add_argument('--audio-head-buckets',action='store_true')
    args=p.parse_args()
    if args.async_output and args.output_weight_prefetch:raise ValueError('Choose one output staging strategy')
    if args.gateup_compiler and not args.down_tile8:raise ValueError('Gate/up compiler option requires the eight-row down preset')
    if args.down_tile8 and not args.qkv_cluster:raise ValueError('Eight-row down tile requires clustered QKV')
    if args.qkv_cluster and not (args.bulk_prefetch and args.output_weight_prefetch):raise ValueError('Clustered QKV requires the selected bulk and register-preload preset')
    if args.qkv_cluster and args.g64_norm_projections:raise ValueError('Clustered QKV requires the selected G32 model')
    if (args.async_output or args.output_weight_prefetch) and not (args.short_scales and args.projection_pdl and args.attention_pdl):raise ValueError('Async output requires G32 short scales and attention/projection PDL')
    if args.g64_norm_projections and not args.bulk_prefetch:raise ValueError('G64 norm experiment requires the selected bulk/PDL preset')
    if args.projection_pdl and (not args.short_scales or args.qkv_load_policy):raise ValueError('Projection PDL requires exact short scales without the experimental QKV load policy')
    if args.attention_pdl and not (args.projection_pdl and args.native_attention and args.attention_quant):raise ValueError('Attention PDL requires projection PDL and native quantized attention')
    if args.qkv_load_policy and not args.norm_projection:raise ValueError('QKV load policy requires G32 norm-projection fusion')
    if args.compressed_scales and (not args.norm_projection or args.g128_plan or args.g128_residual_calibration):raise ValueError('Compressed scales require the selected G32 norm-projection preset')
    if args.short_scales and (not args.norm_projection or args.compressed_scales or args.g128_plan or args.g128_residual_calibration):raise ValueError('Short scales require G32 norm-projection without compressed FP32 scales or G128')
    if bool(args.g128_residual_calibration)!=bool(args.g128_residual_plan) or (args.g128_residual_calibration and (args.g128_plan or not args.norm_projection)):
        raise ValueError('Residual G128 requires its export and plan, plus the G32 norm-projection preset')
    prompts=PROMPTS
    prompt_source=None
    if args.prompt_file:
        raw=args.prompt_file.read_bytes()
        prompts=json.loads(raw)
        if set(prompts)!=set(PROMPTS) or any(not isinstance(v,list) or not v or any(not isinstance(t,str) or not t.strip() for t in v) for v in prompts.values()):
            raise ValueError('Prompt file requires nonempty Chinese and English text lists')
        prompt_source={'path':str(args.prompt_file),'sha256':hashlib.sha256(raw).hexdigest()}
    count=sum(len(prompts[language]) for _,language,_ in VOICES)
    if args.g128_plan and (not args.calibration or args.packing_plan or args.scaled_dp4a or args.norm_projection or args.gateup_quant):raise ValueError('G128 kernels require a calibrated export and their separate plan')
    if args.audio_head_buckets and not (args.decode_buckets and args.norm_projection):raise ValueError('Audio head buckets require the selected norm-projection/context preset')
    if args.norm_layout_limit and not args.fused_dp4a:raise ValueError('Norm layout requires fused DP4A')
    if args.calibration and (args.mode!='none' or args.fp8_prefill):raise ValueError('Calibrated weights require BF16 prefill and mode none')
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Unsafe tag')
    folder=RESULTS/'quality_suite'/args.tag
    if (args.g64_norm_projections or args.async_output or args.output_weight_prefetch or args.qkv_cluster) and (folder/'manifest.json').exists():raise FileExistsError('Preserve experimental quality evidence')
    folder.mkdir(parents=True,exist_ok=True)
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,weight_quantization=args.mode,
        attention_block=args.attention_block,fused_residual=args.fused_residual,fused_gateup=args.fused_gateup,fp8_prefill=args.fp8_prefill,codec_clock=args.codec_clock)
    if args.calibration:
        from .calibrated_backend import install_calibrated
        install_calibrated(engine.llm,args.calibration,args.calibrated_backend)
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
    if args.packing_plan:
        from .dp4a_packing import install_packing
        install_packing(engine.llm,args.packing_plan)
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
    group64_norm_config=None
    if args.g64_norm_projections:
        from .group64_norm_experiment import enable
        group64_norm_config=enable(engine.llm,('up',) if args.g64_norm_projections=='up' else ('up','qkv'))
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
    head_buckets=None
    if args.audio_head_buckets:
        from .audio_head_buckets import DecodeAudioHeadBuckets
        head_buckets=DecodeAudioHeadBuckets(engine.llm,decode_buckets)
        head_buckets.warmup();head_buckets.install()
    first_audio_graph_config=None
    if args.first_audio_graph:
        from .first_audio_graph import enable
        first_audio_graph_config=enable(engine)
    prefix=None
    if args.prefix_cache:
        from .prefix_cache import PrefixPrefillCache
        prefix=PrefixPrefillCache(engine.llm,processor.tokenizer,prefix_length=args.prefix_cache)
        prefix.warmup()
        prefix.install()
    encoder=ReferenceEncoder(processor,buckets=(40,64))
    encoder.warmup()
    records=[]
    for voice,language,filename in VOICES:
        # Uniform 24 kHz mono reference preparation, at most four seconds.
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(ROOT/'assets/audio'/filename),
            '-t','4','-f','f32le','-ac','1','-ar','24000','pipe:1'])
        wave=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy()).reshape(1,-1)
        reference_path=folder/(voice+'_reference.wav')
        sf.write(reference_path,wave.numpy().reshape(-1),24000)
        start=time.perf_counter()
        reference=encoder.encode(wave,24000)
        encode_ms=(time.perf_counter()-start)*1000
        if prefix is not None:
            # Populate conditioning with a placeholder, before real text arrives.
            dummy=processor([[processor.build_user_message(text='Warmup.',reference=[reference],language=language)]],mode='generation').to('cuda')
            prefix.prefill(dummy.input_ids)
        for j,text in enumerate(prompts[language]):
            seed=args.seed_base+j
            torch.manual_seed(seed)
            chunks=list(engine.stream(text,reference,language=language,max_new_tokens=400))
            audio=torch.cat([chunk.pcm for chunk in chunks])
            assert torch.isfinite(audio).all() and audio.numel()
            name=f'{voice}_{j}'
            path=folder/(name+'.wav')
            sf.write(path,audio.numpy(),24000)
            metrics=engine.last_metrics
            record={'id':name,'voice':voice,'language':language,'text':text,'seed':seed,
                'audio':str(path),'reference':str(reference_path),'reference_source':filename,
                'reference_seconds':wave.numel()/24000,'reference_encode_ms':encode_ms,
                'ttfa_ms':metrics['ttfa_ms'],'frames':metrics['frames'],'truncated':metrics['truncated'],
                'prompt_tokens':metrics['prompt_tokens'],'prefill_ms':metrics['prefill_ms'],
                'prepare_ms':metrics['prepare_ms'],
                'finite':True,'audio_seconds':audio.numel()/24000}
            records.append(record)
            print({k:record[k] for k in ('id','ttfa_ms','frames','truncated')},flush=True)
            (folder/'manifest.json').write_text(json.dumps({'mode':'gptq_'+args.calibrated_backend if args.calibration else args.mode,'codebooks':32,
                'codec_dtype':'float32','torch_version':torch.__version__,
                'prompt_source':prompt_source,'seed_base':args.seed_base,
                'fp8_prefill':args.fp8_prefill,'prefix_cache':prefix.stats() if prefix is not None else None,
                'calibration':args.calibration,
                'dp4a_reciprocal':args.dp4a_reciprocal,
                'fused_dp4a':args.fused_dp4a,'grouped_activation':args.grouped_activation,
                'prefill_buckets':args.prefill_buckets,
                'dp4a_gateup':args.dp4a_gateup,
                'norm_layout_limit':args.norm_layout_limit,
                'packing_plan':args.packing_plan,
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
                'group64_norm_experiment':group64_norm_config,
                'group128_config':group128_config,
                'group128_residual_config':group128_residual_config,
                'audio_head_buckets':head_buckets.stats() if head_buckets else None,
                'attention_block':args.attention_block,'fused_residual':args.fused_residual,'fused_gateup':args.fused_gateup,'records':records,
                'scope':f'{count} utterances, four supplied reference assets, Chinese and English; diagnostic suite, not a production quality guarantee.'},indent=2,ensure_ascii=False)+'\n')
    engine.codec.close()


if __name__=='__main__':main()
