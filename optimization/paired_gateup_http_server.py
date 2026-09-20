"""Temporary benchmark server: same production transport, paired decode graphs.

Generated from paired_http_server.py SHA256 d86520dbfca393e861c4119d1b9a83e1657377db93b332a6c26d2ccf39618358.
The only benchmark hooks are extra request identifiers/variant selection,
second graph capture at initialization, and variant/RNG diagnostic metadata.
Bind remains loopback. Never substitute this app for a production service.
"""
import argparse
import asyncio
import base64
import hashlib
import types
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
import io
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .common import ROOT,load_models
from .streaming import StreamingTTS
from .reference_encoder import ReferenceEncoder


class VoiceRequest(BaseModel):
    wav_base64:str=Field(min_length=1,max_length=16_000_000)


class SpeechRequest(BaseModel):
    input:str=Field(min_length=1,max_length=800)
    voice:str='demo'
    language:str='Chinese'
    max_new_tokens:int=Field(default=400,ge=33,le=700)
    seed:int=Field(default=1234,ge=0,le=2**32-1)
    variant:Literal['control','gateup38']='control'
    benchmark_id:str=Field(default='',max_length=80)



@torch.inference_mode()
def setup_variants(engine, state):
    from .gateup_compiler import make_dispatch
    from .llm import FastLLM
    from .decode_buckets import DecodeContextBuckets
    from .audio_head_buckets import DecodeAudioHeadBuckets
    from .first_audio_graph import enable
    fast=engine.llm
    if not fast.qkv_cluster or fast.cfg.n_vq!=32:raise ValueError('Qualified clustered G32 path required')
    if not fast.down_tile8:raise ValueError('Selected eight-row down control required')
    original=fast._bulk_norm_linear
    variants={'control':(fast._decode_context_buckets,fast._audio_head_buckets,engine._first_audio_graphs,original)}
    fn=make_dispatch(original)
    fast._bulk_norm_linear=fn
    fast._decode_context_buckets=None;fast._audio_head_buckets=None;fast.step=types.MethodType(FastLLM.step,fast)
    fast.warmup();contexts=DecodeContextBuckets(fast);contexts.warmup();contexts.install()
    manager=DecodeAudioHeadBuckets(fast,contexts);manager.warmup();manager.install()
    engine._first_audio_graphs=None;enable(engine)
    variants['gateup38']=(contexts,manager,engine._first_audio_graphs,fn)
    def select(name):
        contexts,manager,bursts,functions=variants[name]
        fast._decode_context_buckets=contexts
        fast._audio_head_buckets=manager;fast.step=manager.step;engine._first_audio_graphs=bursts
        fast._bulk_norm_linear=functions
    state['select_variant']=select;select('control')


def create_app(*, attention_block=128, fused_residual=False, fused_gateup=False, weight_quantization='none',calibration=None,packing_plan=None,decode_buckets=False,metrics_file=None,attention_quant=False,native_attention=False,gateup_quant=False,scaled_dp4a=False,cuda_only_seed=False,norm_projection=False,audio_head_buckets=False,compressed_scales=False,short_scales=False,projection_pdl=False,attention_pdl=False,codec_clock=False,first_audio_graph=False,bulk_prefetch=False,prefill_qkv=False,prefill_pointwise=False,async_output=False,output_weight_prefetch=False,qkv_cluster=False,down_tile8=False):
    if not (down_tile8 and qkv_cluster and audio_head_buckets and first_audio_graph and metrics_file):
        raise ValueError('Benchmark requires clustered eight-row control, audio-head/initial-audio graphs and metrics')
    if calibration and weight_quantization!='none':raise ValueError('Calibrated preset requires BF16 source weights')
    if packing_plan and not calibration:raise ValueError('Packed INT4 requires a calibrated export')
    if attention_quant and not calibration:raise ValueError('Fused attention quantizer requires the calibrated preset')
    if gateup_quant and not packing_plan:raise ValueError('Gate/up quantizer fusion requires a packing plan')
    if scaled_dp4a and not gateup_quant:raise ValueError('Scaled DP4A requires gate/up quantizer fusion')
    if norm_projection and not scaled_dp4a:raise ValueError('Norm/projection fusion requires the selected scaled-DP4A preset')
    if audio_head_buckets and not (decode_buckets and norm_projection):raise ValueError('Audio head buckets require the selected norm-projection/context preset')
    if compressed_scales and not norm_projection:raise ValueError('Compressed scales require the selected G32 norm-projection preset')
    if short_scales and (not norm_projection or compressed_scales):raise ValueError('Short scales require G32 norm-projection without compressed FP32 scales')
    if projection_pdl and not short_scales:raise ValueError('Projection PDL requires exact short scales')
    if attention_pdl and not (projection_pdl and native_attention and attention_quant):raise ValueError('Attention PDL requires projection PDL and native quantized attention')
    if async_output and output_weight_prefetch:raise ValueError('Choose one output staging strategy')
    if down_tile8 and not qkv_cluster:raise ValueError('Eight-row down tile requires clustered QKV')
    if qkv_cluster and not (bulk_prefetch and output_weight_prefetch):raise ValueError('Clustered QKV requires the selected bulk and register-preload preset')
    if (async_output or output_weight_prefetch) and not (short_scales and projection_pdl and attention_pdl):raise ValueError('Async output requires G32 short scales and attention/projection PDL')
    prefill_buckets=(128,256,512)
    if calibration:
        # Explicit experimental preset, matching the layout-audited benchmark.
        attention_block=32;fused_residual=True;fused_gateup=False
        prefill_buckets=(128,160,256,512)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='moss-gpu')
    busy=threading.Lock()
    voices={}
    state={}

    def initialize():
        model,codec,processor=load_models()
        engine=StreamingTTS(model,codec,processor,attention_block=attention_block,
            fused_residual=fused_residual,fused_gateup=fused_gateup,weight_quantization=weight_quantization,codec_clock=codec_clock)
        if calibration:
            from .calibrated_backend import install_calibrated,enable_grouped_activation
            from .dp4a_fusions import enable_fusions
            from .dp4a_gateup import enable_gateup
            install_calibrated(engine.llm,calibration,backend='dp4a')
            enable_grouped_activation(engine.llm)
            enable_fusions(engine.llm,layout_limit=8)
            enable_gateup(engine.llm)
            if packing_plan:
                from .dp4a_packing import install_packing
                install_packing(engine.llm,packing_plan)
            if attention_quant:
                from .attention_quant import enable_attention_quant
                enable_attention_quant(engine.llm)
        if native_attention:
            from .attention_native import enable_native_attention
            enable_native_attention(engine.llm)
        if gateup_quant:
            from .dp4a_gateup_quant import enable_gateup_quant
            enable_gateup_quant(engine.llm)
        if scaled_dp4a:
            from .dp4a_scaled import enable_scaled
            enable_scaled(engine.llm)
        if norm_projection:
            from .dp4a_norm_projection import enable_norm_projection
            enable_norm_projection(engine.llm)
        if compressed_scales:
            from .compressed_alloc import enable_compressed_scales
            enable_compressed_scales(engine.llm)
        if short_scales:
            from .short_scales import enable
            enable(engine.llm)
        if projection_pdl:
            from .projection_pdl import enable
            enable(engine.llm)
        if attention_pdl:
            from .attention_pdl import enable
            enable(engine.llm)
        if bulk_prefetch:
            from .bulk_prefetch import enable
            state['bulk_prefetch']=enable(engine.llm)
        if prefill_qkv:
            from .prefill_qkv import enable
            state['prefill_qkv']=enable(engine.llm)
        if prefill_pointwise:
            from .prefill_pointwise import enable
            state['prefill_pointwise']=enable(engine.llm)
        if async_output:
            from .async_output import enable
            state['async_output']=enable(engine.llm)
        if output_weight_prefetch:
            from .async_output import enable
            state['output_weight_prefetch']=enable(engine.llm,register_preload=True)
        if qkv_cluster:
            from .qkv_cluster_model import enable
            state['qkv_cluster']=enable(engine.llm)
        if down_tile8:
            from .down_tile import enable
            state['down_tile8']=enable(engine.llm)
        engine.warmup(prefill_buckets)
        if decode_buckets:
            from .decode_buckets import DecodeContextBuckets
            manager=DecodeContextBuckets(engine.llm)
            manager.warmup();manager.install()
            state['decode_buckets']=manager
        if audio_head_buckets:
            from .audio_head_buckets import DecodeAudioHeadBuckets
            heads=DecodeAudioHeadBuckets(engine.llm,manager)
            heads.warmup();heads.install()
            state['audio_head_buckets']=heads
        if first_audio_graph:
            from .first_audio_graph import enable
            state['first_audio_graph']=enable(engine)
        setup_variants(engine,state)
        reference_encoder=ReferenceEncoder(processor)
        reference_encoder.warmup()
        wave,sr=sf.read(ROOT/'assets/audio/reference_zh.wav',dtype='float32')
        with torch.inference_mode():
            voices['demo']=reference_encoder.encode(torch.from_numpy(wave).reshape(1,-1),sr)
        state['engine']=engine
        state['reference_encoder']=reference_encoder
        # Exercise the full request path, including stochastic graph sampling.
        list(engine.stream('你好，这是一段测试语音。',voices['demo'],max_new_tokens=96))

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.get_running_loop().run_in_executor(pool,initialize)
        yield
        pool.shutdown(wait=True,cancel_futures=True)

    app=FastAPI(lifespan=lifespan)

    @app.get('/health')
    def health():
        return {'benchmark_only':True,'variants':['control','gateup38'],'ready':'engine' in state,'model':'OpenMOSS-Team/MOSS-TTS-v1.5',
            'sample_rate':24000,'channels':1,'pcm_format':'signed 16-bit little endian',
            'codebooks':32,'concurrency':1,'cached_voices':len(voices),
            'llm_dtype':'bfloat16','codec_dtype':'float32',
            'weight_quantization':'gptq_dp4a' if calibration else weight_quantization,
            'calibrated_preset':'g32_grouped_norm_layout8_gateup' if calibration else None,
            'packed_int4':bool(packing_plan),'decode_context_buckets':decode_buckets,
            'attention_quant':attention_quant,
            'native_attention':native_attention,
            'gateup_quant':gateup_quant,
            'scaled_dp4a':scaled_dp4a,
            'norm_projection':norm_projection,
            'audio_head_buckets':audio_head_buckets,
            'compressed_scales':compressed_scales,
            'short_scales':short_scales,
            'projection_pdl':projection_pdl,
            'attention_pdl':attention_pdl,
            'codec_clock':codec_clock,
            'first_audio_graph':first_audio_graph,
            'bulk_prefetch':bulk_prefetch,
            'prefill_qkv':prefill_qkv,
            'prefill_pointwise':prefill_pointwise,
            'async_output':async_output,
            'output_weight_prefetch':output_weight_prefetch,
            'qkv_cluster':qkv_cluster,'down_tile8':down_tile8,
            'request_seed_scope':'current_cuda_device' if cuda_only_seed else 'all_devices',
            'prefill_buckets':prefill_buckets,
            'attention_block':attention_block,'fused_residual':fused_residual,'fused_gateup':fused_gateup}

    @app.post('/v1/voices')
    async def add_voice(request:VoiceRequest):
        if not busy.acquire(False):raise HTTPException(429,'GPU is serving another request')
        def encode():
            try:
                if len(voices)>=64:raise ValueError('Voice cache is full (64 entries)')
                raw=base64.b64decode(request.wav_base64,validate=True)
                wave,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
                if not 0.2<=len(wave)/sr<=15:raise ValueError('Reference must be between 0.2 and 15 seconds')
                if not np.isfinite(wave).all():raise ValueError('Reference contains non-finite samples')
                wave=torch.from_numpy(wave.mean(axis=1)).reshape(1,-1)
                start=time.perf_counter()
                with torch.inference_mode():
                    codes=state['reference_encoder'].encode(wave,sr)
                name=uuid.uuid4().hex
                voices[name]=codes
                return {'voice':name,'reference_encode_ms':(time.perf_counter()-start)*1000}
            finally:busy.release()
        try:return await asyncio.get_running_loop().run_in_executor(pool,encode)
        except (ValueError,RuntimeError) as e:raise HTTPException(400,str(e)) from e

    @app.post('/v1/audio/speech')
    async def speech(request:SpeechRequest):
        if request.voice not in voices:raise HTTPException(404,'Unknown voice; register one at /v1/voices')
        if not busy.acquire(False):raise HTTPException(429,'GPU is serving another request')
        loop=asyncio.get_running_loop()
        queue=asyncio.Queue(maxsize=8)
        cancelled=threading.Event()

        def send(item):
            future=asyncio.run_coroutine_threadsafe(queue.put(item),loop)
            while not cancelled.is_set():
                try:future.result(timeout=0.1);return True
                except FutureTimeout:continue
            future.cancel()
            return False

        def produce():
            generator=None
            completed=False
            dispatch_start=time.perf_counter()
            first_ready_ms=None
            seed_ms=None
            try:
                # This server's sampling and captured RNG state live on its
                # worker's current CUDA device. The opt-in path avoids resetting
                # unused CPU/other-device generators on every request.
                state['select_variant'](request.variant)
                if cuda_only_seed:torch.cuda.manual_seed(request.seed)
                else:torch.manual_seed(request.seed)
                seed_ms=(time.perf_counter()-dispatch_start)*1000
                generator=state['engine'].stream(request.input,voices[request.voice],request.language,request.max_new_tokens)
                for chunk in generator:
                    if first_ready_ms is None:first_ready_ms=(time.perf_counter()-dispatch_start)*1000
                    pcm=(chunk.pcm.clamp(-1,1)*32767).to(torch.int16).numpy().astype('<i2',copy=False).tobytes()
                    if not send(pcm):break
                else:completed=True
            except Exception as e:send(e)
            finally:
                if generator is not None:generator.close()
                if metrics_file:
                    # Developer diagnostics only; no text, reference audio or voice ID.
                    metrics=state['engine'].last_metrics.copy() if completed else None
                    record={'completed':completed,'seed_ms':seed_ms,'first_pcm_ready_ms':first_ready_ms,'engine':metrics,
                        'variant':request.variant,'benchmark_id':request.benchmark_id,
                        'rng_sha256':hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest() if completed else None}
                    try:
                        with Path(metrics_file).open('a') as f:f.write(json.dumps(record)+'\n')
                    except OSError:pass
                busy.release()
                if not cancelled.is_set():send(None)

        pool.submit(produce)
        try:first=await queue.get()
        except BaseException:
            cancelled.set();raise
        if isinstance(first,Exception):
            cancelled.set()
            raise HTTPException(400,str(first))
        if first is None:raise HTTPException(422,'No audio generated within the token budget')

        async def body():
            try:
                yield first
                while True:
                    item=await queue.get()
                    if item is None:return
                    if isinstance(item,Exception):raise RuntimeError('Synthesis failed after audio began') from item
                    yield item
            finally:cancelled.set()

        return StreamingResponse(body(),media_type='audio/pcm',headers={
            'X-Audio-Sample-Rate':'24000','X-Audio-Channels':'1','X-Audio-Format':'s16le',
            'Cache-Control':'no-store','X-Benchmark-Variant':request.variant})
    return app


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=18080)
    parser.add_argument('--attention-block',type=int,choices=(32,128),default=128)
    parser.add_argument('--fused-residual',action='store_true')
    parser.add_argument('--fused-gateup',action='store_true')
    parser.add_argument('--weight-quantization',choices=('none','fp8_all'),default='none',
        help='FP8 is experimental; all 32 codebooks and FP32 codec are retained')
    parser.add_argument('--calibration',help='Experimental complete GPTQ export: grouped DP4A, audited normalization layout, fused gate/up, 160-token prefill bucket')
    parser.add_argument('--packing-plan',help='Optional validated INT4 packing plan for a calibrated export')
    parser.add_argument('--decode-buckets',action='store_true',help='Capture position-bounded decode attention graphs')
    parser.add_argument('--metrics-file',help='Optional local JSONL stage timing diagnostics; excludes text/audio/voice IDs')
    parser.add_argument('--attention-quant',action='store_true',help='Fuse G32 activation quantization into attention reduction')
    parser.add_argument('--native-attention',action='store_true',help='Use SM90 native B32/W4 attention with fixed arithmetic order')
    parser.add_argument('--gateup-quant',action='store_true',help='Fuse grouped activation quantization into gate/up/SiLU')
    parser.add_argument('--scaled-dp4a',action='store_true',help='Use exact scaled-integer INT4 unpacking')
    parser.add_argument('--norm-projection',action='store_true',help='Fuse residual normalization and quantization into QKV and gate/up projections')
    parser.add_argument('--audio-head-buckets',action='store_true',help='Skip only currently masked future audio heads; retain all 32 codebooks per frame')
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
    parser.add_argument('--cuda-only-seed',action='store_true',help='Seed only the current CUDA device used by the inference worker')
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(attention_block=args.attention_block,fused_residual=args.fused_residual,
        fused_gateup=args.fused_gateup,weight_quantization=args.weight_quantization,calibration=args.calibration,
        packing_plan=args.packing_plan,decode_buckets=args.decode_buckets,metrics_file=args.metrics_file,attention_quant=args.attention_quant,native_attention=args.native_attention,gateup_quant=args.gateup_quant,scaled_dp4a=args.scaled_dp4a,cuda_only_seed=args.cuda_only_seed,norm_projection=args.norm_projection,audio_head_buckets=args.audio_head_buckets,compressed_scales=args.compressed_scales,short_scales=args.short_scales,projection_pdl=args.projection_pdl,attention_pdl=args.attention_pdl,codec_clock=args.codec_clock,first_audio_graph=args.first_audio_graph,bulk_prefetch=args.bulk_prefetch,prefill_qkv=args.prefill_qkv,prefill_pointwise=args.prefill_pointwise,async_output=args.async_output,output_weight_prefetch=args.output_weight_prefetch,qkv_cluster=args.qkv_cluster,down_tile8=args.down_tile8),host='127.0.0.1',port=args.port,log_level='info')

if __name__=='__main__':main()
