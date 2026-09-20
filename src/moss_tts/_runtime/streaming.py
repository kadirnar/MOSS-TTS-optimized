"""True incremental PCM generation with voice cloning; no utterance buffering."""
import time
import torch
from .llm import FastLLM, sample_topk
from .codec import StreamingCodec


from ..types import AudioChunk


class StreamingTTS:
    def __init__(self, model, codec, processor, *, graph=True, fused=True, max_length=1024, greedy=False, attention_block=128, attention_warps=4, fused_residual=False, fused_gateup=False, codec_clock=False):
        self.model=model
        self.processor=processor
        self.llm=FastLLM(model,max_length=max_length,graph=graph,fused=fused,greedy=greedy,attention_block=attention_block,attention_warps=attention_warps,fused_residual=fused_residual,fused_gateup=fused_gateup)
        if codec_clock:
            from .codec_clock import ClockedStreamingCodec
            self.codec=ClockedStreamingCodec(codec,graph=graph)
        else:self.codec=StreamingCodec(codec,graph=graph)
        self.greedy=greedy
        self.last_metrics={}
        self._busy=False
        self._first_audio_graphs=None

    @torch.inference_mode()
    def warmup(self,prefill_buckets=(128,256,512)):
        self.llm.warmup()
        if self.llm.graph_enabled:self.llm.capture_prefill(prefill_buckets)
        self.codec.warmup()

    def stream(self,text,reference_codes=None,language="English",max_new_tokens=400):
        if self._busy:
            raise RuntimeError("StreamingTTS owns one KV state; serialize requests or use separate instances")
        if max_new_tokens<33:
            raise ValueError("At least 33 generated steps are needed for a complete 32-codebook frame")
        self._busy=True
        try:
            # A generator must enter inference_mode inside its body: a decorator
            # around construction alone does not cover execution at each yield.
            with torch.inference_mode():
                yield from self._stream(text,reference_codes,language,max_new_tokens)
        finally:
            self._busy=False

    def _stream(self,text,reference_codes,language,max_new_tokens):
        start=time.perf_counter()
        cfg=self.model.config
        reference=None if reference_codes is None else [reference_codes]
        inputs=self.processor([[self.processor.build_user_message(text=text,reference=reference,language=language)]],mode="generation").to("cuda")
        prompt=inputs.input_ids.shape[1]
        if prompt+max_new_tokens>self.llm.max_length:
            raise ValueError("Prompt plus generation budget exceeds KV capacity")
        self.codec.reset()
        buffer=torch.full((max_new_tokens,33),1024,dtype=torch.long,device="cuda")
        channels=torch.arange(32,device="cuda")
        torch.cuda.synchronize()
        prepare_ms=(time.perf_counter()-start)*1000
        logits,_=self.llm.prefill(inputs.input_ids)
        torch.cuda.synchronize()
        prefill_ms=(time.perf_counter()-start)*1000-prepare_ms
        first_audio_row=None
        audio_length=0
        delay_length=-1
        in_audio=False
        emitted=0
        step_times=[]
        codec_times=[]
        burst_times=[]
        first_audio_ms=None
        nxt=None  # Set by text sampling before entering the audio-delay loop.
        step=0
        last_step=-1
        while step<max_new_tokens:
            tick=time.perf_counter()
            burst=None
            if self._first_audio_graphs and in_audio and step==first_audio_row and audio_length==1 and delay_length==-1 and step+32<=max_new_tokens:
                choices=[cap for cap in self._first_audio_graphs if prompt+step-1+32<=cap]
                if choices:burst=self._first_audio_graphs[min(choices)]
            if burst is not None:
                history,status=burst.run(nxt,prompt+step-1)
                buffer[step:step+32].copy_(history)
                text_token,audio_length,delay_length,_=status.tolist()
                nxt=self.llm.ids.clone()
                burst_times.append({'start_step':step,'steps':32,'capacity':burst.capacity,
                                    'elapsed_ms':(time.perf_counter()-tick)*1000})
                # Per-step wall times are not observable inside one GPU graph.
                # Preserve index alignment and report the measured group above.
                step_times.extend([None]*32)
                step+=31
            else:
                if in_audio:
                    nxt=self.llm.step(nxt,prompt+step-1,audio_length,delay_length)
                    text_token=int(nxt[0,0,0].item())
                else:
                    if step>0:
                        logits=self.llm.full_step(nxt,prompt+step-1)
                    logits=logits.clone()
                    banned=[cfg.pad_token_id,cfg.audio_assistant_gen_slot_token_id,cfg.audio_assistant_delay_slot_token_id,cfg.audio_end_token_id]
                    if step<=32:banned.append(cfg.im_end_token_id)
                    logits[:,banned]=-torch.inf
                    text_token=int(sample_topk(logits,0 if self.greedy else 1.5,50,1.).item())
                    nxt=torch.full((1,1,33),1024,dtype=torch.long,device="cuda")
                    nxt[0,0,0]=text_token
                buffer[step].copy_(nxt[0,0])
                if text_token==cfg.audio_start_token_id:
                    in_audio=True
                    first_audio_row=step+1
                if text_token in (cfg.audio_start_token_id,cfg.audio_assistant_gen_slot_token_id,cfg.audio_assistant_delay_slot_token_id):
                    audio_length+=1
                if delay_length<0 and text_token==cfg.audio_assistant_delay_slot_token_id:
                    delay_length=0
                if delay_length>=0:delay_length+=1
                step_times.append((time.perf_counter()-tick)*1000)
            if first_audio_row is not None and step>=first_audio_row+emitted+31:
                rows=first_audio_row+emitted+channels
                codes=buffer[rows,channels+1].reshape(32,1,1)
                # Do not pass a partially drained/padded codebook frame to codec.
                if bool((codes<1024).all()):
                    tick=time.perf_counter()
                    audio=self.codec.decode(codes).float().flatten().cpu()
                    codec_times.append((time.perf_counter()-tick)*1000)
                    elapsed=(time.perf_counter()-start)*1000
                    if first_audio_ms is None:first_audio_ms=elapsed
                    emitted+=1
                    yield AudioChunk(audio,emitted-1,elapsed,sample_rate=24000)
            last_step=step
            if text_token in (cfg.audio_end_token_id,cfg.im_end_token_id):break
            step+=1
        self.last_metrics={"ttfa_ms":first_audio_ms,"prepare_ms":prepare_ms,"prefill_ms":prefill_ms,
            "step_ms":step_times,"codec_ms":codec_times,"frames":emitted,
            "total_ms":(time.perf_counter()-start)*1000,"prompt_tokens":prompt,
            "truncated":last_step+1==max_new_tokens and text_token!=cfg.audio_end_token_id,
            "first_audio_graph_ms":burst_times}
