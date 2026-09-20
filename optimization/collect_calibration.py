"""Collect BF16 decoder activations using prompts separate from quality tests.

Generate real 32-codebook autoregressive sequences, then teacher-force their
inputs in one causal pass. Only generated-sequence activations enter calibration.
The full-sequence pass changes floating-point ordering slightly, as in standard
post-training quantization calibration; this is not a quality evaluation set.
"""
import json
import hashlib
import subprocess
from collections import defaultdict
import numpy as np
import torch
from .common import ROOT,RESULTS,TTS_REVISION,CODEC_REVISION,load_models
from .streaming import StreamingTTS
from .reference_encoder import ReferenceEncoder
from .quality_generate import VOICES,PROMPTS as EVAL_PROMPTS


TEXTS={
    'Chinese':[
        '清晨的公园里，有人在跑步，也有人坐在长椅上看书。',
        '火车即将到达终点站，请整理好随身物品，准备下车。',
        '这家小店的面包每天现做，路过的时候总能闻到香味。'],
    'English':[
        'A gentle breeze moved through the garden as the children played outside.',
        'Your train will arrive at the final station soon. Remember to collect your belongings.',
        'The bakery on the corner makes fresh bread every morning, and it smells wonderful.']}


@torch.inference_mode()
def main():
    folder=RESULTS/'calibration_v1'
    folder.mkdir(exist_ok=True)
    for language,texts in TEXTS.items():assert not set(texts)&set(EVAL_PROMPTS[language])
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,fused_residual=True,fused_gateup=True)
    engine.warmup()
    encoder=ReferenceEncoder(processor,buckets=(40,64));encoder.warmup()
    fast=engine.llm
    original_step,original_full=fast.step,fast.full_step
    observed={}
    def step(ids,position,*args):
        observed[position]=ids.detach().cpu().clone()
        return original_step(ids,position,*args)
    def full(ids,position):
        observed[position]=ids.detach().cpu().clone()
        return original_full(ids,position)
    fast.step,fast.full_step=step,full
    records=[]
    for voice,language,filename in VOICES:
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(ROOT/'assets/audio'/filename),'-t','4','-f','f32le','-ac','1','-ar','24000','pipe:1'])
        reference=encoder.encode(torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy()).reshape(1,-1),24000)
        for j,text in enumerate(TEXTS[language]):
            observed.clear()
            ids=processor([[processor.build_user_message(text=text,reference=[reference],language=language)]],mode='generation').input_ids
            prompt=ids.shape[1]
            torch.manual_seed(8800+j)
            chunks=list(engine.stream(text,reference,language=language,max_new_tokens=400))
            assert chunks and not engine.last_metrics['truncated']
            positions=sorted(observed)
            assert positions==list(range(prompt,prompt+len(positions))),positions
            sequence=torch.cat([ids.cpu()]+[observed[p] for p in positions],dim=1)
            record={'id':f'{voice}_{j}','voice':voice,'language':language,'text':text,
                'prompt_tokens':prompt,'decode_tokens':len(positions),'frames':len(chunks),
                'seed':8800+j,'reference_source':filename}
            torch.save(sequence,folder/(record['id']+'_tokens.pt'))
            records.append(record)
            print('GENERATED',record,flush=True)
    fast.step,fast.full_step=original_step,original_full
    inputs=defaultdict(list)
    prompt_boundary=0
    def capture(name,tensor):
        # Only rows actually fed through the autoregressive decoder are used.
        value=tensor[:,prompt_boundary:].reshape(-1,tensor.shape[-1]).detach().cpu().contiguous()
        assert value.numel() and torch.isfinite(value).all()
        inputs[name].append(value)
    handles=[]
    for i,layer in enumerate(model.language_model.layers):
        a,m=layer.self_attn,layer.mlp
        # Multi-token F.linear math is unchanged; invoking the module makes its
        # pre-hook visible without adding hooks to the measured decode path.
        m._triton_gemv=False
        handles.append(a.register_forward_pre_hook(lambda mod,args,kwargs,i=i:capture(f'{i:02d}_qkv',kwargs['hidden_states']),with_kwargs=True))
        handles.append(a.o_proj.register_forward_pre_hook(lambda mod,args,i=i:capture(f'{i:02d}_out',args[0])))
        handles.append(m.register_forward_pre_hook(lambda mod,args,i=i:capture(f'{i:02d}_up',args[0])))
        handles.append(m.down_proj.register_forward_pre_hook(lambda mod,args,i=i:capture(f'{i:02d}_down',args[0])))
    for record in records:
        sequence=torch.load(folder/(record['id']+'_tokens.pt'),weights_only=True).cuda()
        prompt_boundary=record['prompt_tokens']
        pos=torch.arange(sequence.shape[1],device='cuda')
        mask=(fast.kv_index[None,:]<=pos[:,None]).view(1,1,sequence.shape[1],-1)
        fast.hidden(sequence,pos,mask)
        print('REPLAYED',record['id'],flush=True)
    for handle in handles:handle.remove()
    files={}
    for name,values in sorted(inputs.items()):
        tensor=torch.cat(values)
        path=folder/(name+'.pt')
        torch.save(tensor,path)
        files[name]={'shape':list(tensor.shape),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert len(files)==36*4
    manifest={'tts_revision':TTS_REVISION,'codec_revision':CODEC_REVISION,'torch':torch.__version__,
        'codebooks':32,'records':records,'activations':files,'calibration_texts_overlap_evaluation':False,
        'reference_assets_shared_with_evaluation':True,
        'scope':'12 generated calibration utterances, four supplied voices, three new texts per language. BF16 teacher; decoded input rows only. Full causal replay, no quantized weights. Not a quality evaluation.'}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('SAVED',len(files),'activation matrices',sum(r['decode_tokens'] for r in records),'rows each',flush=True)
    engine.codec.close()


if __name__=='__main__':main()
