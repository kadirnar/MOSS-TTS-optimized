import torch
import soundfile as sf
from .common import ROOT,RESULTS,load_models,timed,stats,save_json
from .reference_encoder import ReferenceEncoder


@torch.inference_mode()
def main():
    model,codec,processor=load_models()
    wave,sr=sf.read(ROOT/'assets/audio/reference_zh.wav',dtype='float32')
    wave=torch.from_numpy(wave).reshape(1,-1)
    expected=processor.encode_audios_from_wav([wave],sr)[0]
    baseline=[]
    for _ in range(10):
        _,ms=timed(lambda:processor.encode_audios_from_wav([wave],sr));baseline.append(ms)
    encoder=ReferenceEncoder(processor,buckets=(40,64))
    encoder.warmup()
    values=[]
    for _ in range(10):
        codes,ms=timed(lambda:encoder.encode(wave,sr));values.append(ms)
    match=(codes==expected).float().mean().item()
    save_json('encoder.json',{'upstream':stats(baseline),'cuda_graph':stats(values),
        'code_agreement':match,'codes_shape':list(codes.shape),
        'dtype':'FP32','reference_seconds':wave.numel()/sr,
        'note':'Graph bucket changes matrix shapes. Code agreement is checked before selecting the path.'})
    assert match>0.99,match

if __name__=='__main__':main()
