import torch
import soundfile as sf
from .common import ROOT,RESULTS,load_models,save_json
from .reference_encoder import ReferenceEncoder

@torch.inference_mode()
def main():
    model,codec,processor=load_models()
    encoder=ReferenceEncoder(processor)
    encoder.warmup()
    raw,sr=sf.read(ROOT/'assets/audio/reference_zh.wav',dtype='float32')
    wave=torch.from_numpy(raw).reshape(1,-1).repeat(1,5)
    tests={}
    for seconds in [0.24,0.64,1.28,2.56,3.112,4.8,8.8,14.5]:
        x=wave[:,:int(seconds*sr)]
        reference=processor.encode_audios_from_wav([x],sr)[0]
        output=encoder.encode(x,sr)
        match=(output==reference).float().mean().item()
        tests[str(seconds)]={'frames':output.shape[0],'code_agreement':match}
        assert match>0.99,tests[str(seconds)]
    save_json('encoder_validation.json',tests)

if __name__=='__main__':main()
