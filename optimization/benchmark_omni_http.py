"""Complete streaming serving-engine measurement with the same cloned voice."""
import argparse
import base64
import json
import time
import requests
import soundfile as sf
import numpy as np
import torch
from .common import ROOT,RESULTS,stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--model',default='/workspace/models/moss-tts-v15')
    p.add_argument('--runs',type=int,default=5)
    p.add_argument('--stream-format',choices=('audio',))
    args=p.parse_args()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    reference=base64.b64encode((ROOT/'assets/audio/reference_zh.wav').read_bytes()).decode()
    body={'model':args.model,'input':fixture['text'],'voice':'default',
        'language':'Chinese','ref_audio':'data:audio/wav;base64,'+reference,
        'response_format':'pcm','stream':True,'max_new_tokens':400,'seed':1234}
    if args.stream_format:
        body['stream_format']=args.stream_format
    # This encoding is outside the timer; request upload is inside it.
    encoded=json.dumps(body,ensure_ascii=False).encode()
    session=requests.Session()
    runs=[]
    for i in range(args.runs+1):
        start=time.perf_counter()
        first=None
        audio=bytearray()
        arrivals=[]
        with session.post(args.base_url+'/v1/audio/speech',data=encoded,
                headers={'Content-Type':'application/json'},stream=True,timeout=(10,180)) as response:
            if response.status_code!=200:
                raise RuntimeError(f'{response.status_code}: {response.text[:2000]}')
            content_type=response.headers.get('content-type')
            if 'text/' in (content_type or '') or 'json' in (content_type or ''):
                raise RuntimeError(f'Expected raw PCM; got {content_type}. Select the engine raw-audio stream format.')
            for chunk in response.iter_content(chunk_size=3840):
                if not chunk:continue
                audio.extend(chunk)
                elapsed=(time.perf_counter()-start)*1000
                arrivals.append(elapsed)
                if len(audio)>=3840 and first is None:first=elapsed
        if first is None or len(audio)%2:
            raise RuntimeError('Missing a complete first 80 ms PCM chunk or malformed PCM length')
        if audio[:4]==b'RIFF':
            raise RuntimeError('Expected raw s16le PCM, received a WAV container')
        d={'ttfa_ms':first,'total_ms':(time.perf_counter()-start)*1000,
            'pcm_bytes':len(audio),'content_type':content_type,'chunk_arrival_ms':arrivals}
        print('RUN',i,{k:v for k,v in d.items() if k!='chunk_arrival_ms'},flush=True)
        if i:runs.append(d)
        if i==1:
            sf.write(RESULTS/(args.name+'.wav'),np.frombuffer(audio,dtype='<i2'),24000,subtype='PCM_16')
        if runs:
            result={'engine':args.name,'model':args.model,'codebooks':32,
                'ttfa':stats([r['ttfa_ms'] for r in runs]),'runs':runs,
                'request_bytes':len(encoded),'seed':1234,
                'scope':'Complete HTTP streaming engine; warm batch one after one warmup, same Chinese prompt and cloned voice; first 3840 bytes of s16le 24 kHz mono PCM (80 ms). Includes loopback upload of the reference data URI on every request; the engine may cache encoded reference tokens. No audio output caching.'}
            (RESULTS/(args.name+'.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
