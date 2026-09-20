"""Loopback HTTP TTFA, voice registration, cancellation and recovery checks."""
import base64
import argparse
import json
import hashlib
import time
import requests
from .common import ROOT,RESULTS,stats,save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:18080')
    parser.add_argument('--name',default='http')
    parser.add_argument('--max-new-tokens',type=int,default=160)
    args=parser.parse_args()
    if not all(c.isalnum() or c=='_' for c in args.name):raise ValueError('Unsafe name')
    url=args.url
    session=requests.Session()
    assert session.get(url+'/health',timeout=3).json()['ready']
    encoded=base64.b64encode((ROOT/'assets/audio/reference_zh.wav').read_bytes()).decode()
    registration_start=time.perf_counter()
    registration=session.post(url+'/v1/voices',json={'wav_base64':encoded},timeout=30)
    registration.raise_for_status()
    registration_http_ms=(time.perf_counter()-registration_start)*1000
    voice=registration.json()['voice']
    # Measure a contiguous fresh-reference path before validation requests.
    fresh=session.post(url+'/v1/audio/speech',json={'input':'你好，这是一段用于测试流式语音合成速度的句子。',
        'voice':voice,'seed':499,'max_new_tokens':args.max_new_tokens},stream=True,timeout=30)
    fresh.raise_for_status()
    fresh_iterator=fresh.iter_content(chunk_size=3840)
    first=next(fresh_iterator)
    assert len(first)==3840
    fresh_reference_to_pcm_ms=(time.perf_counter()-registration_start)*1000
    for _ in fresh_iterator:pass
    fresh.close()
    assert session.post(url+'/v1/audio/speech',json={'input':'Hello','voice':'missing'},timeout=5).status_code==404
    assert session.post(url+'/v1/audio/speech',json={'input':'Hello','max_new_tokens':2},timeout=5).status_code==422
    assert session.post(url+'/v1/audio/speech',json={'input':'语'*800,'voice':voice,'max_new_tokens':700},timeout=10).status_code==400
    rows=[]
    for i in range(21):
        body={'input':'你好，这是一段用于测试流式语音合成速度的句子。','voice':voice,'seed':500+i,'max_new_tokens':args.max_new_tokens}
        start=time.perf_counter()
        response=session.post(url+'/v1/audio/speech',json=body,stream=True,timeout=30)
        response.raise_for_status()
        iterator=response.iter_content(chunk_size=3840)
        first=next(iterator)
        ttfa=(time.perf_counter()-start)*1000
        assert len(first)==3840
        remaining=b''.join(iterator)
        assert len(remaining)%3840==0
        if i:rows.append({'ttfa_ms':ttfa,'total_ms':(time.perf_counter()-start)*1000,'frames':(len(first)+len(remaining))//3840,
            'pcm_sha256':hashlib.sha256(first+remaining).hexdigest()})
    # Terminate a live response and require the next request to recover promptly.
    r=session.post(url+'/v1/audio/speech',json={'input':'这是一段用于测试取消功能的长语音。'*8,'voice':voice,'max_new_tokens':400},stream=True,timeout=30)
    r.raise_for_status();next(r.iter_content(chunk_size=3840));r.close()
    start=time.perf_counter();statuses=[]
    while time.perf_counter()-start<3:
        r=session.post(url+'/v1/audio/speech',json={'input':'取消之后继续生成。','voice':voice,'max_new_tokens':96},stream=True,timeout=30)
        statuses.append(r.status_code)
        if r.status_code==200:
            recovery_iterator=r.iter_content(chunk_size=3840)
            assert len(next(recovery_iterator))==3840
            for _ in recovery_iterator:pass
            r.close();break
        assert r.status_code==429
        r.close();time.sleep(.05)
    else:raise AssertionError('Server did not recover after cancellation')
    save_json(args.name+'.json',{'ttfa':stats([r['ttfa_ms'] for r in rows]),'runs':rows,
        'voice_registration':registration.json(),'cancellation_recovery_statuses':statuses,
        'max_new_tokens':args.max_new_tokens,'voice_registration_http_ms':registration_http_ms,
        'fresh_reference_to_pcm_ms':fresh_reference_to_pcm_ms,
        'fresh_reference_definition':'One continuous loopback HTTP voice registration plus following synthesis through first 3840-byte PCM chunk; WAV/base64 prepared before timing. Includes server waveform parsing, reference encoding, two HTTP requests and generation.',
        'validation_errors_checked':True,'definition':'Warm loopback HTTP request to first complete 3840-byte s16le PCM chunk, cached voice; 20 measured complete requests, one warmup. Includes HTTP overhead.'})

if __name__=='__main__':main()
