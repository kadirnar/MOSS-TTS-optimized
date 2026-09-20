"""Bilingual ASR and speaker-vector comparisons for saved cloning outputs."""
import argparse
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from faster_whisper import WhisperModel
from .common import RESULTS
from .validate_quantized_audio import distance


def units(text,language):
    text=unicodedata.normalize('NFKC',text).lower()
    text=''.join(c for c in text if not unicodedata.category(c).startswith('P'))
    return list(re.sub(r'\s+','',text)) if language=='Chinese' else text.split()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tags',nargs='+',required=True)
    p.add_argument('--asr-device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output',default='evaluation')
    args=p.parse_args()
    if not all(c.isalnum() or c=='_' for c in args.output):raise ValueError('Unsafe output name')
    destination=RESULTS/'quality_suite'/(args.output+'.json')
    torch.set_num_threads(4)
    revision=json.loads((RESULTS/'speaker_model_revision.json').read_text())
    feature=Wav2Vec2FeatureExtractor.from_pretrained(revision['path'])
    speaker=WavLMForXVector.from_pretrained(revision['path']).eval().cuda()
    asr=WhisperModel('small',device=args.asr_device,compute_type='int8' if args.asr_device=='cpu' else 'float16',cpu_threads=4,local_files_only=True)
    embeddings={}
    def vector(path):
        if path not in embeddings:
            wave,sr=sf.read(path,dtype='float32')
            if wave.ndim>1:wave=wave.mean(-1)
            tensor=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000)
            inputs=feature(tensor.numpy(),sampling_rate=16000,return_tensors='pt').to('cuda')
            emb=speaker(**inputs).embeddings
            embeddings[path]=torch.nn.functional.normalize(emb,dim=-1).cpu()
        return embeddings[path]
    results={'speaker_model':revision,'asr':f'faster-whisper-small {args.asr_device} '+('INT8' if args.asr_device=='cpu' else 'FP16')+' beam5',
        'scope':'Diagnostic saved-utterance suite; see each manifest for its size and references. CER for Chinese, WER for English; speaker cosine is an uncalibrated proxy, not human listening or a speaker-identity guarantee.',
        'tags':{}}
    for tag in args.tags:
        manifest=json.loads((RESULTS/'quality_suite'/tag/'manifest.json').read_text())
        references={r['voice']:r['reference'] for r in manifest['records']}
        rows=[]
        for record in manifest['records']:
            lang='zh' if record['language']=='Chinese' else 'en'
            segments,_=asr.transcribe(record['audio'],language=lang,beam_size=5)
            transcript=''.join(s.text for s in segments)
            expected=units(record['text'],record['language'])
            error=distance(expected,units(transcript,record['language']))/len(expected)
            v=vector(record['audio'])
            scores={voice:float((v*vector(path)).sum()) for voice,path in references.items()}
            row={**record,'transcript':transcript,'asr_error_rate':error,
                'speaker_cosine':scores[record['voice']],'speaker_reference_scores':scores,
                'speaker_top1_reference':max(scores,key=scores.get)}
            rows.append(row)
            print(tag,record['id'],error,row['speaker_cosine'],row['speaker_top1_reference'],flush=True)
            results['tags'][tag]={'records':rows}
            destination.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
        results['tags'][tag]['summary']={
            'mean_chinese_cer':float(np.mean([r['asr_error_rate'] for r in rows if r['language']=='Chinese'])),
            'mean_english_wer':float(np.mean([r['asr_error_rate'] for r in rows if r['language']=='English'])),
            'mean_speaker_cosine':float(np.mean([r['speaker_cosine'] for r in rows])),
            'speaker_top1_reference_accuracy':float(np.mean([r['speaker_top1_reference']==r['voice'] for r in rows])),
            'truncated_utterances':sum(r['truncated'] for r in rows)}
        destination.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
