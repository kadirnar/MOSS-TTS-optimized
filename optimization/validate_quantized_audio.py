"""Small intelligibility smoke test only; not a voice-cloning quality gate."""
import json
import unicodedata

import torch
from faster_whisper import WhisperModel
from .common import RESULTS


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).lower()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def main():
    reference=torch.load(RESULTS/'fixture.pt',weights_only=True)['text']
    model=WhisperModel('small',device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    results={'reference':reference,'asr':'Systran/faster-whisper-small, CPU INT8, beam_size=5, language=zh',
             'scope':'One Chinese sentence and one reference voice. No speaker-similarity test or corpus-level quality claim.', 'samples':{}}
    names=['streaming_optimized']+[p.stem for p in sorted(RESULTS.glob('all32_*.wav'))]
    for name in names:
        path=RESULTS/(name+'.wav')
        if not path.exists():
            continue
        segments,_=model.transcribe(str(path),language='zh',beam_size=5)
        actual=''.join(s.text for s in segments)
        results['samples'][name]={'transcript':actual,'normalized_character_error_rate':distance(normalize(reference),normalize(actual))/len(normalize(reference))}
        print(name,results['samples'][name],flush=True)
    (RESULTS/'all32_asr_smoke.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    main()
