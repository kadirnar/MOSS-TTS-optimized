"""Rescore stored ASR text; retain raw scores and disclose normalization."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import cn2an
from opencc import OpenCC
from whisper_normalizer.english import EnglishTextNormalizer
from .quality_evaluate import units
from .validate_quantized_audio import distance

folder=Path('optimization/results/quality_suite')
parser=argparse.ArgumentParser()
parser.add_argument('--input',default='evaluation')
args=parser.parse_args()
if not all(c.isalnum() or c=='_' for c in args.input):raise ValueError('Unsafe input name')
data=json.loads((folder/(args.input+'.json')).read_text())
data['normalization']={
    'Chinese':'OpenCC t2s, cn2an.transform cn2an, then Unicode NFKC, punctuation/whitespace removal. Homophones remain errors.',
    'English':'EnglishTextNormalizer from whisper-normalizer, then word tokenization.',
    'versions':{name:importlib.metadata.version(name) for name in ['opencc-python-reimplemented','cn2an','whisper-normalizer']},
    'raw_metrics_preserved':True,
    'transcriptions_rerun':False}
chinese=OpenCC('t2s')
english=EnglishTextNormalizer()
def normalized(text,language):
    return units(cn2an.transform(chinese.convert(text),'cn2an'),language) if language=='Chinese' else english(text).split()
for tag,value in data['tags'].items():
    for row in value['records']:
        ref=normalized(row['text'],row['language'])
        actual=normalized(row['transcript'],row['language'])
        row['normalized_reference_units']=ref
        row['normalized_transcript_units']=actual
        row['normalized_asr_error_rate']=distance(ref,actual)/len(ref)
    value['normalized_summary']={
        'mean_chinese_cer':float(np.mean([r['normalized_asr_error_rate'] for r in value['records'] if r['language']=='Chinese'])),
        'mean_english_wer':float(np.mean([r['normalized_asr_error_rate'] for r in value['records'] if r['language']=='English']))}
    print(tag,value['normalized_summary'],value['summary']['mean_speaker_cosine'])
(folder/(args.input+'_normalized.json')).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
