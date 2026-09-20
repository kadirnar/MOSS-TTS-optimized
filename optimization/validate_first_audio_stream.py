"""Compare streaming control flow with the archived pre-change implementation."""
import argparse
import hashlib
import json
import tarfile
import types

import torch

from .common import RESULTS
from .benchmark_first_audio_graph import build_engine
from .first_audio_graph import enable


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'first_audio_stream_validation_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    archive=RESULTS/'projection_resources_sources.tar.gz'
    manifest=json.loads((RESULTS/'projection_resources_source_hashes.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
    with tarfile.open(archive) as source:
        original=source.extractfile('optimization/streaming.py').read()
    assert hashlib.sha256(original).hexdigest()==manifest['files']['optimization/streaming.py']
    namespace={'__name__':'optimization._archived_streaming','__package__':'optimization'}
    exec(compile(original,'archived:optimization/streaming.py','exec'),namespace)
    old_stream=namespace['StreamingTTS']._stream
    engine,_=build_engine();config=enable(engine);graphs=engine._first_audio_graphs
    new_stream=engine._stream
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    cases=[('Chinese',fixture['text']),('English','Please confirm the order number and check the delivery address.')]
    rows=[]
    for language,text in cases:
        for budget in (33,34,40,400):
            records={};rngs={}
            for mode in ('archived','control','graph'):
                engine._stream=types.MethodType(old_stream,engine) if mode=='archived' else new_stream
                engine._first_audio_graphs=graphs if mode=='graph' else None
                torch.manual_seed(13000+budget)
                chunks=list(engine.stream(text,fixture['reference'],language=language,max_new_tokens=budget))
                assert chunks and not engine._busy
                pcm=torch.cat([c.pcm for c in chunks]);assert torch.isfinite(pcm).all()
                metrics=engine.last_metrics
                records[mode]={'frames':metrics['frames'],'truncated':metrics['truncated'],
                    'prompt_tokens':metrics['prompt_tokens'],'steps':len(metrics['step_ms']),
                    'pcm_sha256':hashlib.sha256(pcm.numpy().tobytes()).hexdigest()}
                rngs[mode]=torch.cuda.get_rng_state()
                assert [c.frame for c in chunks]==list(range(len(chunks)))
                if mode=='graph':assert len(metrics['first_audio_graph_ms'])==1
            assert records['archived']==records['control']==records['graph'],(language,budget,records)
            assert all(torch.equal(rngs['archived'],rngs[m]) for m in ('control','graph'))
            rows.append({'language':language,'budget':budget,'exact':True,'rng_exact':True,'record':records['graph']})
            print('MATCH',language,budget,records['graph']['frames'],flush=True)
    engine._stream=new_stream
    cancellation=[]
    for mode in ('control','graph'):
        engine._first_audio_graphs=graphs if mode=='graph' else None
        for invalid in (32,1024):
            try:list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=invalid))
            except ValueError:pass
            else:raise AssertionError('Invalid budget accepted')
            assert not engine._busy
        torch.manual_seed(14000)
        stream=engine.stream(fixture['text'],fixture['reference'])
        first=next(stream);saved=first.pcm.clone();assert engine._busy
        try:next(engine.stream(fixture['text'],fixture['reference']))
        except RuntimeError:pass
        else:raise AssertionError('Overlapping stream accepted')
        assert engine._busy
        stream.close();assert not engine._busy
        torch.manual_seed(14000)
        recovered=list(engine.stream(fixture['text'],fixture['reference']))
        assert torch.equal(saved,recovered[0].pcm) and torch.equal(saved,first.pcm) and not engine._busy
        assert not engine.last_metrics['truncated']
        cancellation.append({'mode':mode,'invalid_budgets_rejected':[32,1024],
            'busy_guard_preserves_owner':True,'close_releases_owner':True,
            'recovery_first_pcm_exact':True,'prior_pcm_storage_stable':True,'recovery_complete':True})
    result={'codebooks':32,'all_exact':True,'cases':rows,'cancellation':cancellation,
        'archive_sha256':manifest['archive_sha256'],'configuration':config,
        'scope':'Archived pre-change stream loop versus current disabled/enabled graph on one selected engine. Two languages use the same cached reference; this is control-flow coverage, not an additional voice-quality suite. All budgets compare complete PCM, frame count, truncation, prompt/step counts and final RNG state.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(rows),'three-path cases',flush=True)
    engine.codec.close()


if __name__=='__main__':main()
