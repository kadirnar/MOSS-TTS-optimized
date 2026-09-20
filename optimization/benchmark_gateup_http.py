"""Balanced HTTP requests against two graph variants in one temporary process."""
import argparse
import base64
import hashlib
import json
import socket
import statistics
import subprocess
import sys
import time

import requests
from .common import ROOT, RESULTS, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--fresh-rounds', type=int, default=10)
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag) or min(args.rounds, args.fresh_rounds) < 3:
        raise ValueError('Safe tag and at least three rounds required')
    if 2*(args.fresh_rounds+1)+2 > 64:
        raise ValueError('Fresh-reference requests must fit the unchanged voice-cache capacity')
    prefix = 'paired_http_'+args.tag
    path = RESULTS/(prefix+'.json'); metrics = RESULTS/(prefix+'_stages.jsonl')
    log = RESULTS/(prefix+'_server.log')
    if any(p.exists() for p in (path, metrics, log)):
        raise FileExistsError('Preserve evidence')
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', 18084)) != 0
    command = [sys.executable, '-u', '-m', 'optimization.paired_gateup_http_server', '--port', '18084',
        '--metrics-file', str(metrics), '--calibration', str(RESULTS/'gptq_v1_g32_d10'),
        '--packing-plan', str(RESULTS/'dp4a_direct_exact_plan.json'),
        '--decode-buckets', '--attention-quant', '--native-attention', '--gateup-quant',
        '--scaled-dp4a', '--norm-projection', '--audio-head-buckets', '--short-scales',
        '--projection-pdl', '--attention-pdl', '--codec-clock', '--first-audio-graph',
        '--bulk-prefetch', '--prefill-qkv', '--prefill-pointwise', '--output-weight-prefetch', '--qkv-cluster', '--down-tile8']
    result = {'codebooks': 32, 'complete': False, 'cached_rows': [], 'fresh_rows': [],
        'method': 'Single temporary loopback process, one shared model and FP32 codec, two separately captured graph sets. '
                  'Selected eight-row G32 control versus exact one-CTA Triton 3.8 gate/up. '
                  'Balanced AB/BA complete requests with identical text/reference/seed. '
                  'Cached TTFA includes HTTP through first full 3840-byte PCM chunk. Fresh TTFA starts before '
                  'reference-registration HTTP and includes following synthesis; WAV/base64 prepared beforehand. '
                  'Fresh references are re-encoded, generated audio is never cached. One excluded warmup pair per workload.'}
    def save():
        path.write_text(json.dumps(result, indent=2)+'\n')
    encoded = base64.b64encode((ROOT/'assets/audio/reference_zh.wav').read_bytes()).decode()
    text = '你好，这是一段用于测试流式语音合成速度的句子。'
    session = requests.Session(); url = 'http://127.0.0.1:18084'
    with log.open('w') as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Benchmark server failed; inspect '+str(log))
                if time.monotonic()-started > 240:
                    raise TimeoutError('Benchmark initialization exceeded 240 seconds')
                try:
                    health = session.get(url+'/health', timeout=1).json()
                    if health.get('ready'):
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(.5)
            assert health['benchmark_only'] and health['variants'] == ['control', 'gateup38'] and health['codebooks'] == 32
            result['health'] = health; result['server_pid'] = process.pid; save()
            print('READY', process.pid, flush=True)
            def register():
                start = time.perf_counter()
                response = session.post(url+'/v1/voices', json={'wav_base64': encoded}, timeout=30)
                response.raise_for_status()
                return response.json()['voice'], (time.perf_counter()-start)*1000
            voice, registration = register(); result['initial_registration_ms'] = registration
            for workload, rounds, seed_base in (('cached', args.rounds, 501), ('fresh', args.fresh_rounds, 1501)):
                for repeat in range(-1, rounds):
                    order = ['control', 'gateup38'] if repeat % 2 == 0 else ['gateup38', 'control']
                    records = {}
                    for variant in order:
                        start = time.perf_counter(); reference_ms = None
                        current_voice = voice
                        if workload == 'fresh':
                            current_voice, reference_ms = register()
                        ident = f'{workload}_{repeat}_{variant}'
                        response = session.post(url+'/v1/audio/speech', json={
                            'input': text, 'voice': current_voice, 'seed': seed_base+repeat,
                            'max_new_tokens': 400, 'variant': variant, 'benchmark_id': ident}, stream=True, timeout=30)
                        response.raise_for_status()
                        assert response.headers['X-Benchmark-Variant'] == variant
                        iterator = response.iter_content(chunk_size=3840); first = next(iterator)
                        elapsed = (time.perf_counter()-start)*1000
                        assert len(first) == 3840
                        pcm = first+b''.join(iterator); response.close()
                        assert len(pcm) % 3840 == 0
                        records[variant] = {'benchmark_id': ident, 'ttfa_ms': elapsed,
                            'reference_registration_ms': reference_ms, 'frames': len(pcm)//3840,
                            'pcm_sha256': hashlib.sha256(pcm).hexdigest()}
                    assert records['control']['pcm_sha256'] == records['gateup38']['pcm_sha256']
                    row = {'round': repeat, 'seed': seed_base+repeat, 'order': order,
                        'records': records, 'gain_ms': records['control']['ttfa_ms']-records['gateup38']['ttfa_ms']}
                    if repeat >= 0:
                        result[workload+'_rows'].append(row)
                    save(); print('ROUND', workload, repeat, {n:r['ttfa_ms'] for n,r in records.items()}, flush=True)
            assert session.post(url+'/v1/audio/speech', json={'input':'Hello','variant':'missing'}, timeout=5).status_code == 422
            assert session.post(url+'/v1/audio/speech', json={'input':'Hello','voice':'missing','variant':'gateup38'}, timeout=5).status_code == 404
            response = session.post(url+'/v1/audio/speech', json={'input':'这是一段用于测试取消功能的长语音。'*8,
                'voice':voice,'max_new_tokens':400,'variant':'control','benchmark_id':'cancel'}, stream=True, timeout=30)
            response.raise_for_status(); next(response.iter_content(chunk_size=3840)); response.close()
            statuses = []; started = time.monotonic()
            while time.monotonic()-started < 3:
                response = session.post(url+'/v1/audio/speech', json={'input':'取消之后继续生成。',
                    'voice':voice,'max_new_tokens':96,'variant':'gateup38','benchmark_id':'recovery'}, stream=True, timeout=30)
                statuses.append(response.status_code)
                if response.status_code == 200:
                    iterator = response.iter_content(chunk_size=3840); assert len(next(iterator)) == 3840
                    for _ in iterator: pass
                    response.close(); break
                assert response.status_code == 429; response.close(); time.sleep(.05)
            else:
                raise AssertionError('Cross-variant cancellation recovery failed')
            result['validation_errors_checked'] = True
            result['cancellation_recovery_statuses'] = statuses
            save()
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
            result['server_stopped'] = True; result['server_exit'] = process.returncode; save()
            print('STOPPED', process.pid, process.returncode, flush=True)
    stages = [json.loads(s) for s in metrics.read_text().splitlines()]
    by_id = {r['benchmark_id']: r for r in stages if r['completed']}
    for workload in ('cached', 'fresh'):
        rows = result[workload+'_rows']
        for row in rows:
            for variant, record in row['records'].items():
                stage = by_id[record['benchmark_id']]
                assert stage['variant'] == variant and stage['engine']['frames'] == record['frames']
                assert not stage['engine']['truncated']
                record['engine'] = stage['engine']; record['rng_sha256'] = stage['rng_sha256']
            assert row['records']['control']['rng_sha256'] == row['records']['gateup38']['rng_sha256']
        result[workload+'_summary'] = {'ttfa': {n:stats([r['records'][n]['ttfa_ms'] for r in rows]) for n in ('control','gateup38')},
            'median_paired_gain_ms': statistics.median(r['gain_ms'] for r in rows),
            'faster_pairs': sum(r['gain_ms'] > 0 for r in rows), 'all_pcm_rng_exact': True}
    prior = json.loads((RESULTS/'http_down_tile_down8_v1.json').read_text())
    assert all(row['records']['control']['pcm_sha256'] == old['pcm_sha256']
               for row,old in zip(result['cached_rows'], prior['runs']))
    result['prior_cached_pcm_exact'] = min(len(result['cached_rows']),len(prior['runs']))
    result['complete'] = True; save()
    print('SUMMARY', {n:result[n+'_summary'] for n in ('cached','fresh')}, flush=True)


if __name__ == '__main__':
    main()
