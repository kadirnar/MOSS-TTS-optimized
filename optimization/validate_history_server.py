"""Smoke-check the production server's opt-in historical-KV path over loopback HTTP."""
import argparse
import base64
import hashlib
import json
import socket
import subprocess
import sys
import time
import requests
from .common import ROOT, RESULTS


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe tag required')
    path = RESULTS/f'history_server_smoke_{args.tag}.json'
    log = RESULTS/f'history_server_smoke_{args.tag}.log'
    if path.exists() or log.exists():
        raise FileExistsError('Preserve evidence')
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', 18084)) != 0
    command = [sys.executable, '-m', 'optimization.server', '--port', '18084',
        '--calibration', str(RESULTS/'gptq_v1_g32_d10'),
        '--packing-plan', str(RESULTS/'dp4a_direct_exact_plan.json'),
        '--decode-buckets', '--attention-quant', '--native-attention', '--gateup-quant',
        '--scaled-dp4a', '--norm-projection', '--audio-head-buckets', '--short-scales',
        '--projection-pdl', '--attention-pdl', '--codec-clock', '--first-audio-graph',
        '--bulk-prefetch', '--prefill-qkv', '--prefill-pointwise', '--output-weight-prefetch',
        '--qkv-cluster', '--down-tile8', '--gateup-compiler', '--attention-history']
    result = {'complete': False, 'codebooks': 32, 'command': command,
              'scope': 'Actual production CLI flag, fresh reference registration and one complete HTTP stream; not a timing benchmark.'}
    session = requests.Session(); url = 'http://127.0.0.1:18084'
    with log.open('w') as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        try:
            start = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Production server failed; inspect '+str(log))
                if time.monotonic()-start > 180:
                    raise TimeoutError('Server initialization timed out')
                try:
                    health = session.get(url+'/health', timeout=1).json()
                    if health.get('ready'):
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(.5)
            assert health['codebooks'] == 32 and health['gateup_compiler'] and health['down_tile8'] and health['attention_history']
            result['health'] = health
            encoded = base64.b64encode((ROOT/'assets/audio/reference_zh.wav').read_bytes()).decode()
            response = session.post(url+'/v1/voices', json={'wav_base64': encoded}, timeout=30)
            response.raise_for_status(); voice = response.json()['voice']
            response = session.post(url+'/v1/audio/speech', json={
                'input': '你好，这是一段用于测试流式语音合成速度的句子。',
                'voice': voice, 'seed': 501, 'max_new_tokens': 400}, stream=True, timeout=30)
            response.raise_for_status()
            iterator = response.iter_content(chunk_size=3840); first = next(iterator)
            assert len(first) == 3840
            pcm = first+b''.join(iterator); response.close()
            digest = hashlib.sha256(pcm).hexdigest()
            prior = json.loads((RESULTS/'http_down_tile_down8_v1.json').read_text())['runs'][0]
            assert digest == prior['pcm_sha256'] and len(pcm)//3840 == prior['frames']
            result.update(complete=True, pcm_sha256=digest, frames=len(pcm)//3840,
                first_chunk_bytes=len(first), prior_selected_pcm_exact=True, registered_voice_used=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
            result.update(server_stopped=True, server_exit=process.returncode)
            path.write_text(json.dumps(result, indent=2)+'\n')
    print('PASSED', result['complete'], flush=True)


if __name__ == '__main__':
    main()
