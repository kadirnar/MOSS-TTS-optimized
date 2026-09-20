"""Sequential temporary loopback servers for the projection PDL versus attention PDL comparison.

Each foreground server is terminated in a finally block. Existing supervisor
services and their process-local voice caches are not changed.
"""
import argparse
import json
import socket
import subprocess
import sys
import time
import requests
from .common import ROOT,RESULTS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v1');args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe nonempty tag required')
    tag=args.tag
    base=[sys.executable,'-u','-m','optimization.server','--port','18084',
          '--calibration',str(RESULTS/'gptq_v1_g32_d10'),
          '--packing-plan',str(RESULTS/'dp4a_direct_exact_plan.json'),
          '--decode-buckets','--attention-quant','--native-attention','--gateup-quant','--scaled-dp4a','--norm-projection','--audio-head-buckets','--short-scales','--projection-pdl']
    for mode in ('control','pdl'):
        name='http_attention_pdl_'+mode+'_'+tag;metrics=RESULTS/(name+'_stages.jsonl');log=RESULTS/(name+'_server.log')
        assert not metrics.exists() and not log.exists() and not (RESULTS/(name+'.json')).exists(),'Preserve previous measurements'
        with socket.socket() as probe:
            assert probe.connect_ex(('127.0.0.1',18084))!=0,'Temporary port is already occupied'
        args=base+['--metrics-file',str(metrics)]+(['--attention-pdl'] if mode=='pdl' else [])
        with log.open('w') as output:
            process=subprocess.Popen(args,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            try:
                started=time.monotonic()
                while True:
                    if process.poll() is not None:raise RuntimeError('Server failed; inspect '+str(log))
                    if time.monotonic()-started>180:raise TimeoutError('Server initialization exceeded 180 seconds')
                    try:
                        health=requests.get('http://127.0.0.1:18084/health',timeout=1).json()
                        if health.get('ready'):break
                    except (requests.RequestException,ValueError):pass
                    time.sleep(.5)
                assert health['codebooks']==32 and health['scaled_dp4a'] and health['norm_projection'] and health['audio_head_buckets'] and health['short_scales'] and not health['compressed_scales'] and health['projection_pdl'] and health['attention_pdl']==(mode=='pdl')
                (RESULTS/(name+'_health.json')).write_text(json.dumps(health,indent=2)+'\n')
                print('Ready',mode,'pid',process.pid,flush=True)
                with (RESULTS/(name+'_client.log')).open('w') as client:
                    subprocess.run([sys.executable,'-u','-m','optimization.benchmark_http','--url','http://127.0.0.1:18084','--name',name,'--max-new-tokens','400'],cwd=ROOT,stdout=client,stderr=subprocess.STDOUT,check=True,timeout=300)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                print('Stopped',mode,'pid',process.pid,'exit',process.returncode,flush=True)
    a=json.loads((RESULTS/f'http_attention_pdl_control_{tag}.json').read_text());b=json.loads((RESULTS/f'http_attention_pdl_pdl_{tag}.json').read_text())
    assert len(a['runs'])==len(b['runs'])==20
    same=all(x['pcm_sha256']==y['pcm_sha256'] and x['frames']==y['frames'] for x,y in zip(a['runs'],b['runs']))
    assert same,'PCM output changed'
    result={'method':'Sequential temporary servers, qualified projection PDL control then selected attention PDL (QK trigger 1, attention trigger 2, reduction trigger 1; no Q/K norm preload); identical text/reference/seeds 501-520, all 32 codebooks. No concurrent GPU test.','all_20_full_pcm_streams_exact':same,
            'control_ttfa':a['ttfa'],'pdl_ttfa':b['ttfa'],'control_fresh_single_ms':a['fresh_reference_to_pcm_ms'],'pdl_fresh_single_ms':b['fresh_reference_to_pcm_ms'],
            'control_cancellation_statuses':a['cancellation_recovery_statuses'],'pdl_cancellation_statuses':b['cancellation_recovery_statuses'],'temporary_servers_stopped':True}
    (RESULTS/f'http_attention_pdl_comparison_{tag}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
