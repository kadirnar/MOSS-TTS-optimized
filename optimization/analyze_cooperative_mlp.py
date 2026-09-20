"""Reconstruct cooperative-MLP rejection evidence without running CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'optimization/results'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', default='cooperative_mlp_pass_summary_v1.json')
    a = p.parse_args()
    output = RESULTS/a.output
    if output.parent != RESULTS or output.exists(): raise ValueError('New result filename required')
    inputs = {}
    def read(name):
        path = RESULTS/name; inputs[name] = digest(path)
        return json.loads(path.read_text())
    screens = {}
    for tag in ('pilot_v1', 'pilot_v2', 'pilot_v3', 'ring_v1', 'cluster_pilot_v1', 'cluster_ring_v1', 'final_ring_v1'):
        d = read(f'cooperative_mlp_{tag}.json'); assert d['complete']
        assert all(r['outputs_exact'] and r['whole_cache_exact'] for r in d['checks']+d['graph_checks'])
        screens[tag] = {'attempted_candidates': len(d['configs'])-1, 'timed_candidates': len(d['summary'])-1,
            'intermediate_cache_cases': len(d['checks']), 'graph_cases': len(d['graph_checks']),
            'errors': list(d['errors']), 'rounds': len(d['rows'])}
        if tag == 'pilot_v1':
            assert len(d['errors']) == 3 and all('CalledProcessError' in e for e in d['errors'].values())
        else:
            assert all('Unsafe cooperative cluster grid refused before launch' in e for e in d['errors'].values())
        if tag == 'final_ring_v1': final = d
    assert len(final['summary']) == 33 and len(final['errors']) == 8
    assert len(final['checks']) == 3564 and len(final['graph_checks']) == 264
    assert all(r['median_paired_gain_us'] < 0 and r['faster_rounds'] == 0 for n, r in final['summary'].items() if n != 'control')
    for name, edges in final['edges'].items():
        assert len(edges) == (5 if name == 'control' else 4) and all(e['type'] == 1 for e in edges)
    binaries = {}
    for name, resources in final['resources'].items():
        assert resources['down_source_sha256'] == 'b458c4a76fd8401a1db78c6a217b9f766f840857e6bb3a6eccb49c758e50acba'
        for key, record in resources.items():
            if not isinstance(record, dict): continue
            build = Path(record['build'])
            assert digest(build/'kernel.cubin') == record['cubin_sha256']
            assert digest(build/'kernel.ptx') == record['ptx_sha256']
            assert record['grid_blocks'] <= record['cooperative_capacity']
            if record['max_active_clusters'] is not None:
                assert record['grid_blocks']//record['cluster_size'] <= record['max_active_clusters']
            binaries[name] = record
    sanitizers = {}
    for tool in ('memcheck', 'racecheck', 'synccheck'):
        d = read(f'cooperative_mlp_validation_{tool}_v1.json')
        log = RESULTS/f'cooperative_mlp_{tool}_v1.log'; inputs[log.name] = digest(log)
        text = log.read_text()
        assert d['complete'] and d['cases'] == 315
        assert all(r['exact'] and r['barrier_exact'] for r in d['checks'])
        banner = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors'
        assert banner in text and 'PASSED 315' in text
        sanitizers[tool] = {'cases': d['cases'], 'configs': d['configs'], 'banner': banner,
            'scope': d['scope'], 'production_qkv': d['production_qkv']}
    profiles = {}
    for name, record in final['profiles'].items():
        path = Path(record['path']); inputs[path.name] = digest(path)
        trace = json.loads(path.read_text()); groups = {}
        kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
        for event in kernels:
            args = event.get('args', {})
            key = (event['name'], str(args.get('grid')), args.get('registers per thread'))
            groups.setdefault(key, []).append(event['dur'])
        profiles[name] = {'kernel_events': len(kernels), 'groups': [
            {'name': k[0], 'grid': k[1], 'registers': k[2], 'count': len(v),
             'median_resident_us': statistics.median(v), 'resident_sum_ms': sum(v)/1000}
            for k, v in groups.items()], 'scope': record['scope']}
        if name == 'control':
            up = sorted((e for e in kernels if e.get('args', {}).get('registers per thread') == 148), key=lambda e: e['ts'])
            down = sorted((e for e in kernels if e['name'] == '_project'), key=lambda e: e['ts'])
            assert len(up) == len(down) == 108
            extent = [max(x['ts']+x['dur'], y['ts']+y['dur'])-min(x['ts'], y['ts']) for x, y in zip(up, down, strict=True)]
            overlap = [max(0, min(x['ts']+x['dur'], y['ts']+y['dur'])-max(x['ts'], y['ts'])) for x, y in zip(up, down, strict=True)]
            profiles[name]['mlp_resident_intervals'] = {'pairs': len(up), 'overlapping_pairs': sum(v > 0 for v in overlap),
                'median_joint_extent_us': statistics.median(extent), 'median_overlap_us': statistics.median(overlap),
                'scope': 'Chronological resident intervals including dependency waits. Not useful-work overlap or utilization.'}
    best = min((n for n in final['summary'] if n != 'control'), key=lambda n: final['summary'][n]['median_us'])
    result = {'complete': True, 'codebooks': 32, 'goal_met': False, 'candidate_selected': None,
        'production_changed': False, 'screens': screens, 'final_summary': final['summary'],
        'best_tested_candidate': best, 'binaries': binaries, 'sanitizers': sanitizers,
        'profiles': profiles, 'input_sha256': inputs,
        'scope': 'Cooperative exact PTX MLP experiment only. No new full-request TTFA, HTTP, codec or audio-quality measurements. '
                 'Selected historical-KV preload preset and its qualified measurements remain unchanged.'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'best_tested': best, 'control': final['summary']['control'], 'candidate': final['summary'][best]}, indent=2))


if __name__ == '__main__': main()
