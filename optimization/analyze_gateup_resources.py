"""Summarize the completed CUDA resource sweep without changing serving defaults."""
import hashlib
import json
from pathlib import Path
import random
import statistics

RESULTS = Path(__file__).resolve().parent / 'results'


def read(name):
    return json.loads((RESULTS / (name + '.json')).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = RESULTS / 'gateup_resources_pass_summary_v1.json'
    if destination.exists():
        raise FileExistsError('Preserve evidence')
    manifest = read('gateup_resource_bundle_v1/manifest')
    assert manifest['complete']
    bundle = RESULTS / 'gateup_resource_bundle_v1'
    for name, record in manifest['binaries'].items():
        assert digest(bundle / (name + '.cubin')) == record['cubin_sha256']
        if 'resources' in record:
            resources = record['resources']
            assert digest(bundle / (name + '.ptx')) == resources['ptx_sha256']
            assert record['metadata']['shared'] == resources['dynamic_shared_bytes']
            assert record['spills'] is None
    result = {
        'codebooks': 32, 'complete': True, 'goal_reached': False,
        'selection': 'Retain the preceding --gateup-compiler implementation unchanged.',
        'candidate_status': 'ir4_r144_dynamic remains benchmark-only; its small timing benefit '
                            'does not justify replacing the qualified path in this pass.',
        'bundle': str(bundle), 'binary_count': len(manifest['binaries']),
        'all_cubin_hashes_verified': True,
        'resource_notes': 'Driver occupancy is a residency limit, not measured occupancy. '
                          'Null Triton spill counts are unknown, not zero; compiler spill bytes '
                          'and driver local/shared allocations are separately recorded.',
        'experiments': {}, 'requests': {}, 'sanitizers': {},
    }
    for tag in ('pilot_v1', 'ring_v1'):
        data = read('gateup_resources_' + tag)
        assert not data['errors']
        assert all(all(row[k] for k in ('outputs_exact', 'poisoned_cache_exact', 'norm_debug_exact'))
                   for row in data['checks'])
        result['experiments'][tag] = {
            'checks': len(data['checks']), 'graphs': len(data['graph_edges']),
            'all_exact': True, 'summary': data['summary'],
        }
    for tag in ('v1', 'repeat_v1', 'final_v1'):
        data = read('gateup_resources_paired_' + tag)
        assert data['complete'] and data['all_pcm_rng_exact']
        assert all(row['pcm_rng_exact'] for row in data['rows'])
        assert all(row['selected_eager_candidate_graph_bits_exact'] and
                   row['all_72_poisoned_caches_rng_exact'] for row in data['checks'])
        modes = tuple(data['ttfa'])
        record = {
            'rounds': len(data['rows']), 'measured_streams': len(data['rows']) * len(modes),
            'full_graph_checks': len(data['checks']), 'all_pcm_rng_exact': True,
            'prior_control_hashes_exact': data['control_prior_pcm_hashes_exact'],
            'ttfa': data['ttfa'], 'median_paired_gain_ms': data['median_paired_gain_ms'],
            'faster_rounds': data['faster_rounds'], 'stages_ms': {},
            'descriptive_bootstrap_median_gain_95pct_ms': {},
        }
        for mode in modes:
            rows = [row['records'][mode] for row in data['rows']]
            assert all(not row['truncated'] for row in rows)
            assert all(len(row['first_audio_graph_ms']) == 1 and
                       row['first_audio_graph_ms'][0]['steps'] == 32 for row in rows)
            record['stages_ms'][mode] = {
                **{k: statistics.median(row[k] for row in rows)
                   for k in ('prepare_ms', 'prefill_ms')},
                'initial_32_llm_ms': statistics.median(row['first_audio_graph_ms'][0]['elapsed_ms'] for row in rows),
                'first_codec_ms': statistics.median(row['codec_ms'][0] for row in rows),
            }
            if mode != 'control':
                differences = [row['gains_ms'][mode] for row in data['rows']]
                rng = random.Random(7329)
                medians = sorted(statistics.median(rng.choices(differences, k=len(differences)))
                                 for _ in range(10000))
                record['descriptive_bootstrap_median_gain_95pct_ms'][mode] = [medians[249], medians[9749]]
        record['bootstrap_caveat'] = ('Within-run resampling of pairs only; serial runtime shifts, '
            'screening/selection, repeated seeds and small samples prevent a general latency guarantee. '
            'Separate runs are not pooled into an inferential estimate.')
        result['requests'][tag] = record
    for tool in ('memcheck', 'racecheck', 'synccheck'):
        data = read('gateup_resources_validation_' + tool + '_v1')
        log = (RESULTS / ('gateup_resources_' + tool + '_v1.log')).read_text()
        assert data['complete'] and all(row['exact'] for row in data['checks'])
        message = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck'
                   else 'ERROR SUMMARY: 0 errors')
        assert message in log
        result['sanitizers'][tool] = {'cases': data['cases'], 'all_exact': True,
                                     'summary': message, 'scope': data['scope']}
    events = read('gateup_resources_profile_final_v1_trace')['traceEvents']
    kernels = [event for event in events if event.get('cat') == 'kernel']
    result['profile'] = {'kernel_events': len(kernels), 'projections': {},
        'scope': 'Post-timing truncated 34-step diagnostic: 33 decode steps and two codec chunks. '
                 'Resident intervals include waits and overlap; not critical-path shares or utilization.'}
    for label, name, grid, registers in (
            ('qkv', '_kernel', [384, 1, 1], 64),
            ('attention_output', '_kernel', [256, 1, 1], 96),
            ('candidate_gateup', '_kernel', [384, 1, 1], 144),
            ('down', '_project', [512, 1, 1], 128)):
        durations = [event['dur'] for event in kernels if event['name'] == name and
                     event['args'].get('grid') == grid and
                     event['args'].get('registers per thread') == registers]
        assert len(durations) == 1188
        result['profile']['projections'][label] = {
            'events': len(durations), 'median_resident_us': statistics.median(durations),
            'sum_resident_ms': sum(durations) / 1000,
        }
    result['production_residual_resources'] = {
        name: {k: record[k] for k in ('registers', 'spills', 'cubin_sha256', 'metadata', 'resources') if k in record}
        for name, record in manifest['binaries'].items() if name.endswith('_add1_debug0')
    }
    result['limitations'] = [
        'All candidates preserve the preceding calibrated G32 path on the tested requests; this is not upstream BF16 equivalence.',
        'No candidate serving flag, new HTTP latency claim, fresh-reference timing claim or new 48-utterance quality claim.',
        'The initial and final processes exhibit common runtime shifts in unchanged stages; absolute medians are not kernel gains.',
        'No evidence establishes a 50 ms lower bound. All 32 acoustic codebooks remain required.',
    ]
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(destination)
    print('streams', sum(r['measured_streams'] for r in result['requests'].values()),
          'full_graph_checks', sum(r['full_graph_checks'] for r in result['requests'].values()))


if __name__ == '__main__':
    main()
