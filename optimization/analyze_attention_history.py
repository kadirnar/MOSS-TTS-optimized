"""Verify and summarize historical-KV overlap experiments and integration."""
import hashlib
import json
from pathlib import Path
import statistics

RESULTS = Path(__file__).resolve().parent / 'results'


def read(name):
    return json.loads((RESULTS / (name + '.json')).read_text())


def main():
    path = RESULTS / 'attention_history_pass_summary_v1.json'
    if path.exists():
        raise FileExistsError('Preserve evidence')
    result = {'codebooks': 32, 'complete': True, 'goal_reached': False,
        'selected': 'q1_a1_m3_float', 'flag': '--attention-history', 'default_enabled': False,
        'experiments': {}, 'requests': {}, 'sanitizers': {},
        'scope': 'Warm batch-one streaming voice cloning; selected calibrated G32 decode, BF16 prefill, '
                 'FP32 codec and all 32 acoustic codebooks. Existing supervisor services unchanged.'}
    for tag in ('pilot_v2', 'ring_v1'):
        data = read('attention_history_' + tag)
        assert data['complete']
        assert all(r['outputs_exact'] and r['poisoned_cache_exact']
                   for r in data['checks'] + data['graph_checks'])
        result['experiments'][tag] = {
            'configs': data['configs'], 'operator_checks': len(data['checks']),
            'private_graph_checks': len(data['graph_checks']), 'graphs': len(data['edges']),
            'summary': data['summary'], 'method': data['method'],
        }
    for tag in ('v1', 'repeat_v1'):
        data = read('attention_history_paired_' + tag)
        assert data['complete'] and data['all_pcm_rng_exact']
        assert all(row['pcm_rng_exact'] for row in data['rows'])
        assert all(row['selected_eager_candidate_graph_bits_exact'] and
                   row['all_72_poisoned_caches_rng_exact'] for row in data['checks'])
        entry = {'rounds': len(data['rows']), 'measured_streams': len(data['rows'])*len(data['ttfa']),
            'full_graph_checks': len(data['checks']), 'control_prior_pcm_hashes_exact': data['control_prior_pcm_hashes_exact'],
            'ttfa': data['ttfa'], 'median_paired_gain_ms': data['median_paired_gain_ms'],
            'faster_rounds': data['faster_rounds'], 'all_pcm_rng_exact': True, 'stages_ms': {}}
        for mode in data['ttfa']:
            rows = [row['records'][mode] for row in data['rows']]
            assert all(not row['truncated'] and row['first_audio_graph_ms'][0]['steps'] == 32 for row in rows)
            entry['stages_ms'][mode] = {
                'prepare': statistics.median(row['prepare_ms'] for row in rows),
                'prefill': statistics.median(row['prefill_ms'] for row in rows),
                'initial_32_llm': statistics.median(row['first_audio_graph_ms'][0]['elapsed_ms'] for row in rows),
                'first_codec': statistics.median(row['codec_ms'][0] for row in rows),
            }
        result['requests'][tag] = entry
    for tool, tag in (('memcheck', 'memcheck_v2'), ('racecheck', 'racecheck_v1'), ('synccheck', 'synccheck_v1')):
        data = read('attention_history_validation_' + tag)
        assert data['complete'] and data['cases'] == 360 and all(row['exact'] for row in data['checks'])
        log = (RESULTS / ('attention_history_' + tag + '.log')).read_text()
        expected = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck'
                    else 'ERROR SUMMARY: 0 errors')
        assert expected in log
        result['sanitizers'][tool] = {'cases': data['cases'], 'summary': expected, 'scope': data['scope']}
    quality = read('attention_history_quality_exact_v1')
    assert quality['all_exact'] and sum(s['generated_wavs'] for s in quality['suites']) == 48
    for suite in quality['suites']:
        for name, digest in suite['wav_sha256'].items():
            for key in ('candidate', 'control'):
                assert hashlib.sha256((Path(suite[key]) / name).read_bytes()).hexdigest() == digest
    result['quality'] = quality
    result['cli'] = read('attention_history_cli_exact_v1')
    assert result['cli']['validation_dictionary_exact'] and result['cli']['wav_exact']
    http = read('paired_http_history_v1')
    assert http['complete'] and http['server_stopped'] and http['validation_errors_checked']
    assert http['cancellation_recovery_statuses'][-1] == 200
    result['http'] = {'method': http['method'], 'cached': http['cached_summary'], 'fresh': http['fresh_summary'],
        'prior_cached_pcm_exact': http['prior_cached_pcm_exact'],
        'cancellation_recovery_statuses': http['cancellation_recovery_statuses'],
        'server_stopped': http['server_stopped']}
    result['http_stages'] = {}
    for workload in ('cached', 'fresh'):
        rows = http[workload + '_rows']; stages = {}
        assert http[workload + '_summary']['all_pcm_rng_exact']
        for mode in ('control', 'history'):
            records = [row['records'][mode] for row in rows]
            stages[mode] = {
                'engine_ttfa_ms': statistics.median(r['engine']['ttfa_ms'] for r in records),
                'prefill_ms': statistics.median(r['engine']['prefill_ms'] for r in records),
                'initial_32_llm_ms': statistics.median(r['engine']['first_audio_graph_ms'][0]['elapsed_ms'] for r in records),
                'first_codec_ms': statistics.median(r['engine']['codec_ms'][0] for r in records),
            }
            if workload == 'fresh':
                stages[mode]['reference_registration_ms'] = statistics.median(r['reference_registration_ms'] for r in records)
        stages['paired_engine_gain_ms'] = statistics.median(
            row['records']['control']['engine']['ttfa_ms'] - row['records']['history']['engine']['ttfa_ms'] for row in rows)
        if workload == 'fresh':
            stages['paired_reference_difference_ms_unchanged_encoder'] = statistics.median(
                row['records']['control']['reference_registration_ms'] - row['records']['history']['reference_registration_ms'] for row in rows)
        result['http_stages'][workload] = stages
    smoke = read('history_server_smoke_v1')
    assert smoke['complete'] and smoke['server_stopped'] and smoke['prior_selected_pcm_exact']
    result['production_flag_smoke'] = smoke
    codegen = read('attention_history_codegen_v1')
    source = RESULTS.parent / 'attention_history.cu'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == codegen['source_sha256']
    pair_hash = hashlib.sha256(source.read_bytes() + source.with_name('attention_pdl.cu').read_bytes()).hexdigest()
    assert pair_hash[:16] == codegen['build']
    assert hashlib.sha256((RESULTS/'attention_history_build'/codegen['build']/'attention.so').read_bytes()).hexdigest() == codegen['so_sha256']
    result['codegen'] = codegen
    manifest = read('qkv_cluster_bundle_history_v1/manifest')
    assert manifest['complete']
    for name, record in manifest['binaries'].items():
        assert hashlib.sha256((RESULTS/'qkv_cluster_bundle_history_v1'/(name+'.cubin')).read_bytes()).hexdigest() == record['cubin_sha256']
    result['producer_binaries'] = manifest
    events = read('attention_history_profile_repeat_v1_trace')['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    groups = {
        'qkv': [e for e in kernels if e['name'] == '_kernel' and e['args'].get('registers per thread') == 72],
        'attention': [e for e in kernels if 'attention_history<' in e['name']],
        'output': [e for e in kernels if e['name'] == '_kernel' and e['args'].get('registers per thread') == 96],
        'gateup': [e for e in kernels if e['name'] == '_kernel' and e['args'].get('registers per thread') == 148],
        'down': [e for e in kernels if e['name'] == '_project'],
    }
    assert all(len(v) == 1188 for v in groups.values())
    q, a = [sorted(groups[n], key=lambda e: e['ts']) for n in ('qkv', 'attention')]
    overlaps = [max(0, min(x['ts']+x['dur'], y['ts']+y['dur'])-max(x['ts'], y['ts'])) for x, y in zip(q, a, strict=True)]
    result['profile'] = {'kernel_events': len(kernels), 'projections': {
        name: {'events': len(rows), 'median_resident_us': statistics.median(e['dur'] for e in rows),
               'sum_resident_ms': sum(e['dur'] for e in rows)/1000} for name, rows in groups.items()},
        'qkv_attention_overlapping_pairs': sum(v > 0 for v in overlaps),
        'qkv_attention_median_overlap_us': statistics.median(overlaps),
        'qkv_attention_sum_overlap_ms': sum(overlaps)/1000,
        'scope': 'Post-timing truncated 34-step request; 33 decode steps and two codec chunks. '
                 'Paired chronological QKV/attention resident intervals overlap, including waits. '
                 'These are not useful-work overlap, critical-path shares, utilization or a lower bound.'}
    result['failed_or_unmeasured_trials'] = [
        'First __ldg build moved historical KV loads after the wait; corrected to explicit volatile global-vector PTX. No speed claim uses the first build.',
        'Pilot v1 stopped at the binary guard: an older producer bundle predates the signed-zero fix. Fresh early-trigger cubins were exported from current source; control stays v6.',
        'Memcheck v1 exited on a host validator None-residual bug at layer zero. Fixed validator; only completed memcheck v2 counts as sanitizer evidence.',
    ]
    result['remaining'] = 'Initial LLM generation remains about 58.6 ms before prefill/codec/host work; 50 ms is not reached. No proof justifies reducing codebooks.'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
