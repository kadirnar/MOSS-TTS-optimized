"""Down-projection register preloads with the selected clustered QKV consumer."""
import argparse
import hashlib
import json
import statistics
import traceback

import torch
from .common import RESULTS
from .benchmark_projection_pdl import load_layer as load_mlp, graph_edges
from .benchmark_attention_pdl import load_layer as load_attention
from .bulk_prefetch import configured
from .dp4a_norm_projection import SELECTED
from .short_scales import PLAN
from .qkv_cluster_binary import load_bundle
from .tune_weight_reads import measure


def configs():
    choices = {'control': None}
    for rows in (2, 4, 8):
        for warps in (2, 4, 8):
            for prefetch in (1, 2, 3):
                choices[f'r{rows}_w{warps}_p{prefetch}'] = {
                    'rows': rows, 'warps': warps, 'prefetch': prefetch}
    for rows, warps in ((16, 4), (16, 8)):
        choices[f'r{rows}_w{warps}_p3'] = {
            'rows': rows, 'warps': warps, 'prefetch': 3}
    return choices


def load_layer(layer):
    entry = load_mlp(layer)
    following = load_attention((layer + 1) % 36)
    entry['attention'] = following['attention']
    entry['cache_source'] = following['cache_source']
    return entry


class Chain:
    def __init__(self, options, cluster_options, cluster_launch):
        self.tile = {**PLAN['down'], 'prefetch': 1, **(options or {})}
        self.up = configured('norm', divisor=16)
        self.down = configured('projection', divisor=16)
        self.cluster_options = cluster_options
        self.cluster_launch = cluster_launch

    def __call__(self, entry, index=0, debug=False, audit=False):
        raw = entry['raw']; nd = entry['next']
        weights = entry['weights']; d = entry['attention']
        up = self.up(raw['x'][index:index+1],
            raw['residual'][index:index+1] if raw['has_residual'] else None,
            raw['weight'], raw['eps'], *weights['up'], fused=True,
            **SELECTED['up'], trigger_mode=1)
        summed, hidden, quantized = up
        down, kernel = self.down(hidden, *weights['down'],
            prequantized=quantized, **self.tile, trigger_mode=3, return_kernel=True)
        following = self.cluster_launch(down, summed, nd['weight'], nd['eps'],
            *weights['qkv'], d['qw'], d['kw'], d['cos'], d['sin'],
            d['k'], d['v'], d['position'], d['eps'],
            **self.cluster_options, debug=debug)
        result = up, down, following
        return (result, kernel) if audit else result


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(
            a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
    return all(exact(x, y) for x, y in zip(a, b, strict=True))


def poison(entry):
    d = entry['attention']; pos = int(d['position'])
    for name in ('k', 'v'):
        d[name][:, :, pos, :].fill_(float('nan'))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--layers', type=int, choices=(1, 36), default=36)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--configs', nargs='+', choices=tuple(configs()))
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag) or args.rounds < 2:
        raise ValueError('Safe tag and two rounds required')
    path = RESULTS/f'down_preload_{args.tag}.json'
    if path.exists():
        raise FileExistsError('Preserve evidence')
    torch.set_num_threads(4)
    ring = [load_layer(i) for i in (range(36) if args.layers == 36 else (17,))]
    selected, launchers = load_bundle(RESULTS/'qkv_cluster_bundle_v6')
    cluster_options = selected['c8_t2_exact']; cluster_launch = launchers['c8_t2_exact']
    control = Chain(None, cluster_options, cluster_launch)
    choices = configs()
    if args.configs:
        choices = {n: c for n, c in choices.items() if n == 'control' or n in args.configs}
    result = {'codebooks': 32, 'layers': args.layers, 'configs': choices,
        'checks': [], 'resources': {}, 'graph_edges': {}, 'errors': {}, 'rows': [],
        'method': 'Selected G32 norm/gate/up -> down -> following clustered QKV/head preparation. '
                  'Immutable down weights/scales preload before the original PDL wait; row/warp tiles vary. '
                  'Actual layer weights and three saved inputs, nearest frozen attention metadata; '
                  'layer 35 wraps to layer 0 synthetically. Private-stream outputs and entire poisoned caches '
                  'checked bytewise. Balanced rotated/reversed CUDA event rings; not TTFA.'}
    def save():
        path.write_text(json.dumps(result, indent=2)+'\n')
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    chains = {}
    with torch.cuda.stream(stream):
        for name, options in choices.items():
            try:
                candidate = Chain(options, cluster_options, cluster_launch)
                for entry in ring:
                    for index in (0, 10, 31):
                        poison(entry); expected = control(entry, index, debug=True)
                        cache = [entry['attention'][n].clone() for n in ('k', 'v')]
                        poison(entry); actual = candidate(entry, index, debug=True)
                        output_exact = exact(actual, expected)
                        cache_exact = exact([entry['attention'][n] for n in ('k', 'v')], cache)
                        row = {'config': name, 'layer': entry['layer'], 'input': index,
                               'outputs_exact': output_exact, 'poisoned_cache_exact': cache_exact}
                        result['checks'].append(row)
                        assert output_exact and cache_exact, row
                _, kernel = candidate(ring[0], audit=True)
                result['resources'][name] = {'registers': kernel.n_regs,
                    'spills': kernel.n_spills, 'shared_bytes': kernel.metadata.shared,
                    'cubin_sha256': hashlib.sha256(kernel.asm['cubin']).hexdigest()}
                for _ in range(2):
                    candidate(ring[0])
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, stream=stream):
                    actual = candidate(ring[0])
                edges = graph_edges(graph)
                assert len(edges) == 2 and all(e['type'] == 1 for e in edges), edges
                result['graph_edges'][name] = edges
                poison(ring[0]); graph.replay()
                cache = [ring[0]['attention'][n].clone() for n in ('k', 'v')]
                poison(ring[0]); expected = control(ring[0])
                assert exact(actual, expected), name
                assert exact([ring[0]['attention'][n] for n in ('k', 'v')], cache), name
                chains[name] = candidate
                print('CHECKED', name, result['resources'][name], flush=True)
            except Exception as error:
                result['errors'][name] = traceback.format_exc()
                print('ERROR', name, repr(error), flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    assert 'control' in chains, result['errors'].get('control')
    for repeat in range(args.rounds):
        order = list(chains); offset = (repeat//2) % len(order)
        order = order[offset:]+order[:offset]
        if repeat % 2:
            order.reverse()
        times = {n: measure(chains[n], ring) for n in order}
        result['rows'].append({'round': repeat, 'order': order, 'us': times})
        save(); print('ROUND', repeat, times, flush=True)
    result['summary'] = {n: {
        'median_us': statistics.median(r['us'][n] for r in result['rows']),
        'median_paired_gain_us': statistics.median(r['us']['control']-r['us'][n] for r in result['rows']),
        'faster_rounds': sum(r['us']['control'] > r['us'][n] for r in result['rows'])}
        for n in chains}
    save(); print('SUMMARY', result['summary'], flush=True)


if __name__ == '__main__':
    main()
