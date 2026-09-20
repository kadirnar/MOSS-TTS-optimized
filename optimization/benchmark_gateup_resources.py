"""CUDA 13 gate/up register allocation against the selected Triton 3.8 cubins."""
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
from .gateup_cluster_binary import load_bundle as load_up_bundle
from .tune_weight_reads import measure



def load_layer(layer):
    entry = load_mlp(layer)
    following = load_attention((layer + 1) % 36)
    entry['attention'] = following['attention']
    entry['cache_source'] = following['cache_source']
    return entry


class Chain:
    def __init__(self, options, cluster_options, cluster_launch):
        self.tile = {**PLAN['down'], 'rows': 8, 'warps': 4, 'prefetch': 1, 'trigger_mode': 3, **(options or {}).get('down', {})}
        self.up_options = {**SELECTED['up'], 'trigger_mode': 1, **(options or {}).get('up', {})}
        self.up = configured('norm', divisor=16)
        self.down = configured('projection', divisor=16)
        self.cluster_options = cluster_options
        self.cluster_launch = cluster_launch

    def __call__(self, entry, index=0, debug=False, audit=False):
        raw = entry['raw']; nd = entry['next']
        weights = entry['weights']; d = entry['attention']
        up, up_kernel = self.up(raw['x'][index:index+1],
            raw['residual'][index:index+1] if raw['has_residual'] else None,
            raw['weight'], raw['eps'], *weights['up'], fused=True,
            **self.up_options, return_kernel=True)
        summed, hidden, quantized = up
        down, kernel = self.down(hidden, *weights['down'],
            prequantized=quantized, **self.tile, return_kernel=True)
        following = self.cluster_launch(down, summed, nd['weight'], nd['eps'],
            *weights['qkv'], d['qw'], d['kw'], d['cos'], d['sin'],
            d['k'], d['v'], d['position'], d['eps'],
            **self.cluster_options, debug=debug)
        result = up, down, following
        return (result, {'up': up_kernel, 'down': kernel}) if audit else result


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(
            a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
    return all(exact(x, y) for x, y in zip(a, b, strict=True))


def poison(entry):
    d = entry['attention']; pos = int(d['position'])
    for name in ('k', 'v'):
        d[name][:, :, pos, :].fill_(float('nan'))


def make_up(launcher, options):
    def fn(*args, **kwargs):
        assert kwargs.pop('fused') is True
        assert kwargs.pop('rows') == 32
        kwargs.pop('integer_groups'); kwargs.pop('integer_rows'); kwargs.pop('trigger_mode')
        return launcher(*args, **options, **kwargs)
    return fn


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--layers', type=int, choices=(1, 36), default=36)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--bundle', required=True)
    p.add_argument('--configs', nargs='+')
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag) or args.rounds < 2:
        raise ValueError('Safe tag and two rounds required')
    path = RESULTS/f'gateup_resources_{args.tag}.json'
    if path.exists():
        raise FileExistsError('Preserve evidence')
    torch.set_num_threads(4)
    ring = [load_layer(i) for i in (range(36) if args.layers == 36 else (17,))]
    selected, launchers = load_bundle(RESULTS/'qkv_cluster_bundle_v6')
    cluster_options = selected['c8_t2_exact']; cluster_launch = launchers['c8_t2_exact']
    control = Chain(None, cluster_options, cluster_launch)
    control_choices, control_launchers = load_up_bundle(RESULTS/'gateup_exact_bundle_v1')
    control.up = make_up(control_launchers['c1_ig1ir2'], control_choices['c1_ig1ir2'])
    choices, up_launchers = load_up_bundle(RESULTS/args.bundle)
    choices = {'control': None, **choices}
    if args.configs:
        choices = {n: c for n, c in choices.items() if n == 'control' or n in args.configs}
    result = {'codebooks': 32, 'layers': args.layers, 'configs': choices,
        'checks': [], 'resources': {}, 'graph_edges': {}, 'errors': {}, 'rows': [],
        'method': 'Selected G32 norm/gate/up -> down -> following clustered QKV/head preparation. '
                  'Selected one-CTA IG1/IR2 gate/up and eight-row down control versus CUDA 13 register/resource reassembly. Host remains Triton 3.7.1. '
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
                candidate = Chain(None, cluster_options, cluster_launch)
                candidate.up = control.up
                if options is not None:
                    candidate.up = make_up(up_launchers[name], options)
                for entry in ring:
                    for index in (0, 10, 31):
                        raw = entry['raw']
                        up_args = (raw['x'][index:index+1], raw['residual'][index:index+1] if raw['has_residual'] else None,
                                   raw['weight'], raw['eps'], *entry['weights']['up'])
                        reference_up = control.up(*up_args, fused=True, **control.up_options, debug=True)
                        actual_up = candidate.up(*up_args, fused=True, **candidate.up_options, debug=True)
                        norm_exact = exact(reference_up, actual_up)
                        poison(entry); expected = control(entry, index, debug=True)
                        cache = [entry['attention'][n].clone() for n in ('k', 'v')]
                        poison(entry); actual = candidate(entry, index, debug=True)
                        output_exact = exact(actual, expected)
                        cache_exact = exact([entry['attention'][n] for n in ('k', 'v')], cache)
                        row = {'config': name, 'layer': entry['layer'], 'input': index,
                               'outputs_exact': output_exact, 'poisoned_cache_exact': cache_exact, 'norm_debug_exact': norm_exact}
                        result['checks'].append(row)
                        assert norm_exact and output_exact and cache_exact, row
                _, kernels = candidate(ring[0], audit=True)
                result['resources'][name] = {stage: {'registers': kernel.n_regs,
                    'spills': kernel.n_spills, 'shared_bytes': kernel.metadata.shared,
                    'cubin_sha256': hashlib.sha256(kernel.asm['cubin']).hexdigest(),
                    'cuda_resource_metadata': getattr(kernel, 'record', {}).get('resources')} for stage, kernel in kernels.items()}
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
