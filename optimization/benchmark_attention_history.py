"""Historical KV preloads in actual QKV/attention/output dependency chains."""
import argparse
import functools
import hashlib
import json
import statistics
import types
import torch
from .common import RESULTS
from .attention_history import configs, launch, library
from .benchmark_qkv_cluster import Chain as BaseChain
from .benchmark_attention_pdl import load_layer
from .benchmark_norm_projection import flatten
from .benchmark_projection_pdl import graph_edges
from .qkv_cluster_binary import load_bundle
from .tune_weight_reads import measure


def exact(actual, expected):
    return all(torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
               for a, b in zip(flatten(actual), flatten(expected), strict=True))


def poison(entry):
    data = entry['attention']; pos = int(data['position'])
    for name in ('k', 'v'):
        data[name].view(8, -1, 128)[:, pos].fill_(float('nan'))


class Chain(BaseChain):
    def __init__(self, config, options, launcher):
        super().__init__(options, launcher)
        if config is None:
            self.call = super().__call__
        else:
            def attention(*args, pdl=True, trigger=2):
                if not pdl:
                    raise ValueError('PDL chain required')
                return launch(*args, mode=config['mode'], packed=config['packed'], trigger=config['trigger'])
            original = BaseChain.__call__
            namespace = {**original.__globals__, 'attention': attention}
            function = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
            function.__kwdefaults__ = original.__kwdefaults__
            self.call = types.MethodType(function, self)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


def chains(choices):
    options, launchers = load_bundle(RESULTS / 'qkv_cluster_bundle_v6')
    early_options, early_launchers = load_bundle(RESULTS / 'qkv_cluster_bundle_history_v1')
    options.update(early_options); launchers.update(early_launchers)
    result = {}
    for name, config in choices.items():
        producer = 1 if name == 'early_control' else 2 if config is None else config['producer']
        key = 'c8_exact' if producer == 1 else 'c8_t2_exact'
        result[name] = Chain(config, options[key], launchers[key])
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True)
    parser.add_argument('--layers', type=int, choices=(1, 36), default=1)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--capacity', type=int, choices=(128, 256, 512, 1024), default=256)
    parser.add_argument('--configs', nargs='+')
    args = parser.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag) or args.rounds < 2:
        raise ValueError('Safe tag and at least two rounds required')
    path = RESULTS / f'attention_history_{args.tag}.json'
    if path.exists():
        raise FileExistsError('Preserve evidence')
    choices = {'control': None, 'early_control': None, **configs()}
    if args.configs:
        choices = {n: choices[n] for n in ('control', *args.configs) if n in choices}
    torch.set_num_threads(4); library()
    candidates = chains(choices)
    ring = [load_layer(layer, args.capacity) for layer in (range(36) if args.layers == 36 else (17,))]
    for entry in ring:
        if args.capacity == 128:
            entry['attention']['position'].fill_(127)
    result = {'codebooks': 32, 'complete': False, 'configs': choices, 'checks': [],
        'graph_checks': [], 'edges': {}, 'rows': [], 'layers': args.layers, 'capacity': args.capacity,
        'method': 'Selected exact clustered QKV -> native split attention -> reduction/G32 quantization -> '
                  'register-preloaded output. Historical t<position KV loads precede the ordinary wait; '
                  'current-token reads and Q remain after it. Position/history writes precede the QKV producer. '
                  'Actual per-layer weights and three frozen inputs, nearest saved attention metadata; not a full trajectory or TTFA. '
                  'Balanced rotated/reversed private graph rings; exact intermediate and entire poisoned-cache comparisons.'}
    def save():
        path.write_text(json.dumps(result, indent=2) + '\n')
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for entry in ring:
            for index in (0, 10, 31):
                poison(entry); expected = candidates['control'](entry, index, True)
                cache = tuple(entry['attention'][n].clone() for n in ('k', 'v'))
                for name, candidate in candidates.items():
                    poison(entry); actual = candidate(entry, index, True)
                    check = {'layer': entry['layer'], 'input': index, 'config': name,
                        'outputs_exact': exact(actual, expected),
                        'poisoned_cache_exact': exact(tuple(entry['attention'][n] for n in ('k', 'v')), cache)}
                    result['checks'].append(check); save()
                    assert check['outputs_exact'] and check['poisoned_cache_exact'], check
            print('CHECKED', entry['layer'], flush=True)
        entry = ring[0]; old_position = int(entry['attention']['position'])
        for name, candidate in candidates.items():
            for _ in range(2):
                candidate(entry, debug=True)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                actual = candidate(entry, debug=True)
            edges = graph_edges(graph); result['edges'][name] = edges
            assert len(edges) == 3 and all(edge['type'] == 1 for edge in edges), edges
            for position in sorted({0, 1, 31, 32, args.capacity-1} | {p for p in (127, 128, 144, 255, 256, 511, 512) if p < args.capacity}):
                entry['attention']['position'].fill_(position)
                poison(entry); expected = candidates['control'](entry, debug=True)
                cache = tuple(entry['attention'][n].clone() for n in ('k', 'v'))
                poison(entry); graph.replay()
                check = {'config': name, 'position': position, 'outputs_exact': exact(actual, expected),
                         'poisoned_cache_exact': exact(tuple(entry['attention'][n] for n in ('k', 'v')), cache)}
                result['graph_checks'].append(check); save()
                assert check['outputs_exact'] and check['poisoned_cache_exact'], check
            entry['attention']['position'].fill_(old_position)
            print('GRAPH', name, flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(args.rounds):
        names = list(candidates); offset = (repeat//2) % len(names); names = names[offset:] + names[:offset]
        if repeat % 2:
            names.reverse()
        times = {name: measure(candidates[name], ring) for name in names}
        result['rows'].append({'round': repeat, 'order': names, 'us': times}); save()
        print('ROUND', repeat, times, flush=True)
    result['summary'] = {name: {
        'median_us': statistics.median(row['us'][name] for row in result['rows']),
        'median_paired_gain_us': statistics.median(row['us']['control']-row['us'][name] for row in result['rows']),
        'faster_rounds': sum(row['us'][name] < row['us']['control'] for row in result['rows']),
    } for name in candidates}
    result['complete'] = True; save(); print('SUMMARY', result['summary'], flush=True)


if __name__ == '__main__':
    main()
