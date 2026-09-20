"""Cooperative G32 MLP versus selected overlapped MLP/attention chain."""
import argparse
import hashlib
import json
import statistics
import traceback

import torch

from .common import RESULTS
from .benchmark_projection_pdl import load_layer as load_mlp, graph_edges
from .benchmark_attention_pdl import load_layer as load_attention
from .benchmark_gateup_cluster import Chain as MLP, exact, poison
from .benchmark_attention_history import chains as attention_chains
from .gateup_compiler import make_dispatch
from .qkv_cluster_binary import load_bundle
from .cooperative_mlp import Fused
from .tune_weight_reads import measure


def configs():
    choices = {f'g{blocks}_e{int(early)}_r{cap or 0}': {'blocks': blocks, 'early': early, 'cap': cap}
            for blocks in (128, 256, 384, 512) for early in (False, True)
            for cap in ((None, 128, 160) if blocks < 512 else (128,))}
    for blocks, cap in ((384, None), (512, 128), (360, None), (480, 128)):
        for early in (False, True):
            for cluster in (2, 4, 8):
                if blocks in (360, 480) and cluster == 2:
                    continue
                choices[f'g{blocks}_e{int(early)}_r{cap or 0}_c{cluster}'] = {
                    'blocks': blocks, 'early': early, 'cap': cap, 'cluster': cluster}
    return choices


class Chain:
    def __init__(self, ring, options=None):
        choices, launchers = load_bundle(RESULTS/'qkv_cluster_bundle_history_v1')
        self.control = MLP(None, choices['c8_exact'], launchers['c8_exact'])
        self.control.up = make_dispatch(self.control.up)
        self.follow = attention_chains({'selected': {'producer': 1, 'trigger': 1, 'mode': 3, 'packed': False}})['selected']
        self.fused = {}
        self.resources = {}
        for entry in ring:
            d = entry['raw']
            key = (d['has_residual'], d['eps'])
            if key not in self.fused:
                up = self.control.up(d['x'][:1], d['residual'][:1] if d['has_residual'] else None,
                    d['weight'], d['eps'], *entry['weights']['up'], fused=True, **self.control.up_options)
                _, down_kernel = self.control.down(up[1], *entry['weights']['down'], prequantized=up[2],
                    **self.control.tile, return_kernel=True)
                self.resources['down_source_sha256'] = hashlib.sha256(down_kernel.asm['cubin']).hexdigest()
                self.fused[key] = None if options is None else Fused(entry, down_kernel, **options)
                if options is not None:
                    self.resources[str(key)] = self.fused[key].resources

    def __call__(self, entry, index=0, debug=False):
        d = entry['raw']; fused = self.fused[(d['has_residual'], d['eps'])]
        if fused is None:
            up = self.control.up(d['x'][index:index+1], d['residual'][index:index+1] if d['has_residual'] else None,
                d['weight'], d['eps'], *entry['weights']['up'], fused=True, **self.control.up_options)
            down = self.control.down(up[1], *entry['weights']['down'], prequantized=up[2], **self.control.tile)
        else:
            up, down = fused(entry, index)
        follow = entry['following']
        follow = {**follow, 'raw': {**follow['raw'], 'x': down, 'residual': up[0], 'has_residual': True}}
        result = self.follow(follow, debug=debug)
        return up, down, result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True)
    p.add_argument('--layers', type=int, choices=(1, 36), default=1)
    p.add_argument('--rounds', type=int, default=4); p.add_argument('--configs', nargs='+', choices=tuple(configs()))
    p.add_argument('--profile', action='store_true', help='Trace separate graph replays after all timing')
    a = p.parse_args()
    if not a.tag or not all(c.isalnum() or c == '_' for c in a.tag) or a.rounds < 2:
        raise ValueError('Safe tag and at least two rounds required')
    path = RESULTS/f'cooperative_mlp_{a.tag}.json'
    if path.exists(): raise FileExistsError('Preserve evidence')
    torch.set_num_threads(4)
    ring = []
    for i in (range(36) if a.layers == 36 else (17,)):
        entry = load_mlp(i); entry['following'] = load_attention((i+1)%36)
        entry['attention'] = entry['following']['attention']; ring.append(entry)
    choices = {'control': None, **{n: v for n, v in configs().items() if not a.configs or n in a.configs}}
    result = {'codebooks': 32, 'complete': False, 'layers': a.layers, 'configs': choices,
        'checks': [], 'graph_checks': [], 'edges': {}, 'resources': {}, 'errors': {}, 'rows': [],
        'method': 'Selected Triton 3.8 exact gate/up + selected Triton 3.7 eight-row down versus cooperative PTX composition. '
                  'Both followed by selected early clustered QKV, historical-KV attention, reduction/quantization and register-preloaded output. '
                  'Actual 36-layer weights and three frozen inputs, nearest saved attention metadata, synthetic layer-35 wrap; not TTFA. '
                  'Bytewise intermediates and entire poisoned KV caches, private CUDA graphs, balanced rotated/reversed event rings.'}
    def save(): path.write_text(json.dumps(result, indent=2)+'\n')
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    candidates = {}
    with torch.cuda.stream(stream):
        for name, options in choices.items():
            try:
                candidate = Chain(ring, options)
                result['resources'][name] = candidate.resources
                control = candidate if name == 'control' else candidates['control']
                for entry in ring:
                    for index in (0, 10, 31):
                        poison(entry); expected = control(entry, index, True)
                        cache = tuple(entry['attention'][n].clone() for n in ('k', 'v'))
                        poison(entry); actual = candidate(entry, index, True)
                        row = {'config': name, 'layer': entry['layer'], 'input': index,
                            'outputs_exact': exact(actual, expected),
                            'whole_cache_exact': exact(tuple(entry['attention'][n] for n in ('k', 'v')), cache)}
                        result['checks'].append(row)
                        assert row['outputs_exact'] and row['whole_cache_exact'], row
                entry = ring[0]
                for _ in range(2): candidate(entry, debug=True)
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, stream=stream): actual = candidate(entry, debug=True)
                edges = graph_edges(graph); result['edges'][name] = edges
                assert len(edges) == (5 if options is None else 4) and all(e['type'] == 1 for e in edges), edges
                old_position = int(entry['attention']['position'])
                for position in (0, 1, 31, 32, 127, 128, 144, 255):
                    entry['attention']['position'].fill_(position)
                    poison(entry); graph.replay()
                    cache = tuple(entry['attention'][n].clone() for n in ('k', 'v'))
                    poison(entry); expected = control(entry, debug=True)
                    row = {'config': name, 'position': position, 'outputs_exact': exact(actual, expected),
                           'whole_cache_exact': exact(tuple(entry['attention'][n] for n in ('k', 'v')), cache)}
                    result['graph_checks'].append(row)
                    assert row['outputs_exact'] and row['whole_cache_exact'], row
                entry['attention']['position'].fill_(old_position)
                candidates[name] = candidate
                print('CHECKED', name, candidate.resources, flush=True)
            except Exception:
                result['errors'][name] = traceback.format_exc()
                print('ERROR', name, result['errors'][name], flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    assert 'control' in candidates
    for repeat in range(a.rounds):
        order = list(candidates); offset = repeat//2 % len(order); order = order[offset:]+order[:offset]
        if repeat%2: order.reverse()
        times = {n: measure(candidates[n], ring) for n in order}
        result['rows'].append({'round': repeat, 'order': order, 'us': times})
        save(); print('ROUND', repeat, times, flush=True)
    result['summary'] = {n: {'median_us': statistics.median(r['us'][n] for r in result['rows']),
        'median_paired_gain_us': statistics.median(r['us']['control']-r['us'][n] for r in result['rows']),
        'faster_rounds': sum(r['us'][n] < r['us']['control'] for r in result['rows'])} for n in candidates}
    if a.profile:
        best = min((n for n in candidates if n != 'control'), key=lambda n: result['summary'][n]['median_us'])
        result['profiles'] = {}
        for name in ('control', best):
            for entry in ring: candidates[name](entry)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(3):
                    for entry in ring: candidates[name](entry)
            graph.replay(); torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function(name):
                    graph.replay(); torch.cuda.synchronize()
            target = RESULTS/f'cooperative_mlp_{a.tag}_{name}_trace.json'
            prof.export_chrome_trace(str(target))
            result['profiles'][name] = {'path': str(target), 'scope': 'Separate post-timing graph replay; includes resident waits, not utilization or a lower bound.'}
    result['complete'] = True; save(); print('SUMMARY', result['summary'], flush=True)


if __name__ == '__main__': main()
