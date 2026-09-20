"""Changed-input private graphs and odd row tails for the eight-row down tile."""
import argparse
import json

import torch
from .common import RESULTS
from .benchmark_down_preload import Chain, load_layer, poison, exact
from .benchmark_projection_pdl import graph_edges
from .benchmark_group128 import quantize
from .bulk_prefetch import configured as projection_configured
from .qkv_cluster_binary import load_bundle
from .cluster_placement import configured as placement_configured
from .short_scales import PLAN


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--placement', action='store_true')
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe tag required')
    path = RESULTS/f'down_tile_validation_{args.tag}.json'
    if path.exists():
        raise FileExistsError('Preserve evidence')
    torch.set_num_threads(4)
    folder = RESULTS/'qkv_cluster_bundle_v6'
    choices, launchers = load_bundle(folder)
    options = choices['c8_t2_exact']; launcher = launchers['c8_t2_exact']
    control = Chain(None, options, launcher)
    if args.placement:
        options, launcher = placement_configured(folder, policy=2)
    candidate = Chain({'rows': 8, 'warps': 4, 'prefetch': 1}, options, launcher)
    project = projection_configured('projection', divisor=16)
    result = {'codebooks': 32, 'placement': args.placement, 'complete': False,
        'checks': [], 'edges': {},
        'scope': 'Selected gate/up -> eight-row/four-warp down -> clustered QKV/head-preparation '
                 'private graphs with changed inputs and entire poisoned-cache comparisons. '
                 'Production/debug consumers, positions 0/127/1023; separate odd-row down checks. '
                 'Operator-chain sanitizer coverage, not whole-service sanitization.'}
    def save():
        path.write_text(json.dumps(result, indent=2)+'\n')
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for layer in (0, 17, 35):
            entry = load_layer(layer); raw = entry['raw']; d = entry['attention']
            x = raw['x'][:1]; original = x.clone()
            for debug in (False, True):
                for _ in range(2):
                    candidate(entry, debug=debug)
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, stream=stream):
                    actual = candidate(entry, debug=debug)
                edges = graph_edges(graph)
                assert len(edges) == 2 and all(e['type'] == 1 for e in edges)
                result['edges'][f'{layer}/{debug}'] = edges
                for pos in (0, 127, 1023):
                    d['position'].fill_(pos)
                    for kind in ('real', 'zero', 'spike'):
                        x.copy_(original) if kind == 'real' else x.zero_()
                        if kind == 'spike':
                            x.reshape(-1)[-1] = 3
                        poison(entry); expected = control(entry, debug=debug)
                        cache = tuple(d[n].clone() for n in ('k', 'v'))
                        poison(entry); graph.replay()
                        row = {'scope': 'chain', 'layer': layer, 'debug': debug,
                            'position': pos, 'input': kind,
                            'outputs_exact': exact(actual, expected),
                            'cache_exact': exact(tuple(d[n] for n in ('k', 'v')), cache)}
                        result['checks'].append(row); save()
                        assert row['outputs_exact'] and row['cache_exact'], row
                x.copy_(original)
            down_x = control(entry)[0][1].clone(); down_original = down_x.clone()
            packed, scales = entry['weights']['down']
            for n in (4096, 4093):
                def invoke(candidate_tile):
                    tile = {**PLAN['down'], **({'rows': 8, 'warps': 4} if candidate_tile else {})}
                    return project(down_x, packed[:n], scales[:n], prequantized=quantize(down_x, 32),
                                   **tile, trigger_mode=3, prefetch=1)
                for _ in range(2):
                    invoke(True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = invoke(True)
                for kind in ('real', 'zero', 'spike'):
                    down_x.copy_(down_original) if kind == 'real' else down_x.zero_()
                    if kind == 'spike':
                        down_x.reshape(-1)[-1] = 3
                    graph.replay(); expected = invoke(False)
                    row = {'scope': 'projection', 'layer': layer, 'output_rows': n,
                           'input': kind, 'outputs_exact': exact(actual, expected)}
                    result['checks'].append(row); save()
                    assert row['outputs_exact'], row
                down_x.copy_(down_original)
            print('CHECKED', layer, flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result.update(complete=True, cases=len(result['checks']))
    save(); print('PASSED', result['cases'], flush=True)


if __name__ == '__main__':
    main()
