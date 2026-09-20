"""Changed-input private graphs for exact clustered gate/up memory checking."""
import argparse
import json
import torch
from .common import RESULTS
from .benchmark_gateup_cluster import Chain, load_layer, poison, exact, make_up
from .benchmark_projection_pdl import graph_edges
from .qkv_cluster_binary import load_bundle
from .gateup_cluster_binary import load_bundle as load_up_bundle


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--bundle', required=True)
    p.add_argument('--configs', nargs='+', required=True)
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe tag required')
    path = RESULTS/f'gateup_cluster_validation_{args.tag}.json'
    if path.exists():
        raise FileExistsError('Preserve evidence')
    torch.set_num_threads(4)
    q_choices, q_launchers = load_bundle(RESULTS/'qkv_cluster_bundle_v6')
    choices, launchers = load_up_bundle(RESULTS/args.bundle)
    control = Chain(None, q_choices['c8_t2_exact'], q_launchers['c8_t2_exact'])
    result = {'codebooks': 32, 'complete': False, 'bundle': args.bundle,
        'configs': args.configs, 'checks': [], 'edges': {},
        'scope': 'Private graphs, three real layers, changed real/zero/spike input and residual. '
                 'Standalone gate/up add/no-add and production/debug cubins; dependency chain '
                 'with selected down and clustered QKV, poisoned entire-cache comparison. '
                 'Operator coverage, not whole-service sanitization.'}
    def save():
        path.write_text(json.dumps(result, indent=2)+'\n')
    def record(row):
        result['checks'].append(row); save()
        assert row['exact'], row
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for layer in (0, 17, 35):
            e = load_layer(layer); raw = e['raw']; d = e['attention']
            x = raw['x'][:1]; residual = raw['residual'][:1]
            original_x = x.clone(); original_res = residual.clone()
            def change(kind):
                if kind == 'real':
                    x.copy_(original_x); residual.copy_(original_res)
                else:
                    x.zero_(); residual.zero_()
                    if kind == 'spike':
                        x.reshape(-1)[-1] = 3
            for name in args.configs:
                candidate = Chain(None, q_choices['c8_t2_exact'], q_launchers['c8_t2_exact'])
                candidate.up = make_up(launchers[name], choices[name])
                for add in (False, True):
                    for debug in (False, True):
                        up_args = (x, residual if add else None, raw['weight'], raw['eps'], *e['weights']['up'])
                        def invoke(fn):
                            return fn(*up_args, fused=True, **control.up_options, debug=debug)
                        for _ in range(2):
                            invoke(candidate.up)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            actual = invoke(candidate.up)
                        for kind in ('real', 'zero', 'spike'):
                            change(kind); graph.replay(); expected = invoke(control.up)
                            record({'scope': 'up', 'layer': layer, 'config': name, 'add': add,
                                    'debug': debug, 'input': kind, 'exact': exact(actual, expected)})
                change('real')
                for _ in range(2):
                    candidate(e, debug=True)
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, stream=stream):
                    actual = candidate(e, debug=True)
                edges = graph_edges(graph)
                assert len(edges) == 2 and all(edge['type'] == 1 for edge in edges)
                result['edges'][f'{layer}/{name}'] = edges
                for pos in (0, 127, 1023):
                    d['position'].fill_(pos)
                    for kind in ('real', 'zero', 'spike'):
                        change(kind); poison(e); expected = control(e, debug=True)
                        cache = tuple(d[n].clone() for n in ('k', 'v'))
                        poison(e); graph.replay()
                        record({'scope': 'chain', 'layer': layer, 'config': name, 'position': pos,
                            'input': kind, 'exact': exact(actual, expected) and exact(tuple(d[n] for n in ('k', 'v')), cache)})
                change('real')
                print('CHECKED', layer, name, flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result.update(complete=True, cases=len(result['checks']))
    save(); print('PASSED', result['cases'], flush=True)


if __name__ == '__main__':
    main()
