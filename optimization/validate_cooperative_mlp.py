"""Changed-input graph replay checks for cooperative MLP joins."""
import argparse
import json

import torch

from .common import RESULTS
from .benchmark_cooperative_mlp import Chain, configs, load_mlp, load_attention, exact
from .benchmark_norm_projection import flatten


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True)
    p.add_argument('--configs', nargs='+', required=True, choices=tuple(configs()))
    a = p.parse_args()
    if not a.tag or not all(c.isalnum() or c == '_' for c in a.tag):
        raise ValueError('Safe tag required')
    path = RESULTS/f'cooperative_mlp_validation_{a.tag}.json'
    if path.exists(): raise FileExistsError('Preserve evidence')
    torch.set_num_threads(4)
    result = {'codebooks': 32, 'complete': False, 'checks': [], 'configs': a.configs,
        'production_qkv': True,
        'scope': 'Operator chains, not whole-service sanitization. Private graphs with in-graph position/history copies, '
                 'poisoned current KV and all returned intermediates, changed real/zero/spike MLP inputs, '
                 'three layers and forward/backward positions. Exact outputs/whole caches and reusable barrier sense.'}
    def save(): path.write_text(json.dumps(result, indent=2)+'\n')
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for layer in (0, 17, 35):
            entry = load_mlp(layer); entry['following'] = load_attention((layer+1)%36)
            d = entry['attention'] = entry['following']['attention']; raw = entry['raw']
            assert raw['has_residual']
            x = raw['x'][:1]; residual = raw['residual'][:1]
            original_x, original_residual = x.clone(), residual.clone()
            original_k, original_v = d['k'].clone(), d['v'].clone()
            source_k, source_v = original_k.clone(), original_v.clone()
            source_pos = d['position'].clone()
            control = Chain([entry])
            def call(candidate):
                d['k'].copy_(source_k); d['v'].copy_(source_v); d['position'].copy_(source_pos)
                return candidate(entry, debug=False)
            for name in a.configs:
                candidate = Chain([entry], configs()[name])
                counter = candidate.fused[(raw['has_residual'], raw['eps'])].counter
                for _ in range(2): call(candidate)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream): actual = call(candidate)
                for position in (0, 31, 32, 255, 1):
                    source_pos.fill_(position)
                    for kind in ('real', 'zero', 'spike'):
                        x.copy_(original_x); residual.copy_(original_residual)
                        source_k.copy_(original_k); source_v.copy_(original_v)
                        if kind != 'real':
                            x.zero_(); residual.zero_()
                            source_k[:, :, 0].neg_(); source_v[:, :, 0].neg_()
                            if kind == 'spike':
                                x.reshape(-1)[-1] = 3
                                source_k[:, :, 0, 0].fill_(0.5); source_v[:, :, 0, 0].fill_(0.25)
                        source_k[:, :, position].fill_(float('nan'))
                        source_v[:, :, position].fill_(float('nan'))
                        expected = call(control)
                        cache = (d['k'].clone(), d['v'].clone())
                        for value in flatten(actual):
                            value.fill_(float('nan') if value.is_floating_point() else -91)
                        before = int(counter.item()) & 0xffffffff
                        graph.replay()
                        after = int(counter.item()) & 0xffffffff
                        row = {'layer': layer, 'config': name, 'position': position, 'input': kind,
                            'exact': exact(actual, expected) and exact((d['k'], d['v']), cache),
                            'barrier_before': before, 'barrier_after': after,
                            'barrier_exact': before in (0, 0x80000000) and after == (before ^ 0x80000000)}
                        result['checks'].append(row); save()
                        assert row['exact'] and row['barrier_exact'], row
                print('CHECKED', layer, name, flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result.update(complete=True, cases=len(result['checks'])); save()
    print('PASSED', result['cases'], flush=True)


if __name__ == '__main__': main()
