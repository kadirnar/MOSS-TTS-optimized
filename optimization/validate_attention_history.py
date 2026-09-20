"""Changed-history/position replay checks for historical KV preloads."""
import argparse
import json
import torch
from .common import RESULTS
from .attention_history import configs, library
from .benchmark_attention_history import chains, exact
from .benchmark_attention_pdl import load_layer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--production', action='store_true', help='Use the QKV producer without intermediate debug stores')
    args = parser.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe tag required')
    path = RESULTS / f'attention_history_validation_{args.tag}.json'
    if path.exists():
        raise FileExistsError('Preserve evidence')
    torch.set_num_threads(4); library()
    choices = {'control': None, **{n: configs()[n] for n in args.configs}}
    candidates = chains(choices)
    result = {'codebooks': 32, 'complete': False, 'configs': choices, 'checks': [], 'production_qkv': args.production,
        'scope': 'Private graphs with in-graph position and entire-history copies preceding QKV. '
                 'Current slots are poisoned in staging then overwritten by QKV; old history and input '
                 'change between replays. All intermediates and whole KV buffers compared. '
                 'Layers 0/17/35, capacities 128/1024, forward/backward positions and real/zero/spike inputs. '
                 'Operator-chain coverage, not whole-service sanitization.'}
    def save():
        path.write_text(json.dumps(result, indent=2) + '\n')
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for layer in (0, 17, 35):
            for capacity in (128, 1024):
                entry = load_layer(layer, capacity); d = entry['attention']; raw = entry['raw']
                source_k, source_v = d['k'].clone(), d['v'].clone()
                original_k, original_v = source_k.clone(), source_v.clone()
                source_pos = d['position'].clone(); source_pos.fill_(0)
                x = raw['x'][:1]; residual = raw['residual'][:1] if raw['has_residual'] else None
                original_x = x.clone(); original_res = residual.clone() if residual is not None else None
                def call(candidate):
                    d['k'].copy_(source_k); d['v'].copy_(source_v); d['position'].copy_(source_pos)
                    return candidate(entry, debug=not args.production)
                for name in args.configs:
                    candidate = candidates[name]
                    for _ in range(2):
                        call(candidate)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = call(candidate)
                    for position in (0, 31, 32, capacity-1, 1):
                        source_pos.fill_(position)
                        for kind in ('real', 'zero', 'spike'):
                            x.copy_(original_x)
                            if residual is not None:
                                residual.copy_(original_res)
                            source_k.copy_(original_k); source_v.copy_(original_v)
                            if kind != 'real':
                                x.zero_()
                                if residual is not None:
                                    residual.zero_()
                                source_k[:,:,0,:].neg_(); source_v[:,:,0,:].neg_()
                                if kind == 'spike':
                                    x.reshape(-1)[-1] = 3
                                    source_k[:,:,0,0].fill_(0.5); source_v[:,:,0,0].fill_(0.25)
                            source_k[:,:,position,:].fill_(float('nan'))
                            source_v[:,:,position,:].fill_(float('nan'))
                            expected = call(candidates['control'])
                            cache = (d['k'].clone(), d['v'].clone())
                            graph.replay()
                            check = {'layer': layer, 'capacity': capacity, 'config': name,
                                'position': position, 'input': kind,
                                'exact': exact(actual, expected) and exact((d['k'], d['v']), cache)}
                            result['checks'].append(check); save(); assert check['exact'], check
                    print('CHECKED', layer, capacity, name, flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result.update(complete=True, cases=len(result['checks'])); save()
    print('PASSED', result['cases'], flush=True)


if __name__ == '__main__':
    main()
