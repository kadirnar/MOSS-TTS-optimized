"""Exact MLP/down/next-QKV chains under programmatic dependent launch."""
import argparse
import ctypes
import json
import statistics

import torch

from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_norm_projection import linear as norm_reference, SELECTED
from .dp4a_layout import linear as projection_reference
from .dp4a_norm_pdl import linear as norm_pdl
from .dp4a_layout_pdl import linear as projection_pdl
from .short_scales import PLAN
from .benchmark_norm_projection import flatten
from .tune_weight_reads import measure


class EdgeData(ctypes.Structure):
    _fields_ = [('from_port', ctypes.c_ubyte), ('to_port', ctypes.c_ubyte),
                ('type', ctypes.c_ubyte), ('reserved', ctypes.c_ubyte * 5)]


def graph_edges(graph):
    lib = ctypes.CDLL('libcuda.so.1')
    fn = lib.cuGraphGetEdges_v2
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                   ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(EdgeData),
                   ctypes.POINTER(ctypes.c_size_t)]
    fn.restype = ctypes.c_int
    count = ctypes.c_size_t()
    handle = ctypes.c_void_p(graph.raw_cuda_graph())
    result = fn(handle, None, None, None, ctypes.byref(count))
    if result: raise RuntimeError(('cuGraphGetEdges_v2', result))
    sources = (ctypes.c_void_p * count.value)()
    destinations = (ctypes.c_void_p * count.value)()
    data = (EdgeData * count.value)()
    result = fn(handle, sources, destinations, data, ctypes.byref(count))
    if result: raise RuntimeError(('cuGraphGetEdges_v2', result))
    return [{'from_port': d.from_port, 'to_port': d.to_port, 'type': d.type}
            for d in data]


def load_layer(layer):
    raw = torch.load(RESULTS/f'norm_projection_capture_v1/{layer:02d}_up.pt',
                     map_location='cuda', weights_only=True)
    next_raw = torch.load(RESULTS/f'norm_projection_capture_v1/{(layer+1)%36:02d}_qkv.pt',
                          map_location='cuda', weights_only=True)
    weights = {}
    for name in ('up', 'down', 'qkv'):
        index = (layer + 1) % 36 if name == 'qkv' else layer
        saved = torch.load(RESULTS/f'gptq_v1_g32_d10/{index:02d}_{name}.pt',
                           map_location='cuda', weights_only=True)
        weights[name] = (pack_interleaved(saved['packed']), saved['scales'].bfloat16())
    return {'raw': raw, 'next': next_raw, 'weights': weights, 'layer': layer}


def chain(entry, config, *, index=0, debug=False, stages=3):
    d = entry['raw']; nd = entry['next']; weights = entry['weights']
    norm = norm_reference if config is None else norm_pdl
    project = projection_reference if config is None else projection_pdl
    nc = {} if config is None else {'pdl': config['pdl'], 'trigger_mode': config['norm_trigger']}
    pc = {} if config is None else {'pdl': config['pdl'], 'trigger_mode': config['projection_trigger']}
    if config and 'norm_prefetch' in config:
        from .dp4a_norm_pdl_prefetch import linear as norm
        from .dp4a_layout_pdl_prefetch import linear as project
        nc['prefetch'] = config['norm_prefetch']
        pc['prefetch'] = config['projection_prefetch']
    residual = d['residual'][index:index+1] if d['has_residual'] else None
    result = norm(d['x'][index:index+1], residual, d['weight'], d['eps'],
                  *weights['up'], fused=True, debug=debug, **SELECTED['up'], **nc)
    summed, hidden, quantized = result[0] if debug else result
    output = project(hidden, *weights['down'], prequantized=quantized, **PLAN['down'], **pc)
    if stages == 2: return result, output
    following = norm(output, summed, nd['weight'], nd['eps'],
                     *weights['qkv'], debug=debug, **SELECTED['qkv'], **nc)
    return result, output, following


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--layers', type=int, default=36)
    parser.add_argument('--prefetch', action='store_true')
    args = parser.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe tag required')
    if args.layers not in (1,36) or args.rounds < 2:
        raise ValueError('Use one pilot layer or the complete ring, at least two rounds')
    path = RESULTS/f'projection_pdl_{args.tag}.json'
    if path.exists(): raise FileExistsError('Preserve previous measurements')
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.set_num_threads(4)
    ring = [load_layer(i) for i in (range(36) if args.layers == 36 else (17,))]
    configs = {'control': None, 'clone_no_pdl': {'pdl': False, 'norm_trigger': 0, 'projection_trigger': 0}}
    for nt in (0,1,2,3):
        for pt in (0,1,3):
            configs[f'pdl_n{nt}_p{pt}'] = {'pdl': True, 'norm_trigger': nt, 'projection_trigger': pt}
    if args.prefetch:
        configs = {name:configs[name] for name in ('control','clone_no_pdl','pdl_n1_p3')}
        for np in (0,1,2,3):
            for pp in (0,1,2,3):
                configs[f'pre_n{np}_p{pp}'] = {'pdl':True,'norm_trigger':1,'projection_trigger':3,
                                              'norm_prefetch':np,'projection_prefetch':pp}
    checks = []; edges = {}; rows = []
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for entry in ring:
            for index in (0,10,31):
                expected = chain(entry, None, index=index, debug=True)
                for name, config in configs.items():
                    if config is None: continue
                    actual = chain(entry, config, index=index, debug=True)
                    counts = [int((a != b).sum()) for a,b in zip(flatten(actual), flatten(expected), strict=True)]
                    checks.append({'layer': entry['layer'], 'input': index, 'config': name, 'mismatches': counts})
            print('CHECKED', entry['layer'], flush=True)
        entry = ring[0]
        for name, config in configs.items():
            for _ in range(2): chain(entry, config)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream): captured = chain(entry, config)
            edges[name] = graph_edges(graph)
            graph.replay(); stream.synchronize()
            expected = chain(entry, None)
            assert all(torch.equal(a,b) for a,b in zip(flatten(captured),flatten(expected),strict=True)), name
            # All three captured kernels must have the requested edge type.
            assert len(edges[name]) == 2, (name, edges[name])
            assert all(e['type'] == (1 if config and config['pdl'] else 0) for e in edges[name]), (name,edges[name])
    torch.cuda.current_stream().wait_stream(stream)
    result = {'codebooks':32, 'torch':torch.__version__, 'configs':configs, 'checks':checks,
              'graph_edges':edges, 'rows':rows, 'layers':args.layers,
              'method':'Actual norm/up -> down -> next-layer norm/QKV dependency chains. All buffers rotate with layer. '
                       'Private-stream eager and captured graph checks; graph edge data inspected through CUDA Driver API. '
                       'Trigger points: zero implicit, one start after wait, two after normalization, three before output store. '
                       'Standalone stage-chain timings, not TTFA.'}
    path.write_text(json.dumps(result,indent=2)+'\n')
    for repeat in range(args.rounds):
        order = list(configs); order = order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2: order.reverse()
        timing = {name:measure(lambda e:chain(e,configs[name]),ring) for name in order}
        rows.append({'round':repeat,'order':order,'us':timing})
        print('ROUND',repeat,timing,flush=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
    result['summary'] = {name:{'median_us':statistics.median(r['us'][name] for r in rows),
                                'mismatches':sum(sum(c['mismatches']) for c in checks if c['config']==name)}
                         for name in configs}
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY',result['summary'],flush=True)


if __name__ == '__main__': main()
