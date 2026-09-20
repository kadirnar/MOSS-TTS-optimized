"""Changing-input PDL chains on private CUDA streams and captured graphs."""
import argparse
import json
import torch

from .common import RESULTS
from .benchmark_projection_pdl import chain, graph_edges, load_layer
from .benchmark_norm_projection import flatten


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe tag required')
    path = RESULTS/f'projection_pdl_memory_{args.tag}.json'
    if path.exists(): raise FileExistsError('Preserve previous results')
    torch.set_num_threads(4); torch.manual_seed(6152)
    configs = {'pdl':{'pdl':True,'norm_trigger':1,'projection_trigger':3},
               'prefetch':{'pdl':True,'norm_trigger':1,'projection_trigger':3,
                           'norm_prefetch':0,'projection_prefetch':1}}
    rows = []; edges = {}
    for layer in (0,17,35):
        entry = load_layer(layer); d = entry['raw']; saved = d['x'][:1].clone()
        inputs = [(str(i),d['x'][i:i+1].clone()) for i in (0,10,31)]
        zero = torch.zeros_like(saved); spike = zero.clone(); spike[0,-1] = 100
        inputs.extend((('zero',zero),('spike',spike),('random',torch.randn_like(saved))))
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for name, config in configs.items():
                d['x'][:1].copy_(saved)
                for _ in range(2): chain(entry,config)
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream): actual = chain(entry,config)
                edges[f'{layer}/{name}'] = graph_edges(graph)
                assert all(e['type']==1 for e in edges[f'{layer}/{name}'])
                for label,x in inputs:
                    # The producer copy and the dependent graph use this private stream.
                    d['x'][:1].copy_(x); graph.replay()
                    expected = chain(entry,None)
                    counts = [int((a!=b).sum()) for a,b in zip(flatten(actual),flatten(expected),strict=True)]
                    row = {'layer':layer,'config':name,'input':label,'mismatches':counts}
                    rows.append(row); assert not any(counts),row
                    print('PASS',layer,name,label,flush=True)
                del graph,actual
            d['x'][:1].copy_(saved)
        torch.cuda.current_stream().wait_stream(stream)
    result = {'codebooks':32,'cases':rows,'all_exact':True,'graph_edges':edges,
              'private_stream':True,'changed_inputs_between_replays':True,
              'scope':'36 actual-shape dependent-chain cases; selected PDL and scale-preload candidates, '
                      'layers 0/17/35, three real vectors plus zero/spike/random. '
                      'Complete-model and other-layer checks are separate.'}
    path.write_text(json.dumps(result,indent=2)+'\n'); print('PASS',len(rows),flush=True)


if __name__ == '__main__': main()
