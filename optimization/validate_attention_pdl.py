"""Changed inputs, poisoned future KV, guarded storage and private graph replays."""
import argparse
import json
import torch

from .common import RESULTS
from .benchmark_attention_pdl import load_layer,chain,configurations
from .benchmark_projection_pdl import graph_edges
from .benchmark_norm_projection import flatten
from .attention_pdl import library


def guarded(tensor):
    owner=torch.full((tensor.numel()+64,),17,dtype=tensor.dtype,device=tensor.device)
    value=owner[32:-32].view_as(tensor);value.copy_(tensor)
    return value,owner


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'attention_pdl_memory_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);torch.manual_seed(9918);library()
    configs={k:configurations()[k] for k in ('q1_a2_r1','pre_q2_a1_r1')}
    rows=[];edges={}
    for layer in (0,17,35):
        entry=load_layer(layer);d=entry['attention'];raw=entry['raw']
        original=raw['x'][:1].clone()
        values=[raw['x'][i:i+1].clone() for i in (0,10,31)]
        spike=torch.zeros_like(original);spike[0,-1]=100
        values.extend((torch.zeros_like(original),spike,torch.randn_like(original)))
        saved={key:d[key].clone() for key in ('k','v')};owners=[]
        for key in ('k','v'):
            d[key],owner=guarded(d[key]);owners.append(owner)
        ref=dict(entry);ref['attention']=dict(d)
        for key in ('k','v'):ref['attention'][key]=d[key].clone()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for cap in (128,256,512,1024):
                entry['capacity']=ref['capacity']=cap;d['position'].fill_(0)
                for name,config in configs.items():
                    for key in ('k','v'):d[key].copy_(saved[key])
                    for _ in range(2):chain(entry,config)
                    graph=torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph,stream=stream):actual=chain(entry,config,debug=True)
                    edges[f'{layer}/{cap}/{name}']=graph_edges(graph)
                    assert len(edges[f'{layer}/{cap}/{name}'])==4 and all(e['type']==1 for e in edges[f'{layer}/{cap}/{name}'])
                    for index,pos in enumerate((0,1,31,32,cap//2,cap-1,0)):
                        raw['x'][:1].copy_(values[index%6]);d['position'].fill_(pos)
                        for key in ('k','v'):
                            d[key].copy_(saved[key]);d[key][...,pos+1:,:].fill_(float('nan'))
                            ref['attention'][key].copy_(d[key])
                        graph.replay();expected=chain(ref,debug=True)
                        counts=[int((a!=b).sum()) for a,b in zip(flatten(actual),flatten(expected),strict=True)]
                        caches_exact=all(torch.equal(d[key].view(torch.int16),ref['attention'][key].view(torch.int16)) for key in ('k','v'))
                        guards=all(bool((owner[:32]==17).all() and (owner[-32:]==17).all()) for owner in owners)
                        row={'layer':layer,'capacity':cap,'config':name,'input':index%6,'position':pos,
                             'mismatches':counts,'cache_bytes_exact':caches_exact,'guards_unchanged':guards}
                        rows.append(row);assert not any(counts) and caches_exact and guards,row
                    del graph,actual
                print('PASS',layer,cap,flush=True)
            raw['x'][:1].copy_(original)
        torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'cases':rows,'all_exact':True,'graph_edges':edges,'private_stream':True,
            'changed_inputs_between_replays':True,'future_cache_poison':'BF16 NaN after active position',
            'scope':'Two PDL candidates; three real layers, four capacities, seven position/input replays each. '
                    'Complete five-kernel chain intermediates and both entire KV caches compared with ordinary attention control. '
                    'Guarded cache allocation, independent reference caches; real/zero/spike/random normalization inputs. '
                    'Last replay returns to position zero to detect stale data.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),flush=True)


if __name__=='__main__':main()
