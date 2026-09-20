"""Changed-input and poisoned-cache private graphs for cluster memory checking."""
import argparse
import json
from pathlib import Path
import torch
from .common import RESULTS
from .benchmark_qkv_cluster import Chain
from .benchmark_attention_pdl import load_layer,library
from .benchmark_norm_projection import flatten
from .qkv_cluster_binary import load_bundle


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--binary-bundle',type=Path,required=True)
    p.add_argument('--configs',nargs='+',default=['c8_t2_exact']);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'qkv_cluster_validation_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);library();choices,launchers=load_bundle(a.binary_bundle)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());checks=[];control=Chain()
    with torch.cuda.stream(stream):
        entries=[load_layer(i,1024) for i in (0,17,35)]
        saved=torch.load(RESULTS/'qkv_trajectory_v1/mismatch.pt',map_location='cuda',weights_only=True)
        e=load_layer(30,1024);raw=e['raw'];d=e['attention']
        raw.update(x=saved['x'],residual=saved['residual'],weight=saved['norm_weight'],eps=saved['eps'])
        e['weights']['qkv']=(saved['w'],saved['s'])
        d.update(qw=saved['qw'],kw=saved['kw'],cos=saved['cos'],sin=saved['sin'],k=saved['kc'],v=saved['vc'],position=saved['position'],eps=saved['head_eps'])
        entries.append(e)
        for e in entries:
            raw=e['raw'];d=e['attention'];x=raw['x'][:1];original=x.clone()
            for name in a.configs:
                candidate=Chain(choices[name],launcher=launchers[name])
                for debug in (False,True):
                    for _ in range(2):candidate(e,debug=debug)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):actual=candidate(e,debug=debug)
                    for pos in (0,127,1023):
                        d['position'].fill_(pos)
                        for kind in ('real','zero','spike'):
                            x.copy_(original) if kind=='real' else x.zero_()
                            if kind=='spike':x.reshape(-1)[-1]=3
                            for n in ('k','v'):d[n][:,:,pos,:].fill_(float('nan'))
                            expected=control(e,debug=debug);caches=[d[n].clone() for n in ('k','v')]
                            for n in ('k','v'):d[n][:,:,pos,:].fill_(float('nan'))
                            graph.replay()
                            exact=all(torch.equal(x.reshape(-1).view(torch.uint8),y.reshape(-1).view(torch.uint8)) for x,y in zip(flatten(actual),flatten(expected),strict=True))
                            cache_exact=all(torch.equal(d[n].view(torch.uint8),c.view(torch.uint8)) for n,c in zip(('k','v'),caches,strict=True))
                            row={'layer':e['layer'],'config':name,'debug':debug,'position':pos,'input':kind,'outputs_exact':exact,'poisoned_caches_exact':cache_exact}
                            checks.append(row)
                            path.write_text(json.dumps({'codebooks':32,'checks':checks,'complete':False},indent=2)+'\n')
                            assert exact and cache_exact,row
                    x.copy_(original)
                    print('CHECKED',e['layer'],name,debug,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    path.write_text(json.dumps({'codebooks':32,'checks':checks,'complete':True,'cases':len(checks),'bundle':str(a.binary_bundle),
        'scope':'Private CUDA graphs of the four-kernel fused attention chain. Real/zero/spike activation inputs, positions 0/127/1023, three frozen layers plus the saved layer30 trajectory divergence. Poisoned current KV positions and entire caches compared; debug and production cubins tested.'},indent=2)+'\n')
    print('PASSED',len(checks),flush=True)


if __name__=='__main__':main()
