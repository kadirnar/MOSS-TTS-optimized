"""Bitwise copy, private-stream graphs and DLPack lifetime checks for CUDA VMM."""
import argparse
import gc
import json

import torch

from .common import RESULTS
from .compressed_alloc import clone,library,DTYPES


@torch.inference_mode()
def case(dtype,count,compressed):
    size=torch.empty((),dtype=dtype).element_size()
    raw=torch.randint(0,256,(count*size,),device='cuda',dtype=torch.uint8)
    source=raw.view(dtype)
    target,meta=clone(source,compressed=compressed)
    assert torch.equal(target.view(torch.uint8),raw)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):result=target.view(torch.uint8)^37
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):result=target.view(torch.uint8)^37
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(result,raw^37)
    changed=torch.randint(0,256,raw.shape,device='cuda',dtype=torch.uint8)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        target.view(torch.uint8).copy_(changed)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(result,changed^37)
    del graph,result
    view=target.view(torch.uint8)[1:];del target;gc.collect()
    assert torch.equal(view,changed[1:])
    del view
    torch.cuda.synchronize()
    return {'dtype':str(dtype),'count':count,'compressed':compressed,'allocation':meta,
            'initial_copy_exact':True,'private_graph_exact':True,'updated_graph_exact':True,'view_lifetime_exact':True}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'compressed_alloc_validation_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);torch.manual_seed(923);torch.cuda.init()
    rows=[];initial=library().counters()
    for compressed in (False,True):
        for dtype in DTYPES:
            for n in (1,127,1048591):
                rows.append(case(dtype,n,compressed));torch.cuda.synchronize();gc.collect()
                print('CASE',compressed,str(dtype),n,library().counters(),flush=True)
                assert library().counters()==initial,library().counters()
            print(compressed,dtype,'PASS',flush=True)
    # Unconsumed capsules own and release their allocations too.
    capsule,meta=library().allocate((1024,),1,8,0,1)
    assert library().counters()['live_allocations']==initial['live_allocations']+1
    del capsule;gc.collect();assert library().counters()==initial
    result={'cases':rows,'all_exact':True,'private_stream_graphs':True,
            'final_counters':library().counters(),'unconsumed_capsule_cleanup':True}
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',len(rows),result['final_counters'],flush=True)


if __name__=='__main__':main()
