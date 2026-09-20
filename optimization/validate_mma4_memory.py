"""Private-graph padded-row checks for the rejected INT4-PTX experiment."""
import argparse
import json
import torch
from .common import RESULTS
from .mma4_projection import linear,pack_weight,library
from .dp4a_packing import pack_interleaved
from .dp4a_scaled import linear as reference,SELECTED
from .benchmark_norm_projection import flatten


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'mma4_memory_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);torch.manual_seed(889);library();rows=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cases=[(n,k,False) for n in (1,7,8,15,16,17) for k in (4096,12288)]+[(n,4096,True) for n in (32,64)]
        for n,k,fused in cases:
            original=torch.randint(0,256,(n*2 if fused else n,k//2),device='cuda',dtype=torch.uint8)
            scales=torch.rand(original.shape[0],k//32,device='cuda')*.001
            if fused:scales=scales.bfloat16()
            x=torch.zeros(1,k,device='cuda',dtype=torch.bfloat16)
            q=(torch.randint(-128,128,(k,),device='cuda',dtype=torch.int8),torch.rand(k//32,device='cuda')*.1)
            kind='up' if fused else 'out' if k==4096 else 'down'
            expected=reference(x,pack_interleaved(original),scales,**SELECTED[kind],paired=fused,fused=fused,scale_mode=4 if fused else 0,prequantized=q)
            for m in (8,16):
                packed=pack_weight(original,m)
                for _ in range(2):actual=linear(x,packed,scales,m=m,warps=8,unroll=4,fused=fused,prequantized=q)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):actual=linear(x,packed,scales,m=m,warps=8,unroll=4,fused=fused,prequantized=q)
                graph.replay();stream.synchronize()
                assert all(torch.equal(v,w) for v,w in zip(flatten(actual),flatten(expected),strict=True))
                rows.append({'n':n,'k':k,'fused':fused,'m':m,'exact':True});del graph
    torch.cuda.current_stream().wait_stream(stream)
    path.write_text(json.dumps({'cases':rows,'all_exact':True,'private_graph':True},indent=2)+'\n');print('PASS',len(rows),flush=True)


if __name__=='__main__':main()
