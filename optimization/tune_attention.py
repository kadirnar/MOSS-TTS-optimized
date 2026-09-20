"""Attention split/warp sweep, with L2 flushed before each timed graph replay."""
import json
import statistics
import torch
import triton
from .common import RESULTS
from .kernels import _decode_attn,_decode_attn_reduce


def time_cold_graph(fn,flush):
    for _ in range(3):fn()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn()
    times=[]
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for _ in range(20):
        flush.zero_()  # 256 MiB; excluded from timed interval.
        start.record();graph.replay();end.record();end.synchronize()
        times.append(start.elapsed_time(end)*1000)
    return statistics.median(times)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(123)
    q=torch.randn(32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,8,1024,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    pos=torch.zeros(1,device='cuda',dtype=torch.long)
    flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    results={'torch':torch.__version__,'triton':triton.__version__,
        'method':'Median of 20 GPU-event timed graph replays, each preceded by an untimed 256 MiB L2 flush','cases':[]}
    for p in (127,255,511,1023):
        pos.fill_(p)
        ref=torch.nn.functional.scaled_dot_product_attention(q.view(1,32,1,128),k[:,:,:p+1],v[:,:,:p+1],enable_gqa=True).view(1,1,4096)
        for block in (16,32,64,128,256):
            splits=1024//block
            partial=torch.empty(32,splits,128,device='cuda')
            lse=torch.empty(32,splits,device='cuda')
            out=torch.empty(1,1,4096,device='cuda',dtype=torch.bfloat16)
            for warps in (4,8):
                def fn():
                    _decode_attn[(32,splits)](q,k,v,pos,partial,lse,1024,splits,block,num_warps=warps)
                    _decode_attn_reduce[(32,)](partial,lse,out,splits,triton.next_power_of_2(splits),num_warps=4)
                fn()
                rel=((out.float()-ref.float()).square().mean().sqrt()/ref.float().square().mean().sqrt()).item()
                assert rel<0.01,(p,block,warps,rel)
                d={'position':p,'block':block,'warps':warps,'us':time_cold_graph(fn,flush),'relative_rms_vs_sdpa':rel}
                results['cases'].append(d)
                print(d,flush=True)
                (RESULTS/'attention_tiles.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
