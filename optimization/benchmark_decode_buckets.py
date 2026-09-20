"""Attention output/cache checks at context bucket boundaries and cold timing."""
import json
from types import SimpleNamespace
import torch
from .common import RESULTS
from .kernels import qk_rope_decode
from .tune_attention import time_cold_graph


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(324)
    qkv=torch.randn(1,1,6144,device='cuda',dtype=torch.bfloat16)
    qw=torch.randn(128,device='cuda',dtype=torch.bfloat16);kw=torch.randn_like(qw)
    phase=torch.randn(64,device='cuda').repeat(2);cos=phase.cos().bfloat16();sin=phase.sin().bfloat16()
    k=torch.randn(1,8,1024,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    k2=k.clone();v2=v.clone()
    cache=SimpleNamespace(layers=[SimpleNamespace(keys=k,values=v)])
    other=SimpleNamespace(layers=[SimpleNamespace(keys=k2,values=v2)])
    flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    position=torch.zeros(1,device='cuda',dtype=torch.long)
    rows=[]
    for p in (0,1,31,32,126,127,128,144,176,254,255,256,510,511,512,1023):
        cap=min(c for c in (128,256,512,1024) if p<c)
        position.fill_(p)
        baseline=lambda:qk_rope_decode(qkv,qw,kw,cos,sin,cache,0,position,1e-6,block=32)
        bucketed=lambda:qk_rope_decode(qkv,qw,kw,cos,sin,other,0,position,1e-6,block=32,context_capacity=cap)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):expected=baseline();actual=bucketed()
        torch.cuda.current_stream().wait_stream(stream)
        assert torch.equal(k,k2) and torch.equal(v,v2),'KV writes differ'
        relative=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
        assert relative<.003,(p,cap,relative)
        row={'position':p,'capacity':cap,'relative_rms':relative,'exact':torch.equal(actual,expected),'kv_exact':True,
             'baseline_us':time_cold_graph(baseline,flush),'bucket_us':time_cold_graph(bucketed,flush)}
        rows.append(row);print(row,flush=True)
    (RESULTS/'decode_bucket_kernels.json').write_text(json.dumps({'torch':torch.__version__,'private_stream':True,'cases':rows},indent=2)+'\n')


if __name__=='__main__':main()
