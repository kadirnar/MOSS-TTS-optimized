"""Cache/output equivalence and cold-L2 timing of fused QK/attention."""
import json
from types import SimpleNamespace
import torch
from .common import RESULTS
from .kernels import qk_rope_decode
from .fused_attention import fused_qk_attention
from .tune_attention import time_cold_graph


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(892)
    qkv=torch.randn(1,1,6144,device='cuda',dtype=torch.bfloat16)
    qw=torch.randn(128,device='cuda',dtype=torch.bfloat16)
    kw=torch.randn_like(qw)
    phase=torch.randn(64,device='cuda').repeat(2)
    cos=phase.cos().bfloat16();sin=phase.sin().bfloat16()
    pos=torch.zeros(1,device='cuda',dtype=torch.long)
    k=torch.randn(1,8,1024,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    k2=k.clone();v2=v.clone()
    cache=SimpleNamespace(layers=[SimpleNamespace(keys=k,values=v)])
    flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    results={'torch':torch.__version__,'method':'Non-default stream output/KV comparison; median of 20 graph replays with an untimed 256 MiB L2 flush. Synthetic QKV and past cache.','cases':[]}
    stream=torch.cuda.Stream()
    for block in (32,128):
        for p in (0,1,31,32,127,144,255,511,1023):
            pos.fill_(p)
            old=lambda:qk_rope_decode(qkv,qw,kw,cos,sin,cache,0,pos,1e-6,block=block)
            new=lambda:fused_qk_attention(qkv,qw,kw,cos,sin,k2,v2,pos,1e-6,block)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                expected=old();actual=new()
            torch.cuda.current_stream().wait_stream(stream)
            assert torch.equal(k,k2) and torch.equal(v,v2),'KV differs'
            rel=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
            assert rel<.003,(block,p,rel)
            row={'block':block,'position':p,'relative_rms':rel,'bitwise_equal':bool(torch.equal(actual,expected)),
                'cache_exact':True,'baseline_us':time_cold_graph(old,flush),'fused_us':time_cold_graph(new,flush)}
            results['cases'].append(row)
            print(row,flush=True)
            (RESULTS/'fused_qk_attention.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
