"""Prefill QKV preparation across lengths, offsets and changed graph inputs."""
import argparse
import json
import math

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .common import RESULTS
from .kernels import rmsnorm
from .prefill_qkv import project


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'prefill_qkv_validation_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.manual_seed(49205);checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    qw=torch.randn(128,device='cuda',dtype=torch.bfloat16);kw=torch.randn_like(qw);eps=1e-6
    with torch.cuda.stream(stream):
        for n in (2,3,16,127,128,145,160,256,511,512):
            x=torch.randn((1,n,6144),device='cuda',dtype=torch.bfloat16)
            for offset in sorted(set((0,96,1024-n))):
                pos=torch.arange(offset,offset+n,device='cuda')
                phase=pos.float()[:,None]*torch.exp(-math.log(1000000)*torch.arange(64,device='cuda')/64)[None,:]
                angles=torch.cat((phase,phase),-1)[None]
                cos=angles.cos().bfloat16();sin=angles.sin().bfloat16()
                keys=torch.empty((1,8,1024,128),device='cuda',dtype=torch.bfloat16);values=torch.empty_like(keys)
                ref_keys=torch.empty_like(keys);ref_values=torch.empty_like(keys)
                for _ in range(2):project(x,qw,kw,cos,sin,keys,values,pos,eps)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):actual=project(x,qw,kw,cos,sin,keys,values,pos,eps)
                for probe in ('random','zero','spike'):
                    if probe=='zero':x.zero_()
                    elif probe=='spike':x.zero_();x[...,0]=32;x[...,4096]=-16;x[...,5120]=7
                    else:x.normal_()
                    for t in (keys,values,ref_keys,ref_values):t.fill_(float('nan'))
                    graph.replay()
                    q,k,v=x.split((4096,1024,1024),-1)
                    q=rmsnorm(q.reshape(1,n,32,128),qw,eps).transpose(1,2)
                    k=rmsnorm(k.reshape(1,n,8,128),kw,eps).transpose(1,2)
                    q,k=apply_rotary_pos_emb(q,k,cos,sin)
                    ref_keys.index_copy_(2,pos,k);ref_values.index_copy_(2,pos,v.reshape(1,n,8,128).transpose(1,2))
                    exact=all(torch.equal(u.view(torch.int16),v.view(torch.int16)) for u,v in ((q,actual),(keys,ref_keys),(values,ref_values)))
                    assert q.stride()==actual.stride() and exact,(n,offset,probe)
                    checks.append({'tokens':n,'offset':offset,'probe':probe,'all_q_kv_bits_exact':True,'strides_exact':True})
            print('CHECKED',n,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result={'codebooks':32,'all_exact':True,'cases':len(checks),'checks':checks,
        'scope':'Private-stream CUDA graphs with changed random/zero/spike input; ten lengths, ordinary/suffix/final-capacity positions. All Q bits/strides and entire KV storage, including NaN-poisoned unused slots, match separate norm/rotary/index_copy reference.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(checks),flush=True)


if __name__=='__main__':main()
