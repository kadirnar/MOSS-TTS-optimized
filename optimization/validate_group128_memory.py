"""Private-stream graph memory checks with independent integer-dot references."""
import argparse
import json

import torch
import triton

from .common import RESULTS
from .calibrated_backend import unpack_signed
from .dp4a_packing import pack_interleaved
from .dp4a_group128 import linear, reduce_attention
from .dp4a_group128_norm import linear as norm_linear
from .dp4a_fusions import norm_quant
from .kernels import _decode_attn_reduce, silu_mul
from .benchmark_group128 import quantize


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'group128_memory_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);torch.manual_seed(139);records=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for ag in (32,128):
            plan=json.loads((RESULTS/f'group128_a{ag}_plan_v1.json').read_text())
            for name,k,n2 in (('qkv',4096,5),('out',4096,37),('down',12288,73),('up',4096,256)):
                x=torch.randn(1,k,device='cuda',dtype=torch.bfloat16)
                packed=torch.randint(0,256,(n2,k//2),device='cuda',dtype=torch.uint8)
                weight=pack_interleaved(packed);scales=(torch.rand(n2,k//128,device='cuda')*.01).bfloat16()
                q,sx=quantize(x,ag);paired=name=='up'
                cfg=plan['projections'][name]
                for _ in range(3):linear(x,weight,scales,prequantized=(q,sx),activation_group=ag,paired=paired,**cfg)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):actual=linear(x,weight,scales,prequantized=(q,sx),activation_group=ag,paired=paired,**cfg)
                graph.replay()
                codes=unpack_signed(packed).int()
                dots=(codes*q.int()[None,:]).view(n2,k//ag,ag).sum(-1,dtype=torch.int32)
                expected=(dots.float()*(scales.float().repeat_interleave(128//ag,dim=1)*sx[None,:])).sum(-1).bfloat16().reshape(1,-1)
                if paired:expected=silu_mul(expected)
                value=actual[0] if isinstance(actual,tuple) else actual
                error=((value.float()-expected.float()).square().mean()/expected.float().square().mean().clamp_min(1e-20)).sqrt().item()
                assert error<.001,(ag,name,error)
                records.append({'kind':'projection','activation_group':ag,'projection':name,'n2':n2,'relative_rms_vs_independent':error})
            for splits in (4,8,16,32):
                part=torch.randn(32,splits,128,device='cuda');lse=torch.randn(32,splits,device='cuda')
                part[:,splits//2:]=0;lse[:,splits//2:]=-torch.inf
                for _ in range(3):reduce_attention(part,lse)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):actual,qx=reduce_attention(part,lse)
                graph.replay();expected=torch.empty_like(actual)
                _decode_attn_reduce[(32,)](part,lse,expected,splits,triton.next_power_of_2(splits),num_warps=4)
                eq=quantize(expected,128)
                assert torch.equal(actual,expected) and all(torch.equal(x,y) for x,y in zip(qx,eq))
                records.append({'kind':'attention','splits':splits,'output_and_quant_exact':True})
        plan=json.loads((RESULTS/'group128_a32_fused_plan_v2.json').read_text())
        for name in ('qkv','up'):
            for add in (False,True):
                cfg=plan['norm_projections'][name];n2=74 if name=='qkv' else 256
                packed=torch.randint(0,256,(n2,2048),device='cuda',dtype=torch.uint8)
                weight=pack_interleaved(packed);scales=(torch.rand(n2,32,device='cuda')*.01).bfloat16()
                x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16);res=torch.randn_like(x) if add else None
                nw=torch.randn_like(x).flatten();kwargs=dict(fused=name=='up',activation_group=32,**cfg)
                for _ in range(3):norm_linear(x,res,nw,1e-6,weight,scales,**kwargs)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):actual=norm_linear(x,res,nw,1e-6,weight,scales,debug=True,**kwargs)
                graph.replay();summed,normed,qx=norm_quant(x,res,nw,1e-6,32,8)
                assert torch.equal(actual[0][0],summed)
                assert all(torch.equal(x,y) for x,y in zip(actual[1],(normed,*qx)))
                if cfg.get('output_quant'):
                    expected_q=quantize(actual[0][1],32)
                    assert all(torch.equal(x,y) for x,y in zip(actual[0][2],expected_q))
                records.append({'kind':'norm_projection','projection':name,'add':add,'producer_and_output_quant_exact':True})
    torch.cuda.current_stream().wait_stream(stream)
    result={'private_stream':True,'cuda_graph':True,'codebooks':32,'cases':records}
    path.write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':main()
