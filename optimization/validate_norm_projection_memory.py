"""Padded producer-fusion memory checks on private CUDA graphs."""
import argparse
import json
import torch
import triton
from .common import RESULTS
from .dp4a_norm_projection import linear,_kernel,SELECTED
from .dp4a_scaled import linear as old,SELECTED as OLD
from .dp4a_fusions import norm_quant
from .dp4a_packing import pack_interleaved
from .benchmark_norm_projection import flatten


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    outfile=RESULTS/f'norm_projection_memory_{a.tag}.json'
    if outfile.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);torch.manual_seed(721);rows=[];resources={}
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name in ('qkv','up'):
            fused=name=='up';cfg=SELECTED[name]
            for n in ((5,37,6144) if not fused else (32,96,12288)):
                n2=n*2 if fused else n
                w=pack_interleaved(torch.randint(0,256,(n2,2048),device='cuda',dtype=torch.uint8))
                s=(torch.rand(n2,128,device='cuda')*.01).bfloat16();nw=torch.randn(4096,device='cuda',dtype=torch.bfloat16)
                x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16)
                for add in (False,True):
                    residual=torch.randn_like(x) if add else None
                    for _ in range(3):linear(x,residual,nw,1e-6,w,s,fused=fused,**cfg)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):actual=linear(x,residual,nw,1e-6,w,s,fused=fused,**cfg)
                    graph.replay()
                    summed,normalized,qx=norm_quant(x,residual,nw,1e-6,32,8)
                    out=old(normalized,w,s,**OLD[name],paired=fused,fused=fused,scale_mode=4 if fused else 0,prequantized=qx)
                    expected=(summed,*out) if fused else (summed,out)
                    stream.synchronize();counts=[int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]
                    rows.append({'projection':name,'n':n,'add':add,'mismatches':counts});assert not any(counts),rows[-1]
                    if n in (6144,12288) and add:
                        y=actual[1];oq,os=actual[2] if fused else (torch.empty(0,device='cuda',dtype=torch.int8),torch.empty(0,device='cuda'))
                        ny=torch.empty(0,device='cuda',dtype=x.dtype);nq=torch.empty(0,device='cuda',dtype=torch.int8);ns=torch.empty(0,device='cuda')
                        kernel=_kernel[(triton.cdiv(n,cfg['rows']),)](x,residual,nw,w,s,actual[0],y,oq,os,ny,nq,ns,n,cfg['rows'],1e-6,True,fused,cfg['integer_groups'],cfg['integer_rows'],False,num_warps=4)
                        resources[name]={'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared}
                        for ext in ('ptx','ttgir'):(RESULTS/f'norm_projection_{name}_{a.tag}.{ext}').write_text(kernel.asm[ext])
    torch.cuda.current_stream().wait_stream(stream)
    result={'cases':rows,'all_exact':True,'private_stream':True,'cuda_graph':True,'resources':resources}
    outfile.write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':main()
