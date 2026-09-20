"""Quantized GEMV against independently materialized FP32 weights, on CUDA graphs."""
import json
import torch
from .common import RESULTS
from .kernels import quantized_linear_decode


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(123)
    results=[]
    for mode in ('fp8','int8','int4'):
        for n,k in ((24576,4096),(4096,12288),(6144,4096),(4096,4096)):
            x=torch.randn(1,1,k,device='cuda',dtype=torch.bfloat16)
            fallback=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*0.02
            if mode=='int4':
                signed=torch.randint(-7,8,(n,k),device='cuda',dtype=torch.int8)
                unsigned=signed.to(torch.uint8)&15
                weight=(unsigned[:,::2]|(unsigned[:,1::2]<<4)).contiguous()
                scale=torch.rand(n,k//128,device='cuda')*0.01
                reference_weight=(signed.float().reshape(n,-1,128)*scale[:,:,None]).reshape(n,k)
            else:
                dtype=torch.int8 if mode=='int8' else torch.float8_e4m3fn
                weight=(torch.randn(n,k,device='cuda')*20).to(dtype)
                scale=torch.rand(n,device='cuda')*0.003
                reference_weight=weight.float()*scale[:,None]
            reference=torch.nn.functional.linear(x.float(),reference_weight)
            for tiled in (False,True):
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):quantized_linear_decode(x,weight,scale,fallback,tiled)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):output=quantized_linear_decode(x,weight,scale,fallback,tiled)
                with torch.cuda.stream(stream):
                    graph.replay()
                    rel=((output.float()-reference).square().mean().sqrt()/reference.square().mean().sqrt()).item()
                torch.cuda.current_stream().wait_stream(stream)
                # BF16 output rounding alone is about 0.17% RMS.
                assert rel<0.003,(mode,n,k,tiled,rel)
                torch.cuda.synchronize()
                original=x.clone()
                x.zero_();graph.replay()
                assert torch.count_nonzero(output).item()==0
                x.copy_(original)
                prefill=x.expand(1,2,k).contiguous()
                assert torch.equal(quantized_linear_decode(prefill,weight,scale,fallback,tiled),torch.nn.functional.linear(prefill,fallback))
                results.append({'mode':mode,'shape':[n,k],'tiled':tiled,'relative_rms_vs_fp32':rel,
                    'graph_replay':True,'non_default_stream':True,'zero_input_replay':True,'bf16_prefill_exact':True})
                del graph,output
            del reference_weight,reference,weight,scale,fallback,x
            print(mode,n,k,'passed',flush=True)
    (RESULTS/'weight_kernel_validation.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
