"""Memcheck workload for the selected fused gate/up quantizer and padding."""
import torch
from .dp4a_gateup_quant import gateup_quant
from .dp4a_packing import pack_interleaved
from .benchmark_gateup_quant import quantize


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(197);cases=0
    for n,k in ((32,32),(96,96),(160,160),(12288,4096)):
        x=torch.randn(1,k,device='cuda',dtype=torch.bfloat16)
        packed=pack_interleaved(torch.randint(0,256,(n*2,k//2),device='cuda',dtype=torch.uint8))
        scales=(torch.rand(n*2,k//32,device='cuda')*.005).bfloat16()
        qx=quantize(x)
        for rows in (32,64):
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):gateup_quant(x,packed,scales,rows=rows,warps=4,direct=True,prequantized=qx,scale_mode=4)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):out,(q,s)=gateup_quant(x,packed,scales,rows=rows,warps=4,direct=True,prequantized=qx,scale_mode=4)
                graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            assert out.numel()==n and q.numel()==n and s.numel()==n//32
            assert torch.isfinite(out).all() and torch.isfinite(s).all() and (s>=1e-8).all()
            rq,rs=quantize(out)
            assert torch.equal(q,rq) and torch.equal(s,rs)
            cases+=1
    torch.cuda.synchronize();print(f'{cases} private-stream graph and padding checks passed.',flush=True)


if __name__=='__main__':main()
