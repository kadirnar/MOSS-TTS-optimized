"""Ensure fused residual/RMSNorm preserves both BF16 rounding boundaries."""
import json
from pathlib import Path
import torch
from .kernels import rmsnorm,add_rmsnorm

torch.manual_seed(99)
stream=torch.cuda.Stream()
rows=[]
with torch.inference_mode(),torch.cuda.stream(stream):
    for amplitude in (0.001,1.0,1000.0):
        x=torch.randn(1,1,4096,device='cuda',dtype=torch.bfloat16)*amplitude
        residual=torch.randn_like(x)*amplitude
        weight=torch.randn(4096,device='cuda',dtype=torch.bfloat16)
        expected_sum=x+residual
        expected_norm=rmsnorm(expected_sum,weight,1e-6)
        for _ in range(3):add_rmsnorm(x,residual,weight,1e-6)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            actual_sum,actual_norm=add_rmsnorm(x,residual,weight,1e-6)
        graph.replay()
        stream.synchronize()
        assert torch.equal(actual_sum,expected_sum)
        assert torch.equal(actual_norm,expected_norm)
        rows.append({'amplitude':amplitude,'residual_bitwise_equal':True,'norm_bitwise_equal':True})
result={'non_default_cuda_stream':True,'cuda_graph_replay':True,'cases':rows}
Path('optimization/results/residual_fusion_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
