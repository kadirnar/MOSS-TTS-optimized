"""One tagged HBM gate/up projection for Nsight Compute inspection."""
import json
import torch
import triton
from .fused_mlp import _fp8_silu_gemv
from .common import RESULTS

torch.manual_seed(771)
x=torch.randn(4096,device='cuda',dtype=torch.bfloat16)
w=(torch.randn(24576,4096,device='cuda')*25).to(torch.float8_e4m3fn)
s=torch.rand(24576,device='cuda')*.005
y=torch.empty(12288,device='cuda',dtype=torch.bfloat16)
def launch():return _fp8_silu_gemv[(12288,)](x,w,s,y,12288,4096,4096,True,num_warps=4)
kernel=launch()
torch.cuda.synchronize()
flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
flush.zero_()
torch.cuda.synchronize()
with torch.cuda.nvtx.range('profile_weight'):
    launch()
    torch.cuda.synchronize()
metadata={'torch':torch.__version__,'triton':triton.__version__,'registers_per_thread':kernel.n_regs,
    'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared,'weight_bytes':w.numel()*w.element_size(),
    'grid_ctas':12288,'warps_per_cta':4}
(RESULTS/'fp8_gateup_resource_usage.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(metadata)
