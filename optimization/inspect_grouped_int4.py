"""Capture compiler resource/assembly evidence for the slow SIMT INT4 trial."""
import json
import torch
from .grouped_int4 import _grouped_int4
from .common import RESULTS

x=torch.ones(4096,device='cuda',dtype=torch.bfloat16)
w=torch.zeros(24576,2048,device='cuda',dtype=torch.uint8)
s=torch.ones(24576,32,device='cuda')
y=torch.empty(24576,device='cuda',dtype=torch.bfloat16)
rows=[]
for warps in (1,2,4,8,16):
    compiled=_grouped_int4[(24576,)](x,w,s,y,4096,128,32,num_warps=warps)
    rows.append({'warps':warps,'registers':compiled.n_regs,'spills':compiled.n_spills,'shared':compiled.metadata.shared})
    if warps in (4,16):
        (RESULTS/f'grouped_int4_w{warps}.ptx').write_text(compiled.asm['ptx'])
torch.cuda.synchronize()
(RESULTS/'grouped_int4_resources.json').write_text(json.dumps(rows,indent=2)+'\n')
print(rows)
