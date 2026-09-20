"""Compiler register, spill and shared-memory evidence for selected layouts."""
import json
import torch
import triton
from .common import RESULTS
from .dp4a_packing import _packed_gemv,pack_interleaved


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    rows=[]
    for label in ('','exact_'):
        plan=json.loads((RESULTS/f'dp4a_packing_{label}plan.json').read_text())
        for name,cfg in plan['projections'].items():
            saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'00_{name}.pt',map_location='cuda',weights_only=True)
            w=pack_interleaved(saved['packed']) if cfg['scheme']=='interleaved' else saved['packed']
            s=saved['scales'] if cfg['scale_dtype']=='bfloat16' else saved['scales'].float()
            n,k=saved['shape'];paired=name=='up';out_n=n//2 if paired else n
            q=torch.zeros(k,device='cuda',dtype=torch.int8);sx=torch.ones(k//32,device='cuda')
            out=torch.empty(out_n,device='cuda',dtype=torch.bfloat16)
            compiled=_packed_gemv[(triton.cdiv(out_n,cfg['rows']),)](q,sx,w,s,out,out_n,k,32,
                triton.next_power_of_2(k//32),cfg['rows'],cfg['scheme']=='interleaved',paired,num_warps=cfg['warps'])
            torch.cuda.synchronize()
            tag=('exact' if label else 'fast')+'_'+name
            for ext in ('ptx','ttgir'):(RESULTS/f'packed_{tag}.{ext}').write_text(compiled.asm[ext])
            row={'plan':label or 'fast','projection':name,**cfg,'registers_per_thread':compiled.n_regs,
                 'spills':compiled.n_spills,'shared_bytes':compiled.metadata.shared,
                 'grid_ctas':triton.cdiv(out_n,cfg['rows']),
                 'weight_bytes':w.numel()*w.element_size(),'scale_bytes':s.numel()*s.element_size()}
            rows.append(row);print(row,flush=True)
            del saved,w,s,q,sx,out
    (RESULTS/'packed_kernel_resources.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
