"""Independent grid-search and GPTQ feedback checks for optional group scales."""
import argparse
import json

import torch

from .common import RESULTS
from .calibration_scales import search_scales
from .calibrated_quant import gptq


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'scale_search_validation_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    rows=[];torch.set_num_threads(4);torch.manual_seed(391)
    for group in (32,128):
        for mode in ('mse','diagonal'):
            for kind in ('random','outlier','zero'):
                w=torch.randn(17,256,device='cuda',dtype=torch.bfloat16)*.1
                if kind=='outlier':w[:,3]=3
                if kind=='zero':w.zero_()
                x=torch.randn(80,256,device='cuda',dtype=torch.bfloat16);x[:,3]*=20
                actual=search_scales(w,x,group,mode)
                value=w.float().view(17,256//group,group);maximum=value.abs().amax(-1).clamp_min(1e-8)
                importance=x.float().square().mean(0).view(1,256//group,group) if mode=='diagonal' else torch.ones_like(value)
                best=torch.full_like(maximum,float('inf'));expected=torch.zeros_like(maximum)
                # Tensor division preserves correctly rounded division. Torch's
                # Python-scalar /7 uses reciprocal multiplication and changed
                # one BF16 tie in the initial independent-reference attempt.
                divisor=torch.full_like(maximum,7.)
                for i in range(33):
                    s=((maximum*(1-i/64))/divisor).bfloat16().float()
                    q=(value/s[:,:,None]).round().clamp(-7,7)
                    error=(q*s[:,:,None]).bfloat16().float()-value
                    loss=((error*error)*importance).sum(-1)
                    update=loss<best;expected=torch.where(update,s,expected);best=torch.minimum(loss,best)
                diff=int((actual!=expected.bfloat16()).sum())
                rows.append({'group':group,'mode':mode,'input':kind,'mismatches':diff});assert diff==0,rows[-1]
    checks=[]
    for mode in ('mse','diagonal'):
        w=torch.randn(12,256,device='cuda',dtype=torch.bfloat16);x=torch.randn(96,256,device='cuda',dtype=torch.bfloat16)
        s=search_scales(w,x,128,mode)
        qa,_=gptq(w,x,128,.1,backend='triton',scales_override=s)
        qb,_=gptq(w,x,128,.1,backend='torch',scales_override=s)
        same=torch.equal(qa,qb);assert same
        checks.append({'mode':mode,'integer_codes_exact':same})
    result={'scale_cases':rows,'gptq_checks':checks,'scope':'Independent Torch grid search, 33 clipping fractions, BF16 scales/reconstruction; random/outlier/zero weights; two GPTQ feedback comparisons.',
            'initial_reference_correction':'Python-scalar division /7 changed one BF16 rounding tie. Tensor division matches explicit Triton div.rn and the float64 diagnostic; saved initial case in scale_search_initial_disagreement.pt. No search-kernel change was made to pass this check.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':main()
