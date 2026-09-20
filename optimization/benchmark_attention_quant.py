"""Exact split-output/quantizer checks and fused operator timings."""
import json
import argparse
import torch
import triton
from .common import RESULTS
from .kernels import _decode_attn,_decode_attn_reduce
from .int4_dp4a import _int8_activation_grouped
from .attention_quant import reduce_quant,_reduce_quant
from .tune_attention import time_cold_graph


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='');args=parser.parse_args()
    if args.tag and not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Unsafe tag')
    suffix='_'+args.tag if args.tag else ''
    torch.set_num_threads(4);torch.manual_seed(738)
    q=torch.randn(32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,8,1024,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    pos=torch.zeros(1,device='cuda',dtype=torch.int64)
    out=torch.empty(1,1,4096,device='cuda',dtype=torch.bfloat16)
    qi=torch.empty(4096,device='cuda',dtype=torch.int8);scale=torch.empty(128,device='cuda')
    flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    records=[];timings=[]
    for capacity in (128,256,512,1024):
        splits=capacity//32
        part=torch.empty(32,splits,128,device='cuda');lse=torch.empty(32,splits,device='cuda')
        def reference():
            _decode_attn_reduce[(32,)](part,lse,out,splits,triton.next_power_of_2(splits),num_warps=4)
            _int8_activation_grouped[(32,)](out,qi,scale,4096,32,4,num_warps=4)
            return out,(qi,scale)
        for position in sorted(set([0,1,31,32,capacity//2,capacity-1])):
            pos.fill_(position)
            _decode_attn[(32,splits)](q,k,v,pos,part,lse,1024,splits,32,num_warps=4)
            reference()
            for warps in (1,2,4,8):
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):actual,(aq,ascale)=reduce_quant(part,lse,warps)
                torch.cuda.current_stream().wait_stream(stream)
                row={'capacity':capacity,'position':position,'warps':warps,'output_exact':torch.equal(out,actual),
                    'quant_exact':torch.equal(qi,aq),'scale_exact':torch.equal(scale,ascale),
                    'output_mismatches':int((out!=actual).sum()),'quant_mismatches':int((qi!=aq).sum())}
                records.append(row)
                error=((out.float()-actual.float()).square().mean().sqrt()/out.float().square().mean().sqrt()).item()
                row['relative_rms']=error
                assert error<.003 and torch.isfinite(actual).all(),row
        timing={'capacity':capacity,'baseline_us':time_cold_graph(reference,flush),'fused_us':{}}
        for warps in (1,2,4,8):timing['fused_us'][str(warps)]=time_cold_graph(lambda:reduce_quant(part,lse,warps),flush)
        timings.append(timing);print(timing,flush=True)
    eligible=[w for w in (1,2,4,8) if all(r['output_exact'] and r['quant_exact'] and r['scale_exact'] for r in records if r['warps']==w)]
    result={'checks':records,'timings':timings,'eligible_warps':eligible,'method':'Cold-L2 CUDA graph operator timing. Quantizer output, FP32 scale and BF16 attention output compared exactly on a private stream.'}
    result['torch']=torch.__version__;result['triton']=triton.__version__
    (RESULTS/f'attention_quant{suffix}_kernels.json').write_text(json.dumps(result,indent=2)+'\n')
    compiled=_reduce_quant[(32,)](part,lse,out,qi,scale,32,32,num_warps=4)
    for ext in ('ptx','ttgir'):(RESULTS/f'attention_quant{suffix}.{ext}').write_text(compiled.asm[ext])
    print('Checks',len(records),'eligible',eligible,'registers',compiled.n_regs,'spills',compiled.n_spills,'shared',compiled.metadata.shared,flush=True)


if __name__=='__main__':main()
