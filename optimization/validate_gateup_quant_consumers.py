"""Check the fused epilogue's consumed quantized values and down-projection output.

The BF16 gate/up buffer is retained for shape/API compatibility, but a downstream
DP4A projection supplied with prequantized input does not read its values.
"""
import json
import argparse
import torch
import triton
from .common import RESULTS
from .dp4a_packing import pack_interleaved,linear as reference
from .dp4a_direct import linear as down
from .dp4a_gateup_quant import gateup_quant,_gateup_quant
from .benchmark_gateup_quant import quantize


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scale-mode',type=int,default=0);args=parser.parse_args()
    suffix=f'_scale{args.scale_mode}' if args.scale_mode else ''
    torch.set_num_threads(4);records=[]
    indices=(0,37,116,231,352,463,528,671,829,1007,1186,1275)
    for layer in range(36):
        up=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{layer:02d}_up.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(up['packed']);s=up['scales'].bfloat16()
        dw=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{layer:02d}_down.pt',map_location='cuda',weights_only=True)
        wd=pack_interleaved(dw['packed']);sd=dw['scales']
        xs=torch.load(RESULTS/'calibration_v1'/f'{layer:02d}_up.pt',weights_only=True)
        zero=torch.zeros(1,4096,device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
        inputs=[(str(i),xs[i:i+1].cuda()) for i in indices]+[('zero',zero),('spike',spike)]
        for label,x in inputs:
            qx=quantize(x)
            expected=reference(x,w,s,rows=4,warps=4,interleaved=True,paired=True,prequantized=qx)
            eq,es=quantize(expected)
            expected_down=down(expected,wd,sd,rows=4,warps=2,prequantized=(eq,es))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual,(aq,ascale)=gateup_quant(x,w,s,rows=32,warps=4,direct=True,prequantized=qx,scale_mode=args.scale_mode)
                actual_down=down(actual,wd,sd,rows=4,warps=2,prequantized=(aq,ascale))
            torch.cuda.current_stream().wait_stream(stream)
            row={'layer':layer,'input':label,'auxiliary_bf16_mismatches':int((actual!=expected).sum()),
                'quant_mismatches':int((aq!=eq).sum()),'scale_mismatches':int((ascale!=es).sum()),
                'down_mismatches':int((actual_down!=expected_down).sum())}
            records.append(row)
        print('layer',layer,'consumed mismatches',sum(r['quant_mismatches']+r['scale_mismatches']+r['down_mismatches'] for r in records if r['layer']==layer),flush=True)
        if layer==0:
            kernel=_gateup_quant[(384,)](*qx,w,s,actual,aq,ascale,12288,4096,128,32,True,args.scale_mode,num_warps=4)
            resources={'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared}
            for ext in ('ptx','ttgir'):(RESULTS/f'gateup_quant_selected{suffix}.{ext}').write_text(kernel.asm[ext])
        del up,dw,w,s,wd,sd,xs,zero,spike,inputs,x,qx,expected,eq,es,expected_down,actual,aq,ascale,actual_down
    result={'rows':records,'checks':len(records),'private_stream':True,'resources':resources,
        'all_consumed_tensors_exact':all(not r['quant_mismatches'] and not r['scale_mismatches'] and not r['down_mismatches'] for r in records),
        'auxiliary_bf16_mismatches':sum(r['auxiliary_bf16_mismatches'] for r in records),
        'scope':'36 layers x (12 recorded BF16 activations + zero + spike). Grouped INT8 activations, FP32 scales and final BF16 down-projection outputs. Auxiliary BF16 gate/up values are not consumed when prequantized inputs are supplied.'}
    result['scale_mode']=args.scale_mode
    (RESULTS/f'gateup_quant_consumer_validation{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':main()
