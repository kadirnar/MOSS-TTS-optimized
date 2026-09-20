"""All-layer check of the wider QKV tile before any packing-plan change."""
import hashlib
import json
import torch
import triton
from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_direct import linear,_direct_gemv
from .benchmark_gateup_quant import quantize


@torch.inference_mode()
def main():
    torch.set_num_threads(4);rows=[]
    for layer in range(36):
        saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{layer:02d}_qkv.pt',map_location='cuda',weights_only=True)
        w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16()
        xs=torch.load(RESULTS/'calibration_v1'/f'{layer:02d}_qkv.pt',weights_only=True)
        zero=torch.zeros(1,4096,device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
        for label,x in [(str(i),xs[i:i+1].cuda()) for i in (0,37,116,231,352,463,528,671,829,1007,1186,1275)]+[('zero',zero),('spike',spike)]:
            qx=quantize(x);expected=linear(x,w,s,rows=4,warps=4,prequantized=qx)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):actual=linear(x,w,s,rows=8,warps=8,prequantized=qx)
            torch.cuda.current_stream().wait_stream(stream)
            rows.append({'layer':layer,'input':label,'mismatches':int((actual!=expected).sum())})
        if layer==0:
            compiled=_direct_gemv[(768,)](*qx,w,s,actual,6144,4096,128,8,False,num_warps=8)
            resources={'registers':compiled.n_regs,'spills':compiled.n_spills,'shared_bytes':compiled.metadata.shared}
            for ext in ('ptx','ttgir'):(RESULTS/f'dp4a_wide_qkv.{ext}').write_text(compiled.asm[ext])
        print('layer',layer,'mismatches',sum(r['mismatches'] for r in rows if r['layer']==layer),flush=True)
        del saved,w,s,xs,x,qx,zero,spike,actual,expected
    result={'checks':len(rows),'all_exact':all(not r['mismatches'] for r in rows),'rows':rows,'resources':resources,'private_stream':True}
    path=RESULTS/'dp4a_wide_qkv_validation.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='rows'},flush=True)
    # Only emit an experimental plan after all exact checks pass. Full-model
    # speed, quality and streaming validation must still precede selection.
    if result['all_exact']:
        plan=json.loads((RESULTS/'dp4a_direct_exact_plan.json').read_text())
        plan['projections']['qkv'].update(rows=8,warps=8)
        plan['wide_qkv_validation_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        plan['selection']='Experimental QKV R8/W8 after 504 exact checks; full-model selection recorded separately.'
        (RESULTS/'dp4a_wide_qkv_plan.json').write_text(json.dumps(plan,indent=2)+'\n')


if __name__=='__main__':main()
