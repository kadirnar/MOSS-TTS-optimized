"""Repeat the only near-winning native/layout output-projection candidates."""
import ctypes
import json
import statistics
import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_scaled import linear as reference,SELECTED
from .dp4a_layout import linear as layout
from . import dp4a_staged
from .benchmark_gateup_quant import quantize
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    # Preserved independent-dot, both-sums-and-coefficients shared variant.
    path=RESULTS/'dp4a_staged_build/dp4a_ca3428a6bed244c6.so'
    lib=ctypes.CDLL(str(path));fn=lib.launch_dp4a_staged
    fn.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*6+[ctypes.c_void_p];fn.restype=ctypes.c_int
    dp4a_staged.library=lambda:lib
    saved=torch.load(RESULTS/'gptq_v1_g32_d10/00_out.pt',map_location='cuda',weights_only=True)
    w=pack_interleaved(saved['packed']);s=saved['scales'].float()
    x=torch.load(RESULTS/'calibration_v1/00_out.pt',weights_only=True)[231:232].cuda();qx=quantize(x)
    ring=[(w,s)]+[(w.clone(),s.clone()) for _ in range(7)]
    methods={
        'control':lambda pair:reference(x,*pair,**SELECTED['out'],prequantized=qx),
        'staged':lambda pair:dp4a_staged.linear(x,*pair,rows=4,warps=8,prequantized=qx),
        'layout':lambda pair:layout(x,*pair,rows=4,warps=4,integer_groups=1,integer_rows=1,prequantized=qx),
    }
    expected=methods['control'](ring[0])
    for method in methods.values():assert torch.equal(method(ring[0]),expected)
    rows=[]
    for i in range(12):
        order=list(methods);order=order[i%3:]+order[:i%3]
        if i%2:order.reverse()
        row={'round':i,'order':order,'us':{name:measure(methods[name],ring) for name in order}}
        rows.append(row);print(row,flush=True)
    medians={name:statistics.median(r['us'][name] for r in rows) for name in methods}
    gains={name:statistics.median((r['us']['control']-r['us'][name])/r['us']['control']*100 for r in rows) for name in ('staged','layout')}
    result={'method':'Twelve rotating/reversing rounds on the same eight-weight ring; each entry median of nine 24-kernel CUDA graph replays. FP32 output-projection scales.','rows':rows,'median_us':medians,'median_paired_gain_percent':gains,'native_source':str(path.with_suffix('.cu')),'promotion_threshold_percent':2.0}
    (RESULTS/'native_projection_repeat_v1.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':main()
