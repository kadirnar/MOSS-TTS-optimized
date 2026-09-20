"""Additional native CUDA INT4 trials: vLLM ExLlama and PyTorch tinygemm.

This tests kernels, not GPTQ calibration: both use symmetric round-to-nearest.
"""
import json
import torch
from .common import RESULTS
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    import vllm._C_stable_libtorch
    torch.set_num_threads(4)
    torch.manual_seed(128)
    result={'torch':torch.__version__,'codebooks_unchanged':32,'cases':[]}
    for backend in ('exllama','tinygemm'):
        for group in (128,32):
            for n,k in ((24576,4096),(4096,12288),(6144,4096),(4096,4096)):
                case={'backend':backend,'group':group,'shape':[n,k]}
                try:
                    weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*0.02
                    groups=weight.float().reshape(n,-1,group)
                    scales=(groups.abs().amax(-1).clamp_min(1e-8)/7).bfloat16()
                    signed=(groups/scales.float()[...,None]).round().clamp(-7,7)
                    reference=(signed*scales.float()[...,None]).reshape(n,k).bfloat16()
                    q=(signed.reshape(n,k)+8).to(torch.int32)
                    x=torch.randn(1,k,device='cuda',dtype=torch.bfloat16)
                    if backend=='exllama':
                        packed=torch.zeros((k//8,n),device='cuda',dtype=torch.int32)
                        for i in range(8):packed.bitwise_or_(q.T[i::8]<<(i*4))
                        empty=torch.empty(0,device='cuda',dtype=torch.int32)
                        torch.ops._C.gptq_shuffle(packed,empty,4)
                        zeros=torch.full((k//group,n//8),0x77777777,device='cuda',dtype=torch.int32)
                        native_scales=scales.T.contiguous().half()
                        def fn(w):
                            # Include required conversions in the measured graph.
                            return torch.ops._C.gptq_gemm(x.half(),w,zeros,native_scales,empty,True,False,4).bfloat16()
                    else:
                        bytes_=((q[:,0::2]<<4)|q[:,1::2]).to(torch.uint8)
                        packed=torch._convert_weight_to_int4pack(bytes_,8)
                        native_scales=torch.stack((scales.T,torch.zeros_like(scales.T)),dim=-1).contiguous()
                        def fn(w):return torch._weight_int4pack_mm(x,w,group,native_scales)
                    expected=torch.nn.functional.linear(x,reference)
                    actual=fn(packed)
                    rel=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
                    case['relative_rms_vs_dequantized_bf16']=rel
                    case['kernel_check_passed']=rel<0.005
                    if case['kernel_check_passed']:
                        matrices=[packed]+[packed.clone() for _ in range(7)]
                        case['us']=measure(fn,matrices)
                        del matrices
                    del weight,groups,scales,signed,reference,q,x,packed,actual,expected
                except (RuntimeError,AttributeError,NotImplementedError) as error:
                    case['error']=str(error)[:1000]
                result['cases'].append(case)
                print(case,flush=True)
                (RESULTS/'int4_native_kernels.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
