"""Native FP6-LLM CUDA kernels on actual shapes, including BF16 conversions."""
import json
import subprocess
import torch
import fp6_llm
from .common import RESULTS
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(364)
    result={'torch':torch.__version__,'source':'https://github.com/usyd-fsalab/fp6_llm',
        'source_commit':subprocess.check_output(['git','-C','/workspace/fp6_llm','rev-parse','HEAD'],text=True).strip(),
        'build_change':'Native sm90 cubin instead of upstream sm80; all kernel code unchanged.',
        'method':'FP6_e3m2 weights, FP16 internal activations, BF16 input/output conversions included. Eight-matrix HBM rotation and graph timing; non-default-stream validation. Synthetic packed FP6 values, actual model projection shapes.',
        'cases':[]}
    for n,k in ((24576,4096),(4096,12288),(6144,4096),(4096,4096)):
        packed=torch.randint(0,2**32-1,(n,k//16*3),dtype=torch.int64).int()
        scale=(torch.rand(n)*.003+.0001).half()
        ref=fp6_llm.weight_dequant_cpu(packed,scale).cuda()
        weights=[fp6_llm.weight_prepacking_cpu(packed).cuda()]
        weights.extend(weights[0].clone() for _ in range(7))
        scale=scale.cuda()
        x=torch.randn(1,k,device='cuda',dtype=torch.bfloat16)
        expected=torch.nn.functional.linear(x.half(),ref).bfloat16()
        trials=[]
        for split in (1,2,4,8,16,32):
            def fn(w):return fp6_llm.linear_forward_cuda(x.half(),w,scale,split).bfloat16()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):actual=fn(weights[0])
            torch.cuda.current_stream().wait_stream(stream)
            rel=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
            assert rel<.005 and torch.isfinite(actual).all(),(n,k,split,rel)
            trials.append({'split_k':split,'us':measure(fn,weights),'relative_rms_vs_dequantized_fp16':rel})
        case={'shape':[n,k],'trials':trials,'best':min(trials,key=lambda t:t['us'])}
        result['cases'].append(case)
        print(case,flush=True)
        (RESULTS/'fp6_kernels.json').write_text(json.dumps(result,indent=2)+'\n')
        del ref,weights,packed,scale,x,actual,expected


if __name__=='__main__':main()
