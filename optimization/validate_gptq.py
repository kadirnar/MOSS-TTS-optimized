"""Compare fused offline quantization to a straightforward Torch algorithm."""
import json
import torch
from .common import RESULTS
from .calibrated_quant import gptq,dequantize,static_scales


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(479)
    result=[]
    for k in (128,256,512):
        w=(torch.randn(48,k,device='cuda')*.04).bfloat16()
        latent=torch.randn(768,24,device='cuda')
        x=(latent@torch.randn(24,k,device='cuda')+torch.randn(768,k,device='cuda')*.2).bfloat16()
        for actorder in (False,True):
            expected,es=gptq(w,x,actorder=actorder,backend='torch')
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):actual,scales=gptq(w,x,actorder=actorder)
            torch.cuda.current_stream().wait_stream(stream)
            agreement=(expected==actual).float().mean().item()
            assert agreement>.999,(k,actorder,agreement)
            assert torch.equal(es,scales)
            q=dequantize(actual,scales,32)
            rtn=(w.float().view(48,-1,32)/scales.float()[:,:,None]).round().clamp(-7,7).to(torch.int8).reshape_as(w)
            baseline=dequantize(rtn,scales,32)
            target=torch.nn.functional.linear(x.float(),w.float())
            loss=lambda qw:((torch.nn.functional.linear(x.float(),qw.float())-target).square().mean()/target.square().mean()).item()
            row={'columns':k,'actorder':actorder,'code_agreement':agreement,'gptq_mse':loss(q),'rtn_mse':loss(baseline)}
            assert row['gptq_mse']<row['rtn_mse'],row
            print(row,flush=True);result.append(row)
    (RESULTS/'gptq_algorithm_validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
