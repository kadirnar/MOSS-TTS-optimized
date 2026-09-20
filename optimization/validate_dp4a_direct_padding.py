"""Sanitizer-sized odd-row and padded-K checks for inline PTX memory loads."""
import json
import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved,linear as reference
from .dp4a_direct import linear


@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.manual_seed(31)
    rows=[]
    for k in (32,96,160,384):
        for paired in (False,True):
            n=14 if paired else 7
            packed=torch.randint(0,256,(n,k//2),device='cuda',dtype=torch.uint8)
            packed=pack_interleaved(packed)
            scales=torch.rand(n,k//32,device='cuda').bfloat16().float()
            x=torch.randn(1,k,device='cuda',dtype=torch.bfloat16)
            expected=reference(x,packed,scales,rows=4,warps=2,interleaved=True,paired=paired)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):actual=linear(x,packed,scales,rows=4,warps=2,paired=paired)
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(actual,expected,rtol=.001,atol=1e-5)
            assert torch.isfinite(actual).all()
            rows.append({'k':k,'output_rows':7,'paired':paired,'exact':torch.equal(actual,expected)})
    torch.cuda.synchronize()
    (RESULTS/'dp4a_direct_padding.json').write_text(json.dumps({'checks':rows,'stream':'private CUDA stream'},indent=2)+'\n')
    print('Passed',len(rows),'odd-row/padded-K cases',flush=True)


if __name__=='__main__':main()
