"""Native attention: captured activations, cache boundaries, streams and timing."""
import argparse
import json
import torch
import triton
from .common import RESULTS
from .compiler_audit import cuda
from .attention_native import launch,library
from .attention_quant import reduce_quant
from .kernels import _decode_attn
from .tune_attention import time_cold_graph


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    torch.set_num_threads(4);torch.manual_seed(9271);library()
    folder=RESULTS/'compiler_capture_v1';manifest=json.loads((folder/'manifest.json').read_text())
    records=[];timings=[]
    def check(label,q,k,v,pos,ref,rlse):
        part=torch.empty_like(ref);lse=torch.empty_like(rlse)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):launch(q,k,v,pos,part,lse)
        torch.cuda.current_stream().wait_stream(stream)
        actual,(aq,asc)=reduce_quant(part,lse);expected,(eq,esc)=reduce_quant(ref,rlse)
        row={'label':label,'partial_mismatches':int((part!=ref).sum()),'lse_mismatches':int((lse!=rlse).sum()),
            'max_partial_abs':float((part-ref).abs().max()),'output_mismatches':int((actual!=expected).sum()),
            'quant_mismatches':int((aq!=eq).sum()),'scale_mismatches':int((asc!=esc).sum())}
        records.append(row);print(row,flush=True)
    for r in manifest['records']:
        if r['kind']!='attention':continue
        d=cuda(torch.load(folder/r['file'],weights_only=True))
        check(r['label'],d['q'],d['k'],d['v'],d['position'],d['part'],d['lse'])
    q=torch.randn(32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,8,1024,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    pos=torch.zeros(1,device='cuda',dtype=torch.int64)
    flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    for cap in (128,256,512,1024):
        splits=cap//32;ref=torch.empty(32,splits,128,device='cuda');rlse=torch.empty(32,splits,device='cuda')
        part=torch.empty_like(ref);lse=torch.empty_like(rlse)
        def baseline():_decode_attn[(32,splits)](q,k,v,pos,ref,rlse,1024,splits,32,num_warps=4)
        def candidate():launch(q,k,v,pos,part,lse)
        for p in sorted(set([0,1,31,32,cap//2,cap-1])):
            pos.fill_(p);baseline();check(f'cap{cap}_pos{p}',q,k,v,pos,ref,rlse)
        timings.append({'capacity':cap,'baseline_cold_us':time_cold_graph(baseline,flush),'native_cold_us':time_cold_graph(candidate,flush),
            'baseline_warm_us':triton.testing.do_bench_cudagraph(baseline)*1000,'native_warm_us':triton.testing.do_bench_cudagraph(candidate)*1000})
        print(timings[-1],flush=True)
    result={'torch':torch.__version__,'triton':triton.__version__,'private_stream':True,'records':records,'timings':timings}
    (RESULTS/f'attention_native_{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
