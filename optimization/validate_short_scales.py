"""All-layer and padded private-graph checks for exact BF16 scale storage."""
import argparse
import json
import torch
from .common import RESULTS
from .short_scales import PLAN
from .dp4a_layout import linear
from .dp4a_scaled import linear as reference,SELECTED
from .dp4a_packing import pack_interleaved
from .benchmark_group128 import quantize


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('all','memory'));p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'short_scales_{a.mode}_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);torch.manual_seed(998);rows=[]
    for name,k in (('out',4096),('down',12288)):
        for layer in (range(36) if a.mode=='all' else (1,3,4,5,37,4095,4096,4097)):
            if a.mode=='all':
                saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
                w=pack_interleaved(saved['packed']);s=saved['scales'].float();short=saved['scales'].bfloat16()
                states=torch.load(RESULTS/f'calibration_v1/{layer:02d}_{name}.pt',weights_only=True)
                inputs=[(str(i),states[i:i+1].cuda()) for i in (0,77,155,231,310,399,528,650,760,865,1012,1186)]
                inputs.extend((f'random_{scale}',(torch.randn(1,k,device='cuda')*scale).bfloat16()) for scale in (.001,.1,10,1000))
                zero=torch.zeros(1,k,device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
                inputs.extend((('zero',zero),('spike',spike)))
            else:
                w=pack_interleaved(torch.randint(0,256,(layer,k//2),device='cuda',dtype=torch.uint8))
                short=(torch.rand(layer,k//32,device='cuda')*.001).bfloat16();s=short.float()
                inputs=[('random',torch.randn(1,k,device='cuda',dtype=torch.bfloat16))]
            assert torch.equal(short.float(),s)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for label,x in inputs:
                    q=quantize(x,32);expected=reference(x,w,s,prequantized=q,**SELECTED[name])
                    if a.mode=='memory':
                        for _ in range(2):linear(x,w,short,prequantized=q,**PLAN[name])
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):actual=linear(x,w,short,prequantized=q,**PLAN[name])
                        graph.replay()
                    else:actual=linear(x,w,short,prequantized=q,**PLAN[name])
                    count=int((actual!=expected).sum())
                    rows.append({'projection':name,'layer_or_rows':layer,'input':label,'mismatches':count})
                    assert count==0,rows[-1]
            torch.cuda.current_stream().wait_stream(stream)
            print(name,layer,'PASS',flush=True)
    result={'cases':rows,'all_exact':True,'mode':a.mode,'private_stream':True,'cuda_graph':a.mode=='memory','codebooks':32}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),flush=True)


if __name__=='__main__':main()
