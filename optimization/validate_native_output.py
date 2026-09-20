"""All-layer real-input and private-stream graph checks for native output GEMV."""
import argparse
import json
import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_scaled import linear as reference,SELECTED
from .dp4a_output_native import linear,library
from .benchmark_gateup_quant import quantize


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--memory-only',action='store_true');p.add_argument('--tag',default='v1');args=p.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe nonempty tag required')
    torch.set_num_threads(4);torch.manual_seed(177);library();records=[]
    if not args.memory_only:
        for layer in range(36):
            saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{layer:02d}_out.pt',map_location='cuda',weights_only=True)
            w=pack_interleaved(saved['packed']);s=saved['scales'].float()
            xs=torch.load(RESULTS/f'calibration_v1/{layer:02d}_out.pt',weights_only=True)
            zero=torch.zeros(1,4096,device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
            inputs=[(str(i),xs[i:i+1].cuda()) for i in (0,37,116,231,352,463,528,671,829,1007,1186,1275)]+[('zero',zero),('spike',spike)]
            for label,x in inputs:
                qx=quantize(x);expected=reference(x,w,s,**SELECTED['out'],prequantized=qx)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):actual=linear(x,w,s,prequantized=qx)
                torch.cuda.current_stream().wait_stream(stream)
                mismatch=int((actual!=expected).sum());records.append({'layer':layer,'input':label,'mismatches':mismatch})
            print(layer,sum(r['mismatches'] for r in records if r['layer']==layer),flush=True)
    graph_rows=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for n in (1,3,4,5,37,4095,4096,4097):
            w=pack_interleaved(torch.randint(0,256,(n,2048),device='cuda',dtype=torch.uint8))
            s=torch.rand(n,128,device='cuda')*.01;x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16);qx=quantize(x)
            for _ in range(3):linear(x,w,s,prequantized=qx)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):actual=linear(x,w,s,prequantized=qx)
            graph.replay();expected=reference(x,w,s,**SELECTED['out'],prequantized=qx);stream.synchronize()
            mismatch=int((actual!=expected).sum());graph_rows.append({'n':n,'mismatches':mismatch})
    torch.cuda.current_stream().wait_stream(stream)
    exact=all(not r['mismatches'] for r in records+graph_rows)
    result={'checks':len(records),'records':records,'graphs':graph_rows,'all_exact':exact,'private_stream':True,'memory_only':args.memory_only}
    (RESULTS/f'native_output_validation_{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='records'},flush=True)
    assert exact


if __name__=='__main__':main()
