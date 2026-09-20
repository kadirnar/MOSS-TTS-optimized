"""Integer exhaustive/edge checks plus private-stream graph memory validation."""
import json
import torch
import triton
import triton.language as tl
from .common import RESULTS
from .dp4a_direct import _direct_dot,linear as old_linear
from .dp4a_scaled import _dot,linear
from .dp4a_gateup_quant import gateup_quant
from .dp4a_packing import pack_interleaved
from .benchmark_gateup_quant import quantize


@triton.jit
def integer_check(W,Q,Y,Z,N:tl.constexpr,MODE:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    w=tl.load(W+i,i<N,0)
    # Last CTA deliberately includes out-of-range activation pointers. The
    # assembly's predicated load must suppress every inactive lane.
    p=tl.cast(Q,tl.pointer_type(tl.uint64))+i
    actual=_dot(w,p,i<N,MODE)
    if MODE==2:actual=actual>>4
    expected=_direct_dot(w,p,i<N)
    tl.store(Y+i,actual,i<N);tl.store(Z+i,expected,i<N)


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(931);stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Every 4-bit weight / signed-byte activation pair, in every one of
        # eight independent lanes, then all lanes together; plus random dots.
        weights=torch.arange(-8,8,device='cuda',dtype=torch.int32).repeat_interleave(256)
        activations=torch.arange(-128,128,device='cuda',dtype=torch.int32).repeat(16)
        ws=[];qs=[]
        for lane in range(9):
            a=torch.zeros(4096,8,device='cuda',dtype=torch.int32);b=a.clone()
            if lane==8:a[:]=weights[:,None];b[:]=activations[:,None]
            else:a[:,lane]=weights;b[:,lane]=activations
            ws.append(a);qs.append(b)
        ws.append(torch.randint(-8,8,(65537,8),device='cuda',dtype=torch.int32))
        qs.append(torch.randint(-128,128,(65537,8),device='cuda',dtype=torch.int32))
        wvalues=torch.cat(ws);qvalues=torch.cat(qs);n=wvalues.shape[0]
        nib=wvalues&15;words=nib[:,:4]|(nib[:,4:]<<4)
        shifts=torch.arange(4,device='cuda',dtype=torch.int64)*8
        packed=(words.long()<<shifts).sum(1).to(torch.uint32)
        q=qvalues.to(torch.int8).contiguous();expected=(wvalues*qvalues).sum(1).int()
        integer=[]
        for mode in (1,2):
            actual=torch.empty(n,device='cuda',dtype=torch.int32);old=actual.clone()
            integer_check[(triton.cdiv(n,256),)](packed,q,actual,old,n,mode,256)
            stream.synchronize()
            assert torch.equal(actual,expected) and torch.equal(old,expected)
            integer.append({'mode':mode,'dots':n,'exact':True})
        rows=[]
        for fused,n,k,r,warps in [(False,5,32,4,4),(False,37,96,4,4),(False,65,160,4,2),(False,4096,12288,4,2),(True,32,32,32,4),(True,96,96,64,4),(True,160,160,32,4),(True,12288,4096,32,4)]:
            count=n*2 if fused else n
            w=pack_interleaved(torch.randint(0,256,(count,k//2),device='cuda',dtype=torch.uint8))
            s=(torch.rand(count,k//32,device='cuda')*.005).bfloat16() if fused or k<4096 else torch.rand(count,k//32,device='cuda')*.005
            x=torch.randn(1,k,device='cuda',dtype=torch.bfloat16);qx=quantize(x)
            for mode in (1,2):
                def run():return linear(x,w,s,rows=r,warps=warps,mode=mode,paired=fused,fused=fused,scale_mode=4 if fused else 0,prequantized=qx)
                for _ in range(3):run()
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):actual=run()
                graph.replay();stream.synchronize()
                if fused:
                    output,aq=actual;expected_q=quantize(output)
                    stream.synchronize();assert all(torch.equal(a,b) for a,b in zip(aq,expected_q))
                    expected=gateup_quant(x,w,s,rows=r,warps=warps,prequantized=qx)
                    stream.synchronize();assert torch.equal(output,expected[0])
                else:
                    expected=old_linear(x,w,s,rows=r,warps=warps,prequantized=qx)
                    stream.synchronize();assert torch.equal(actual,expected)
                rows.append({'fused':fused,'n':n,'k':k,'rows':r,'warps':warps,'mode':mode,'exact':True})
    torch.cuda.current_stream().wait_stream(stream)
    result={'integer':integer,'graphs':rows,'all_exact':True,'private_stream':True}
    (RESULTS/'dp4a_scaled_memory_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
