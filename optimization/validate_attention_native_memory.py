"""Small native-only memory/race sanitizer workload, including graph replay."""
import torch
from .attention_native import launch,library


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(91);library()
    q=torch.randn(32,128,device='cuda',dtype=torch.bfloat16)
    original_k=torch.randn(1,8,1024,128,device='cuda',dtype=torch.bfloat16)
    original_v=torch.randn_like(original_k)
    pos=torch.zeros(1,device='cuda',dtype=torch.int64)
    cases=0
    for capacity in (128,256,512,1024):
        part=torch.empty(32,capacity//32,128,device='cuda');lse=torch.empty(32,capacity//32,device='cuda')
        for position in (0,31,32,capacity-1):
            k=original_k.clone();v=original_v.clone();pos.fill_(position)
            # Invalid future slots are poison, not initialized benign zeros.
            k[:,:,position+1:]=float('nan');v[:,:,position+1:]=float('nan')
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                launch(q,k,v,pos,part,lse)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):launch(q,k,v,pos,part,lse)
                graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            assert torch.isfinite(part).all()
            active=position//32+1
            assert torch.isfinite(lse[:,:active]).all()
            assert (lse[:,active:]==float('-inf')).all()
            assert (part[:,active:]==0).all()
            if position==0:
                expected=original_v[0,:,0].float().repeat_interleave(4,dim=0)
                assert torch.equal(part[:,0],expected)
            cases+=1
    torch.cuda.synchronize()
    print(f'{cases} native private-stream CUDA-graph poison-boundary checks passed.',flush=True)


if __name__=='__main__':main()
