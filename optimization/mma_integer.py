"""Independent integer-dot validation for the experimental INT8 MMA mapping."""
import ctypes
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from .common import RESULTS
from .dp4a_packing import pack_interleaved
from .dp4a_mma import pack_mma,unpack_mma


@functools.lru_cache(None)
def library():
    source=Path(__file__).with_suffix('.cu');tag=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder=RESULTS/'mma_integer_build';folder.mkdir(exist_ok=True)
    so=folder/f'integer_{tag}.so';snapshot=folder/f'integer_{tag}.cu'
    if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    if not so.exists():
        command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_90','--shared','-Xcompiler','-fPIC','-Xptxas=-v',str(source),'-o',str(so)]
        result=subprocess.run(command,text=True,capture_output=True)
        (folder/f'integer_{tag}.log').write_text(result.stdout+result.stderr);result.check_returncode()
    lib=ctypes.CDLL(str(so));fn=lib.launch_integer_groups
    fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
    return lib


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(5139);library()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        weight=torch.arange(-8,8,device='cuda',dtype=torch.int8).repeat_interleave(256)
        activation=torch.arange(-128,128,device='cuda',dtype=torch.int16).repeat(16).to(torch.int8)
        ws=[];qs=[]
        for lane in range(33):
            w=torch.zeros(4096,32,device='cuda',dtype=torch.int8);q=w.clone()
            if lane==32:w[:]=weight[:,None];q[:]=activation[:,None]
            else:w[:,lane]=weight;q[:,lane]=activation
            ws.append(w);qs.append(q)
        ws.append(torch.randint(-8,8,(65536,32),device='cuda',dtype=torch.int8))
        qs.append(torch.randint(-128,128,(65536,32),device='cuda',dtype=torch.int8))
        w0=torch.cat(ws);q=torch.cat(qs).contiguous();w=torch.stack((w0,w0.flip(-1)))
        groups=q.shape[0];codes=w.reshape(2,-1).to(torch.uint8)&15
        packed=pack_mma(pack_interleaved(codes[:,::2]|(codes[:,1::2]<<4)))
        assert torch.equal(unpack_mma(packed),w.reshape(2,-1))
        expected=(w.int()*q.int()[None]).sum(-1).int();actual=torch.empty_like(expected)
        def launch():
            code=library().launch_integer_groups(packed.data_ptr(),q.data_ptr(),actual.data_ptr(),groups,stream.cuda_stream)
            if code:raise RuntimeError(f'CUDA error {code}')
        launch();stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):launch()
        graph.replay();stream.synchronize()
        assert torch.equal(actual,expected),int((actual!=expected).sum())
        result={'groups':groups,'dots':actual.numel(),'exact':True,'packing_roundtrip_exact':True,'private_stream':True,'cuda_graph':True,'method':'All 16 signed INT4 weights x 256 signed INT8 activations in each of 32 positions, then all positions together; 65536 random groups; a second reversed weight row shares activations. Independent Torch INT32 product/sum reference.'}
        (RESULTS/'mma_integer_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
    torch.cuda.current_stream().wait_stream(stream)


if __name__=='__main__':main()
