"""Independent exhaustive-edge/random INT4 x INT8 integer verification."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .common import RESULTS
from .mma4_projection import pack_groups,integer,library


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'mma4_integer_{a.tag}.json'
    if path.exists():raise FileExistsError('Preserve results')
    torch.set_num_threads(4);torch.manual_seed(221);library()
    rows=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for m in (8,16):
            for pos in list(range(32))+['all','random']:
                if pos=='random':
                    w=torch.randint(-8,8,(65536,m,32),device='cuda',dtype=torch.int8)
                    q=torch.randint(-128,128,(65536,32),device='cuda',dtype=torch.int8)
                else:
                    values=torch.arange(-8,8,device='cuda',dtype=torch.int16).repeat_interleave(256)
                    activations=torch.arange(-128,128,device='cuda',dtype=torch.int16).repeat(16)
                    # Different row values also verify the A fragment row map.
                    weight=((values[:,None]+torch.arange(m,device='cuda')[None]+8)%16-8).to(torch.int8)
                    w=torch.zeros((4096,m,32),device='cuda',dtype=torch.int8);q=torch.zeros((4096,32),device='cuda',dtype=torch.int8)
                    if pos=='all':w[:]=weight[:,:,None];q[:]=activations[:,None].to(torch.int8)
                    else:w[:,:,pos]=weight;q[:,pos]=activations.to(torch.int8)
                packed=pack_groups(w);expected=(w.int()*q[:,None,:].int()).sum(-1).int()
                for _ in range(2):actual=integer(packed,q,m)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):actual=integer(packed,q,m)
                graph.replay();stream.synchronize()
                mismatch=int((actual!=expected).sum());assert mismatch==0,(m,pos,mismatch)
                rows.append({'m':m,'position':pos,'groups':q.shape[0],'dots':actual.numel(),'exact':True})
                del graph,actual,expected,packed,w,q
            print('PASS M',m,flush=True)
    torch.cuda.current_stream().wait_stream(stream)
    result={'cases':rows,'all_exact':True,'dots':sum(r['dots'] for r in rows),'private_stream':True,'cuda_graph':True,
            'native_source_sha256':hashlib.sha256(Path(__file__).with_name('mma4_projection.cu').read_bytes()).hexdigest(),
            'method':'Every signed INT4/INT8 scalar pair in each of 32 positions and all positions simultaneously, row permutations, then 65536 independent random multirow groups. Native low-unsigned/high-signed decomposition versus independent Torch INT32 products/sums.'}
    path.write_text(json.dumps(result,indent=2)+'\n');print('PASS',result['dots'],flush=True)


if __name__=='__main__':main()
