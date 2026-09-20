"""Save selected projection intermediates for cross-compiler arithmetic audits."""
import argparse
import json
import torch
import triton
from .common import RESULTS
from .benchmark_attention_pdl import load_layer
from .bulk_address import configured
from .dp4a_norm_pdl import SELECTED


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    folder=RESULTS/f'qkv_compiler_{a.tag}';folder.mkdir(exist_ok=False)
    torch.set_num_threads(4);e=load_layer(2);d=e['raw'];fn=configured('norm')
    (result,intermediates),k=fn(d['x'][:1],d['residual'][:1],d['weight'],d['eps'],*e['weights']['qkv'],
        **SELECTED['qkv'],debug=True,return_kernel=True)
    tensors=dict(zip(('summed','qkv','normalized','quantized','scales'),(*result,*intermediates),strict=True))
    torch.save({n:t.cpu() for n,t in tensors.items()},folder/'outputs.pt')
    for suffix in ('ptx','ttgir','llir','cubin'):
        value=k.asm[suffix];(folder/('kernel.'+suffix)).write_bytes(value if isinstance(value,bytes) else value.encode())
    (folder/'manifest.json').write_text(json.dumps({'torch':torch.__version__,'triton':triton.__version__,'layer':2,'input':0},indent=2)+'\n')
    print('SAVED',folder,flush=True)


if __name__=='__main__':main()
