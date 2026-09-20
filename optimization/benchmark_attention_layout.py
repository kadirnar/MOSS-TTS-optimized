"""Frozen-input and synthetic comparisons of explicit attention layouts."""
import argparse
import json
import torch
import triton
from .common import RESULTS
from .compiler_audit import cuda
from .attention_layout import launch
from .kernels import _decode_attn
from .tune_attention import time_cold_graph


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    torch.set_num_threads(4);torch.manual_seed(9271)
    folder=RESULTS/'compiler_capture_v1'
    manifest=json.loads((folder/'manifest.json').read_text())
    frozen=[cuda(torch.load(folder/r['file'],weights_only=True)) for r in manifest['records'] if r['kind']=='attention']
    q=torch.randn(32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,8,1024,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    pos=torch.zeros(1,device='cuda',dtype=torch.int64)
    flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    configs=[(qd,od,w) for qd in (False,True) for od in (False,True) for w in (1,2,4,8)]
    records=[]
    for qd,od,w in configs:
        row={'q_direct':qd,'out_direct':od,'warps':w,'frozen_checks':[],'synthetic_checks':[],'timings':[]}
        for i,d in enumerate(frozen):
            part=torch.empty_like(d['part']);lse=torch.empty_like(d['lse'])
            launch(d['q'],d['k'],d['v'],d['position'],part,lse,q_direct=qd,out_direct=od,warps=w)
            row['frozen_checks'].append({'index':i,'part_mismatches':int((part!=d['part']).sum()),'lse_mismatches':int((lse!=d['lse']).sum())})
        for cap in (128,256,512,1024):
            splits=cap//32
            part=torch.empty(32,splits,128,device='cuda');lse=torch.empty(32,splits,device='cuda')
            ref=torch.empty_like(part);rlse=torch.empty_like(lse)
            def baseline():_decode_attn[(32,splits)](q,k,v,pos,ref,rlse,1024,splits,32,num_warps=4)
            def candidate():return launch(q,k,v,pos,part,lse,q_direct=qd,out_direct=od,warps=w)
            for p in sorted(set([0,1,31,32,cap//2,cap-1])):
                pos.fill_(p);baseline()
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):candidate()
                torch.cuda.current_stream().wait_stream(stream)
                row['synthetic_checks'].append({'capacity':cap,'position':p,'part_mismatches':int((part!=ref).sum()),'lse_mismatches':int((lse!=rlse).sum())})
            row['timings'].append({'capacity':cap,'baseline_us':time_cold_graph(baseline,flush),'candidate_us':time_cold_graph(candidate,flush)})
        compiled=candidate()
        row.update(registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared)
        row['frozen_exact']=all(not c['part_mismatches'] and not c['lse_mismatches'] for c in row['frozen_checks'])
        row['synthetic_exact']=all(not c['part_mismatches'] and not c['lse_mismatches'] for c in row['synthetic_checks'])
        records.append(row)
        print({k:v for k,v in row.items() if not k.endswith('_checks')},flush=True)
        if qd and od and w==4:
            for ext in ('ptx','ttgir'):(RESULTS/f'attention_layout_{args.tag}.{ext}').write_text(compiled.asm[ext])
    (RESULTS/f'attention_layout_{args.tag}.json').write_text(json.dumps({'torch':torch.__version__,'triton':triton.__version__,'records':records},indent=2)+'\n')


if __name__=='__main__':main()
