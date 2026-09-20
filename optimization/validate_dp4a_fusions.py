"""Exact producer/quantizer checks and graph timing for DP4A input fusion."""
import argparse
import json
import torch
import triton
from .common import RESULTS
from .kernels import rmsnorm,add_rmsnorm,silu_mul
from .int4_dp4a import _int8_activation,_int8_activation_grouped
from .dp4a_fusions import norm_quant,silu_quant
from .tune_weight_reads import measure


def quant(x,group=0):
    k=x.numel()
    q=torch.empty(k,device=x.device,dtype=torch.int8)
    s=torch.empty(k//group if group else 1,device=x.device,dtype=torch.float32)
    if group:_int8_activation_grouped[(triton.cdiv(k//group,4),)](x,q,s,k,group,4,num_warps=4)
    else:_int8_activation[(1,)](x,q,s,k,triton.next_power_of_2(k),True)
    return q,s


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',type=int,choices=(0,32,128),default=0)
    parser.add_argument('--layout-limit',type=int,choices=(0,8),default=0)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(8349)
    cases=[]
    for kind in ('norm','residual_norm','silu'):
        k=12288 if kind=='silu' else 4096
        for magnitude in (0.,.01,1.,10.):
            x=(torch.randn(1,1,k*(2 if kind=='silu' else 1),device='cuda')*magnitude).bfloat16()
            r=torch.randn(1,1,k,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(k,device='cuda',dtype=torch.bfloat16)
            def reference(x):
                if kind=='norm':y=rmsnorm(x,w,1e-6);s=x
                elif kind=='residual_norm':s,y=add_rmsnorm(x,r,w,1e-6)
                else:y=silu_mul(x);s=None
                return s,y,quant(y,args.group)
            def fused(x):
                if kind=='silu':
                    y,q=silu_quant(x,args.group)
                    return None,y,q
                return norm_quant(x,r if kind=='residual_norm' else None,w,1e-6,args.group,args.layout_limit)
            expected=reference(x)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):actual=fused(x)
            torch.cuda.current_stream().wait_stream(stream)
            if expected[0] is not None:assert torch.equal(expected[0],actual[0]),(kind,'residual')
            assert torch.equal(expected[1],actual[1]),(kind,'activation',magnitude)
            assert torch.equal(expected[2][0],actual[2][0]),(kind,'int8',magnitude)
            assert torch.equal(expected[2][1],actual[2][1]),(kind,'scale',magnitude)
            row={'kind':kind,'magnitude':magnitude,'bf16_and_int8_and_scale':'exact'}
            if magnitude==1.:
                row['reference_us']=measure(reference,[x]*8)
                row['fused_us']=measure(fused,[x]*8)
            cases.append(row)
            print(row,flush=True)
    name='dp4a_fusion_validation'+(f'_g{args.group}' if args.group else '')+(f'_layout{args.layout_limit}' if args.layout_limit else '')
    (RESULTS/(name+'.json')).write_text(json.dumps({'torch':torch.__version__,'group':args.group,'layout_limit':args.layout_limit,
        'private_stream':True,'graph_timing':True,'cases':cases},indent=2)+'\n')


if __name__=='__main__':main()
