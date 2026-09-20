"""All-layer prefill activation/normalization screen, separate from TTFA."""
import argparse
import json
import statistics

import torch
import torch.nn.functional as F

from .common import RESULTS
from .benchmark_first_audio_graph import build_engine
from .kernels import rmsnorm,add_rmsnorm
from .prefill_pointwise import make_silu_table,silu_mul
from .tune_weight_reads import measure


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--rounds',type=int,default=6);args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag) or args.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'prefill_pointwise_{args.tag}.json';assert not path.exists(),'Preserve evidence'
    engine,_=build_engine(bulk=True,prefill_qkv=True);fast=engine.llm;layers=fast.model.language_model.layers
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);source=fixture['inputs']['input_ids'].cuda()
    ids=torch.full((1,160,33),1024,device='cuda',dtype=torch.long);ids[...,0]=fast.cfg.pad_token_id;ids[:,:source.shape[1]].copy_(source)
    positions=torch.arange(160,device='cuda');last=torch.tensor([source.shape[1]-1],device='cuda')
    states=[{} for _ in layers];hooks=[]
    for i,layer in enumerate(layers):
        def pre_layer(module,a,k,i=i):states[i]['residual']=(a[0] if a else k['hidden_states']).clone()
        def post_attn(module,a,out,i=i):states[i]['attention']=out[0].clone()
        def pre_mlp(module,a,i=i):states[i]['gateup']=F.linear(a[0],module._gate_up)
        def pre_norm(module,a,i=i):states[i]['post_residual']=a[0].clone()
        def post_mlp(module,a,out,i=i):states[i]['mlp']=out.clone()
        hooks.extend((layer.register_forward_pre_hook(pre_layer,with_kwargs=True),layer.self_attn.register_forward_hook(post_attn),
            layer.mlp.register_forward_pre_hook(pre_mlp),layer.post_attention_layernorm.register_forward_pre_hook(pre_norm),layer.mlp.register_forward_hook(post_mlp)))
    try:fast._prefill_forward(ids,positions,last)
    finally:
        for hook in hooks:hook.remove()
    lut=make_silu_table('cuda');ring=[s['gateup'] for s in states]
    functions={'control':lambda x:F.silu(x[...,:12288])*x[...,12288:]};configs={}
    for mode in (0,1,2):
        for block,warps in ((128,4),(256,4),(512,4),(1024,4),(2048,4),(512,8),(1024,8)):
            name=f'm{mode}_b{block}_w{warps}';configs[name]={'mode':mode,'block':block,'warps':warps}
            functions[name]=lambda x,config=configs[name]:silu_mul(x,lut,**config)
    result={'codebooks':32,'tokens':160,'configs':configs,'checks':[],'resources':{},'rows':[],'norm_checks':[],'norm_rows':[],
        'method':'Actual prefill activation states from all 36 layers; rotating layer ring, alternating/rotating configurations. Twenty-one fused SiLU/product alternatives versus separate reference kernels; residual/RMSNorm fusion across 72 actual sites. Operator times are not TTFA.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    # Cover the complete BF16 input domain with several independent up values.
    bits=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    domain=torch.zeros((1,6,24576),device='cuda',dtype=torch.bfloat16);index=torch.arange(65536,device='cuda')
    domain[0,index//12288,index%12288]=bits
    probes=[]
    for scale in (1.0,-2.0,0.0):
        probe=domain.clone();probe[...,12288:]=scale;probes.append(probe)
    for name,fn in list(functions.items()):
        if name=='control':continue
        for i,x in enumerate(ring+probes):
            ref=functions['control'](x);actual=fn(x)
            count=int((ref.view(torch.int16)!=actual.view(torch.int16)).sum())
            result['checks'].append({'config':name,'input':i,'bit_mismatches':count})
        _,compiled=silu_mul(ring[0],lut,**configs[name],return_kernel=True)
        result['resources'][name]={'registers':compiled.n_regs,'spills':compiled.n_spills,'shared_bytes':compiled.metadata.shared}
        count=sum(row['bit_mismatches'] for row in result['checks'] if row['config']==name)
        if count:functions.pop(name)
        save();print('CHECK',name,count,flush=True)
    norm_ring=[]
    for i,(state,layer) in enumerate(zip(states,layers,strict=True)):
        norm=layer.post_attention_layernorm
        norm_ring.append((state['attention'],state['residual'],norm.weight,norm.variance_epsilon))
        norm=layers[i+1].input_layernorm if i+1<len(layers) else fast.model.language_model.norm
        norm_ring.append((state['mlp'],state['post_residual'],norm.weight,norm.variance_epsilon))
    def norm_control(e):
        x,r,w,eps=e;summed=x+r;return summed,rmsnorm(summed,w,eps)
    def norm_candidate(e):return add_rmsnorm(*e)
    for i,e in enumerate(norm_ring):
        ref=norm_control(e);actual=norm_candidate(e);count=sum(int((u.view(torch.int16)!=v.view(torch.int16)).sum()) for u,v in zip(ref,actual,strict=True))
        result['norm_checks'].append({'site':i,'bit_mismatches':count});assert count==0
    for repeat in range(args.rounds):
        order=list(functions);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        timing={name:measure(functions[name],ring) for name in order}
        result['rows'].append({'round':repeat,'order':order,'us':timing})
        names=('control','candidate') if repeat%2==0 else ('candidate','control')
        result['norm_rows'].append({'round':repeat,'us':{name:measure(norm_control if name=='control' else norm_candidate,norm_ring) for name in names}})
        save();print('ROUND',repeat,flush=True)
    result['median_us']={name:statistics.median(row['us'][name] for row in result['rows']) for name in functions}
    result['norm_median_us']={name:statistics.median(row['us'][name] for row in result['norm_rows']) for name in ('control','candidate')}
    save();print('BEST',sorted(result['median_us'].items(),key=lambda x:x[1])[:5],'NORM',result['norm_median_us'],flush=True);engine.codec.close()


if __name__=='__main__':main()
