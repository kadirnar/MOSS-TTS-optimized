"""G64-weight/G32-activation chain screen against current selected G32."""
import argparse
import json
import statistics

import torch

from .common import RESULTS
from .benchmark_projection_pdl import load_layer,graph_edges
from .benchmark_bulk_address import Chain as SelectedChain
from .benchmark_norm_projection import flatten
from .dp4a_packing import pack_interleaved
from .dp4a_norm_pdl import SELECTED
from .short_scales import PLAN
from .dp4a_group64 import norm_linear,linear
from .tune_weight_reads import measure


def load(layer):
    entry=load_layer(layer);entry['g64']={};entry['repeat']={}
    for name in ('up','down','qkv'):
        index=(layer+1)%36 if name=='qkv' else layer
        saved=torch.load(RESULTS/f'gptq_v1_g64_diag_d10/{index:02d}_{name}.pt',map_location='cuda',weights_only=True)
        assert saved['group']==64
        w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16()
        entry['g64'][name]=(w,s);entry['repeat'][name]=(w,s.repeat_interleave(2,dim=1))
    return entry


class Chain:
    def __init__(self,mode,factors=None,tiles=None):
        self.mode=mode;self.factors=factors or {};self.tiles=tiles or {};self.selected=SelectedChain({'qkv':{'address_mode':1}})

    def __call__(self,e,index=0,audit=False):
        if self.mode in ('control','repeat'):
            value=e if self.mode=='control' else dict(e,weights=e['repeat'])
            return self.selected(value,index,audit=audit)
        d=e['raw'];nd=e['next'];w=e['g64'];kernels={}
        options={**SELECTED['up'],**self.tiles.get('up',{})}
        up,k=norm_linear(d['x'][index:index+1],d['residual'][index:index+1] if d['has_residual'] else None,
            d['weight'],d['eps'],*w['up'],fused=True,factor=self.factors.get('up',0),**options,return_kernel=True)
        kernels['up']=k;summed,hidden,quantized=up
        options={**PLAN['down'],**self.tiles.get('down',{})}
        down,k=linear(hidden,*w['down'],prequantized=quantized,divisor=16,factor=self.factors.get('down',0),**options,return_kernel=True)
        kernels['down']=k
        options={**SELECTED['qkv'],**self.tiles.get('qkv',{})}
        following,k=norm_linear(down,summed,nd['weight'],nd['eps'],*w['qkv'],factor=self.factors.get('qkv',0),**options,return_kernel=True)
        kernels['qkv']=k
        return ((up,down,following),kernels) if audit else (up,down,following)


def configs(pilot=False):
    output={'control':{'mode':'control'},'repeat':{'mode':'repeat'},'g64_ordered':{'mode':'g64'},
        'g64_factored':{'mode':'g64','factors':{n:1 for n in ('up','down','qkv')}}}
    if pilot:return output
    for name in ('up','down','qkv'):output[name+'_factored']={'mode':'g64','factors':{name:1}}
    for name,tiles in (('qkv',[{'rows':4},{'rows':16}]),('up',[{'rows':64},{'integer_groups':1},{'integer_rows':2}]),
                       ('down',[{'rows':2},{'rows':8},{'warps':4}])):
        for i,tile in enumerate(tiles):
            output[f'{name}_tile{i}']={'mode':'g64','tiles':{name:tile}}
    return output


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--layers',type=int,choices=(1,36),default=36)
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--pilot',action='store_true');args=p.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag) or args.rounds<2:raise ValueError('Safe tag and two rounds required')
    path=RESULTS/f'group64_pdl_{args.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);ring=[load(i) for i in (range(36) if args.layers==36 else (17,))]
    choices=configs(args.pilot);chains={};result={'codebooks':32,'weight_group':64,'activation_group':32,'layers':args.layers,
        'configs':choices,'checks':[],'resources':{},'errors':{},'rows':[],
        'method':'Selected G32 control versus G64-weight/G32-activation MLP/down/next-QKV chains. Repeated G64 scales on original kernels provide arithmetic reference. The initial pilot exposed a BF16 intermediate mismatch even without explicit factoring; both variants require separate numerical and speech qualification. Three frozen inputs per layer plus private graph. Rotating layer ring; not TTFA or speech-quality evidence.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        reference=Chain('repeat')
        for name,config in choices.items():
            try:
                candidate=Chain(**config)
                for e in ring:
                    for index in (0,10,31):
                        expected=reference(e,index);actual=candidate(e,index)
                        counts=[int((x!=y).sum()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]
                        errors=[float(((x.float()-y.float()).square().mean()/y.float().square().mean().clamp_min(1e-20)).sqrt()) for x,y in zip(flatten(actual),flatten(expected),strict=True)]
                        result['checks'].append({'config':name,'layer':e['layer'],'input':index,'mismatches':counts,'relative_rms':errors})
                _,kernels=candidate(ring[0],audit=True)
                result['resources'][name]={stage:{'registers':k.n_regs,'spills':k.n_spills,'shared_bytes':k.metadata.shared} for stage,k in kernels.items()}
                for _ in range(2):candidate(ring[0])
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):actual=candidate(ring[0])
                edges=graph_edges(graph);assert len(edges)==2 and all(e['type']==1 for e in edges)
                graph.replay();expected=candidate(ring[0])
                exact=all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(flatten(actual),flatten(expected),strict=True));assert exact
                result['checks'].append({'config':name,'layer':ring[0]['layer'],'input':'eager_graph','mismatches':[0],'relative_rms':[0.0]})
                rows=[x for x in result['checks'] if x['config']==name]
                if name not in ('control','repeat'):
                    assert max(max(row['relative_rms']) for row in rows)<.002,'Large error against repeated-scale reference'
                print('CHECKED',name,'max_relative_rms',max(max(x['relative_rms']) for x in rows),flush=True);chains[name]=candidate
            except Exception as error:result['errors'][name]=repr(error);print('ERROR',name,repr(error),flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(args.rounds):
        order=list(chains);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        times={name:measure(chains[name],ring) for name in order};result['rows'].append({'round':repeat,'order':order,'us':times});save();print('ROUND',repeat,flush=True)
    result['summary']={name:{'median_us':statistics.median(row['us'][name] for row in result['rows']),
        'median_paired_gain_us':statistics.median(row['us']['control']-row['us'][name] for row in result['rows']),
        'max_relative_rms':max(max(row['relative_rms']) for row in result['checks'] if row['config']==name)} for name in chains}
    save();print('BEST',sorted(result['summary'],key=lambda name:result['summary'][name]['median_us'])[:6],flush=True)


if __name__=='__main__':main()
