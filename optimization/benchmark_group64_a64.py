"""Screen G64 activation groups, with all 32 acoustic codebooks retained.

Reference projections use the original adjacent-nibble packing and standalone
normalization/quantization. Changed-precision candidates are not quality claims.
"""
import argparse
import json
import statistics

import torch

from .common import RESULTS
from .benchmark_group64_pdl import load as load_base
from .benchmark_bulk_address import Chain as SelectedChain
from .benchmark_projection_pdl import graph_edges
from .benchmark_norm_projection import flatten
from .benchmark_group128 import quantize
from .dp4a_fusions import norm_quant
from .int4_dp4a import int4_dp4a
from .kernels import silu_mul
from .dp4a_norm_pdl import SELECTED
from .short_scales import PLAN
from .dp4a_group64_a64 import norm_linear, linear
from .tune_weight_reads import measure


def load(layer):
    e=load_base(layer);e['original']={}
    for group,folder in ((32,'gptq_v1_g32_d10'),(64,'gptq_v1_g64_diag_d10')):
        e['original'][group]={}
        for name in ('up','down','qkv'):
            i=(layer+1)%36 if name=='qkv' else layer
            d=torch.load(RESULTS/f'{folder}/{i:02d}_{name}.pt',map_location='cuda',weights_only=True)
            e['original'][group][name]=(d['packed'],d['scales'].float())
    return e


class Chain:
    def __init__(self,stages):
        self.stages=stages
        self.control=SelectedChain({'qkv':{'address_mode':1}})

    def __call__(self,e,index=0,audit=False,reference=False):
        d=e['raw'];nd=e['next'];kernels={}
        output_group=self.stages.get('down',{}).get('activation_group',32)

        def norm_stage(name,x,res,nw,eps):
            c=dict(self.stages.get(name,{}));ag=c.pop('activation_group',32)
            wg=64 if name in self.stages else 32
            if reference:
                summed,z,q=norm_quant(x,res,nw,eps,ag,8)
                w,s=e['original'][wg][name]
                y=int4_dp4a(z,w,s.repeat_interleave(wg//ag,dim=1),ag,4,True,q,True)
                return (summed,silu_mul(y),quantize(silu_mul(y),output_group)) if name=='up' else (summed,y)
            if wg==32:
                assert ag==32 and (name!='up' or output_group==32)
                value,k=self.control.functions[name](x,res,nw,eps,*e['weights'][name],
                    fused=name=='up',**SELECTED[name],return_kernel=True)
            else:
                value,k=norm_linear(x,res,nw,eps,*e['g64'][name],fused=name=='up',
                    activation_group=ag,output_group=output_group,
                    **{**SELECTED[name],**c},return_kernel=True)
            kernels[name]=k
            return value

        up=norm_stage('up',d['x'][index:index+1],d['residual'][index:index+1] if d['has_residual'] else None,d['weight'],d['eps'])
        summed,hidden,q=up
        c=dict(self.stages.get('down',{}));ag=c.pop('activation_group',32)
        if reference:
            wg=64 if 'down' in self.stages else 32;w,s=e['original'][wg]['down']
            down=int4_dp4a(hidden,w,s.repeat_interleave(wg//ag,dim=1),ag,4,True,q,True)
        elif 'down' in self.stages:
            down,k=linear(hidden,*e['g64']['down'],prequantized=q,activation_group=ag,
                divisor=16,**{**PLAN['down'],**c},return_kernel=True);kernels['down']=k
        else:
            down,k=self.control.functions['down'](hidden,*e['weights']['down'],prequantized=q,
                **PLAN['down'],trigger_mode=3,prefetch=1,return_kernel=True);kernels['down']=k
        following=norm_stage('qkv',down,summed,nd['weight'],nd['eps'])
        value=(up,down,following)
        return (value,kernels) if audit else value


def configs(pilot):
    up={'activation_group':64};qkv={'activation_group':64}
    mixed={'up':up,'qkv':qkv}
    full={'up':{**up,'rows':64},'down':{'activation_group':64},'qkv':qkv}
    choices={'control':{},'norm_a64':mixed,'up_a64':{'up':up},'qkv_a64':{'qkv':qkv},
             'all_a64':full,'norm_a64_down_a32':{**mixed,'down':{}}}
    if not pilot:
        for r in (4,16):choices[f'qkv_r{r}']={**mixed,'qkv':{**qkv,'rows':r}}
        for ig,ir in ((1,1),(1,2),(1,4),(2,1),(2,2),(4,4)):
            choices[f'up_ig{ig}_ir{ir}']={**mixed,'up':{**up,'integer_groups':ig,'integer_rows':ir}}
        for ig,ir in ((1,1),(1,2),(1,4),(2,2)):
            choices[f'all_up_ig{ig}_ir{ir}']={**full,'up':{**full['up'],'integer_groups':ig,'integer_rows':ir}}
        for r,w in ((2,2),(4,4),(8,4)):
            choices[f'all_down_r{r}w{w}']={**full,'down':{'activation_group':64,'rows':r,'warps':w}}
    return choices


def weight_only_configs():
    choices={'control':{}}
    for name in ('up','down','qkv'):
        for factor in (0,1):choices[f'{name}_f{factor}']={name:{'factor':factor}}
    choices['up_qkv_f1']={'up':{'factor':1},'qkv':{'factor':1}}
    choices['up_down_f1']={'up':{'factor':1},'down':{'factor':1}}
    choices['all_f1']={n:{'factor':1} for n in ('up','down','qkv')}
    for stage,tiles in (('up',[{'integer_groups':1},{'integer_rows':2}]),
                        ('down',[{'rows':2},{'rows':8},{'warps':4}]),
                        ('qkv',[{'rows':4},{'rows':16}])):
        for i,tile in enumerate(tiles):choices[f'{stage}_f1_tile{i}']={stage:{'factor':1,**tile}}
    return choices


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--layers',type=int,choices=(1,36),default=36)
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--pilot',action='store_true')
    p.add_argument('--weight-only',action='store_true');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag) or a.rounds<2:raise ValueError('Safe tag and at least two rounds required')
    path=RESULTS/f'group64_a64_{a.tag}.json';assert not path.exists(),'Preserve evidence'
    torch.set_num_threads(4);ring=[load(i) for i in (range(36) if a.layers==36 else (17,))]
    choices=weight_only_configs() if a.weight_only else configs(a.pilot)
    chains={};result={'codebooks':32,'configs':choices,'layers':a.layers,
        'checks':[],'resources':{},'errors':{},'graph_edges':{},'rows':[],
        'method':'Actual layer ring with independently packed original DP4A and standalone norm/quant arithmetic reference for each mixed-precision plan. Three frozen inputs per layer, eager/private graph comparisons, rotating/reversing measurement order. Not TTFA or quality evidence.'}
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name,stages in choices.items():
            candidate=Chain(stages)
            try:
                for e in ring:
                    for index in (0,10,31):
                        expected=candidate(e,index,reference=True);actual=candidate(e,index)
                        aa,bb=flatten(actual),flatten(expected)
                        errors=[float(((x.float()-y.float()).square().mean()/y.float().square().mean().clamp_min(1e-20)).sqrt()) for x,y in zip(aa,bb,strict=True)]
                        row={'config':name,'layer':e['layer'],'input':index,'relative_rms':errors,
                            'mismatches':[int((x!=y).sum()) for x,y in zip(aa,bb,strict=True)]}
                        # Quantizers must agree with standalone quantization of
                        # the actual candidate output, even if its projection differs.
                        standalone=quantize(actual[0][1],stages.get('down',{}).get('activation_group',32))
                        row['output_quant_exact']=all(torch.equal(x,y) for x,y in zip(actual[0][2],standalone,strict=True))
                        result['checks'].append(row)
                        assert row['output_quant_exact'],'Output quantizer mismatch'
                        assert max(errors)<.006,'Excessive chain error against independently packed reference'
                _,kernels=candidate(ring[0],audit=True)
                result['resources'][name]={n:{'registers':k.n_regs,'spills':k.n_spills,'shared_bytes':k.metadata.shared} for n,k in kernels.items()}
                for _ in range(2):candidate(ring[0])
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):actual=candidate(ring[0])
                edges=graph_edges(graph);result['graph_edges'][name]=edges
                assert len(edges)==2 and all(e['type']==1 for e in edges)
                graph.replay();expected=candidate(ring[0])
                assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(flatten(actual),flatten(expected),strict=True))
                chains[name]=candidate
                print('CHECKED',name,'max_relative_rms',max(max(x['relative_rms']) for x in result['checks'] if x['config']==name),flush=True)
            except Exception as error:
                result['errors'][name]=repr(error);print('ERROR',name,repr(error),flush=True)
            save()
    torch.cuda.current_stream().wait_stream(stream)
    for repeat in range(a.rounds):
        order=list(chains);order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2:order.reverse()
        times={name:measure(chains[name],ring) for name in order}
        result['rows'].append({'round':repeat,'order':order,'us':times});save();print('ROUND',repeat,flush=True)
    result['summary']={name:{'median_us':statistics.median(row['us'][name] for row in result['rows']),
        'median_paired_gain_us':statistics.median(row['us']['control']-row['us'][name] for row in result['rows']),
        'max_relative_rms':max(max(row['relative_rms']) for row in result['checks'] if row['config']==name)} for name in chains}
    save();print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
