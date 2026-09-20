"""Capture all-layer raw normalization inputs and qualify producer fusion."""
import argparse
import json
from pathlib import Path
import torch
from .common import RESULTS,load_models


@torch.inference_mode()
def capture(folder):
    from .streaming import StreamingTTS
    from .calibrated_backend import install_calibrated,enable_grouped_activation
    from . import dp4a_fusions
    from .dp4a_gateup import enable_gateup
    from .dp4a_packing import install_packing
    from .attention_quant import enable_attention_quant
    from .attention_native import enable_native_attention
    from .dp4a_gateup_quant import enable_gateup_quant
    from .dp4a_scaled import enable_scaled
    if folder.exists():raise FileExistsError('Use a new capture folder')
    folder.mkdir(parents=True)
    model,codec,processor=load_models();engine=StreamingTTS(model,codec,processor,graph=False,attention_block=32,fused_residual=True);fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10',backend='dp4a')
    enable_grouped_activation(fast);dp4a_fusions.enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json');enable_attention_quant(fast);enable_native_attention(fast);enable_gateup_quant(fast);enable_scaled(fast)
    fast.capture_prefill((128,160,256,512))
    wanted={0,1,3,7,15,31,32,47,63,79,95,111};buffers={};position=-1;norm_index=0;record=None
    original=dp4a_fusions.norm_quant
    def observed(x,residual,weight,eps,group=0,layout_limit=0):
        nonlocal norm_index
        index=norm_index;norm_index+=1
        if position in wanted:
            name=f'{index//2:02d}_'+('up' if index%2 else 'qkv')
            b=buffers.setdefault(name,{'x':[],'residual':[],'weight':weight.detach().cpu().clone(),'eps':eps,'meta':[],'has_residual':residual is not None})
            assert b['has_residual']==(residual is not None)
            b['x'].append(x.detach().reshape(1,4096).cpu().clone())
            if residual is not None:b['residual'].append(residual.detach().reshape(1,4096).cpu().clone())
            b['meta'].append({'utterance':record['id'],'decode_step':position})
        return original(x,residual,weight,eps,group,layout_limit)
    dp4a_fusions.norm_quant=observed
    manifest=json.loads((RESULTS/'calibration_v1/manifest.json').read_text());utterances=[]
    try:
        for record in manifest['records']:
            if not record['id'].endswith('_2'):continue
            sequence=torch.load(RESULTS/f"calibration_v1/{record['id']}_tokens.pt",weights_only=True).cuda()
            prompt=record['prompt_tokens'];position=-1;fast.prefill(sequence[:,:prompt])
            for position in range(sequence.shape[1]-prompt):
                actual=prompt+position;cap=next(c for c in (128,256,512,1024) if actual<c)
                for layer in model.language_model.layers:layer.self_attn._decode_capacity=cap
                norm_index=0;fast.step(sequence[:,actual:actual+1],actual,position+1,-1)
                assert norm_index==72
            utterances.append({'id':record['id'],'steps':position+1,'prompt':prompt,'voice':record['voice'],'language':record['language']})
            print('CAPTURED',utterances[-1],flush=True)
    finally:dp4a_fusions.norm_quant=original
    files={}
    for name,b in buffers.items():
        b['x']=torch.cat(b['x']);b['residual']=torch.cat(b['residual']) if b['has_residual'] else None
        torch.save(b,folder/(name+'.pt'));files[name]={'rows':len(b['meta'])}
    assert len(files)==72
    (folder/'manifest.json').write_text(json.dumps({'codebooks':32,'files':files,'utterances':utterances,'scope':'Instrumented eager teacher forcing of four held-out calibration sequences with selected scaled DP4A, native attention, G32 quantization and 160-token prefill bucket. Raw residual/norm inputs across all layers; no timing claims.'},indent=2)+'\n')
    engine.codec.close();print('SAVED',len(files),'sites',sum(f['rows'] for f in files.values()),'inputs',flush=True)


@torch.inference_mode()
def validate(folder,tag):
    from .benchmark_norm_projection import flatten
    from .dp4a_fusions import norm_quant
    from .dp4a_packing import pack_interleaved
    from .dp4a_scaled import linear as old,SELECTED
    from .dp4a_norm_projection import linear
    torch.set_num_threads(4);records=[]
    configs={'qkv':{'rows':8,'integer_groups':1,'integer_rows':1},'up':{'rows':32,'integer_groups':2,'integer_rows':4}}
    outfile=RESULTS/f'norm_projection_validation_{tag}.json'
    if outfile.exists():raise FileExistsError('Preserve previous results')
    for layer in range(36):
        for name in ('qkv','up'):
            d=torch.load(folder/f'{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
            saved=torch.load(RESULTS/f'gptq_v1_g32_d10/{layer:02d}_{name}.pt',map_location='cuda',weights_only=True)
            w=pack_interleaved(saved['packed']);s=saved['scales'].bfloat16();fused=name=='up'
            inputs=[(str(i),d['x'][i:i+1],d['residual'][i:i+1] if d['has_residual'] else None) for i in range(d['x'].shape[0])]
            zero=torch.zeros(1,4096,device='cuda',dtype=torch.bfloat16);spike=zero.clone();spike[0,-1]=100
            inputs.extend([('zero',zero,zero if d['has_residual'] else None),('spike',spike,zero if d['has_residual'] else None)])
            for label,x,residual in inputs:
                summed,normalized,q=norm_quant(x,residual,d['weight'],d['eps'],32,8)
                out=old(normalized,w,s,**SELECTED[name],paired=fused,fused=fused,scale_mode=4 if fused else 0,prequantized=q)
                expected=((summed,*out) if fused else (summed,out),(normalized,*q))
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):actual=linear(x,residual,d['weight'],d['eps'],w,s,fused=fused,debug=True,**configs[name])
                torch.cuda.current_stream().wait_stream(stream)
                counts=[int((a!=b).sum()) for a,b in zip(flatten(actual),flatten(expected),strict=True)]
                records.append({'layer':layer,'projection':name,'input':label,'mismatches':counts})
            print(layer,name,'mismatches',sum(sum(r['mismatches']) for r in records if r['layer']==layer and r['projection']==name),flush=True)
    exact=all(not any(r['mismatches']) for r in records)
    result={'codebooks':32,'capture':str(folder),'configs':configs,'records':records,'checks':len(records),'all_exact':exact,'private_stream':True}
    outfile.write_text(json.dumps(result,indent=2)+'\n');print({'checks':len(records),'all_exact':exact},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('capture','validate'));p.add_argument('--folder',default=str(RESULTS/'norm_projection_capture_v1'));p.add_argument('--tag',default='v1');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    if a.mode=='capture':capture(Path(a.folder))
    else:validate(Path(a.folder),a.tag)


if __name__=='__main__':main()
