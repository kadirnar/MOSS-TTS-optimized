"""Freeze real operator inputs, then compare independent Triton compiler runs.

Capture is deliberately instrumented and is never a performance measurement.
Replays use identical tensors, independent of subsequent generation divergence.
"""
import argparse
import json
from pathlib import Path
import torch
import triton
from .common import RESULTS,load_models
from . import llm as llm_module,dp4a_fusions,dp4a_packing,kernels
from .attention_quant import reduce_quant,enable_attention_quant


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,tuple):return tuple(cpu(x) for x in value)
    if isinstance(value,list):return [cpu(x) for x in value]
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    return value


def cuda(value):
    if isinstance(value,torch.Tensor):return value.cuda()
    if isinstance(value,tuple):return tuple(cuda(x) for x in value)
    if isinstance(value,list):return [cuda(x) for x in value]
    if isinstance(value,dict):return {k:cuda(v) for k,v in value.items()}
    return value


@torch.inference_mode()
def capture(folder):
    from .streaming import StreamingTTS
    from .calibrated_backend import install_calibrated,enable_grouped_activation
    from .dp4a_gateup import enable_gateup
    if folder.exists():raise ValueError('Use a new capture folder')
    folder.mkdir(parents=True)
    model,codec,processor=load_models()
    engine=StreamingTTS(model,codec,processor,graph=False,attention_block=32,fused_residual=True)
    fast=engine.llm
    install_calibrated(fast,RESULTS/'gptq_v1_g32_d10',backend='dp4a')
    enable_grouped_activation(fast);dp4a_fusions.enable_fusions(fast,layout_limit=8);enable_gateup(fast)
    dp4a_packing.install_packing(fast,RESULTS/'dp4a_direct_exact_plan.json');enable_attention_quant(fast)
    weight_names={w.data_ptr():n for n,w in model.named_parameters()}
    modules={id(m):i for i,l in enumerate(model.language_model.layers) for m in (l.self_attn,l.mlp)}
    selected={0,17,35};steps={0,10,31};phase='prefill';step=-1;norm_index=0;records=[]
    def save(kind,label,data):
        name=f'{len(records):03d}_{kind}.pt'
        torch.save(cpu(data),folder/name)
        records.append({'kind':kind,'label':label,'file':name})
    original_norm=llm_module.rmsnorm
    def norm(x,w,eps):
        out=original_norm(x,w,eps)
        name=weight_names.get(w.data_ptr(),'')
        if phase=='prefill' and (any(f'layers.{i}.' in name for i in selected) or name=='language_model.norm.weight'):
            save('rmsnorm',name,{'x':x,'weight':w,'eps':eps,'expected':out})
        return out
    llm_module.rmsnorm=norm
    original_norm_quant=dp4a_fusions.norm_quant
    def norm_quant(x,r,w,eps,group=0,layout_limit=0):
        nonlocal norm_index
        index=norm_index%72;norm_index+=1
        out=original_norm_quant(x,r,w,eps,group,layout_limit)
        if step in steps and index//2 in selected:
            save('norm_quant',f'step{step}_norm{index}',{'x':x,'residual':r,'weight':w,'eps':eps,'group':group,'layout_limit':layout_limit,'expected':out})
        return out
    dp4a_fusions.norm_quant=norm_quant
    original_add=llm_module.add_rmsnorm
    def add(x,r,w,eps):
        out=original_add(x,r,w,eps)
        if step in steps:save('add_rmsnorm',f'step{step}_final',{'x':x,'residual':r,'weight':w,'eps':eps,'expected':out})
        return out
    llm_module.add_rmsnorm=add
    original_project=dp4a_packing.project
    def project(module,name,x,prequantized=None):
        out=original_project(module,name,x,prequantized)
        layer=modules[id(module)]
        if step in steps and layer in selected:
            save('projection',f'step{step}_layer{layer}_{name}',{'x':x,'prequantized':prequantized,'layer':layer,'name':name,'cfg':module._dp4a_packing[name],'expected':out})
        return out
    dp4a_packing.project=project
    original_embedding=llm_module.embedding_sum
    def embedding(ids,text,audio):
        out=original_embedding(ids,text,audio)
        if phase=='prefill' or step in steps:
            n=ids.shape[1]
            # Compact lookup tables retain every fetched vector and arithmetic
            # order without copying the 1.5 GB full embedding vocabulary.
            compact_text=text[ids[0,:,0]]
            compact_audio=torch.stack([audio[q,ids[0,:,q+1]] for q in range(32)])
            compact_ids=torch.arange(n,device=ids.device).view(1,n,1).expand(1,n,33).contiguous()
            save('embedding',f'{phase}_step{step}',{'ids':compact_ids,'text':compact_text,'audio':compact_audio,'expected':out})
        return out
    llm_module.embedding_sum=embedding
    original_attention=llm_module.qk_rope_decode
    def attention(qkv,qw,kw,cos,sin,cache,layer,position,eps,backend=None,block=128,warps=4,context_capacity=None,quantize_output=False,native_attention=False):
        out=original_attention(qkv,qw,kw,cos,sin,cache,layer,position,eps,backend,block,warps,context_capacity,quantize_output,native_attention)
        if step in steps and layer in selected:
            kc,vc=cache.layers[layer].keys,cache.layers[layer].values
            q=torch.empty(32,128,device='cuda',dtype=torch.bfloat16)
            kernels._qk_rope_cache[(40,)](qkv,qw,kw,cos,sin,q,kc,vc,position,1024,eps,enable_fp_fusion=False)
            splits=(context_capacity or 1024)//block
            part=torch.empty(32,splits,128,device='cuda');lse=torch.empty(32,splits,device='cuda')
            kernels._decode_attn[(32,splits)](q,kc,vc,position,part,lse,1024,splits,block,num_warps=warps)
            reduced=reduce_quant(part,lse)
            assert torch.equal(out[0],reduced[0])
            save('attention',f'step{step}_layer{layer}',{'qkv':qkv,'qw':qw,'kw':kw,'cos':cos,'sin':sin,'position':position,'eps':eps,'block':block,'warps':warps,'splits':splits,'k':kc,'v':vc,'q':q,'part':part,'lse':lse,'expected':reduced})
        return out
    llm_module.qk_rope_decode=attention
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    fast.prefill(fixture['inputs']['input_ids'].cuda())
    phase='decode';tokens=fixture['upstream_ids'].cuda()
    for step in range(32):
        norm_index=0
        fast.step(tokens[step:step+1][None],145+step,step+1,-1)
    manifest={'torch':torch.__version__,'triton':triton.__version__,'records':records,'codebooks':32,
              'scope':'Instrumented 145-token prefill and 32 teacher-forced decode steps; layers 0/17/35, steps 0/10/31. Captured operator inputs and outputs; no performance claims.'}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Captured',len(records),'operators in',folder,flush=True)
    engine.codec.close()


@torch.inference_mode()
def replay(folder,tag):
    from .dp4a_direct import linear as direct_linear
    torch.set_num_threads(4)
    manifest=json.loads((folder/'manifest.json').read_text());rows=[];weights={}
    def compare(label,kind,actual,expected):
        if isinstance(actual,tuple):
            for i,(a,b) in enumerate(zip(actual,expected)):compare(label+f'/{i}',kind,a,b)
            return
        equal=actual==expected
        finite=torch.isfinite(actual)&torch.isfinite(expected)
        err=(actual.float()-expected.float()).abs()
        row={'label':label,'kind':kind,'elements':actual.numel(),'mismatches':int((~equal).sum()),'max_abs':float(err[finite].max()) if finite.any() else 0.0,'dtype':str(actual.dtype)}
        rows.append(row)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    for record in manifest['records']:
        d=cuda(torch.load(folder/record['file'],weights_only=True));kind=record['kind'];label=record['label']
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if kind=='rmsnorm':actual=kernels.rmsnorm(d['x'],d['weight'],d['eps'])
            elif kind=='norm_quant':actual=dp4a_fusions.norm_quant(d['x'],d['residual'],d['weight'],d['eps'],d['group'],d['layout_limit'])
            elif kind=='add_rmsnorm':actual=kernels.add_rmsnorm(d['x'],d['residual'],d['weight'],d['eps'])
            elif kind=='embedding':actual=kernels.embedding_sum(d['ids'],d['text'],d['audio'])
            elif kind=='projection':
                key=(d['layer'],d['name']);cfg=d['cfg']
                if key not in weights:
                    saved=torch.load(RESULTS/'gptq_v1_g32_d10'/f'{key[0]:02d}_{key[1]}.pt',map_location='cuda',weights_only=True)
                    weights[key]=(dp4a_packing.pack_interleaved(saved['packed']),saved['scales'].to(getattr(torch,cfg['scale_dtype'])))
                w,s=weights[key]
                args={'rows':cfg['rows'],'warps':cfg['warps'],'paired':d['name']=='up','prequantized':d['prequantized']}
                actual=direct_linear(d['x'],w,s,**args) if cfg.get('activation_load')=='direct' else dp4a_packing.linear(d['x'],w,s,interleaved=True,**args)
            elif kind=='attention':
                k=d['k'].clone();v=d['v'].clone();q=torch.empty_like(d['q']);pos=int(d['position'].item())
                kernels._qk_rope_cache[(40,)](d['qkv'],d['qw'],d['kw'],d['cos'],d['sin'],q,k,v,d['position'],1024,d['eps'],enable_fp_fusion=False)
                compare(label+'/q','qk_rope',q,d['q']);compare(label+'/k','qk_rope',k[:,:,pos],d['k'][:,:,pos]);compare(label+'/v','qk_rope',v[:,:,pos],d['v'][:,:,pos])
                part=torch.empty_like(d['part']);lse=torch.empty_like(d['lse'])
                compiled=kernels._decode_attn[(32,d['splits'])](d['q'],d['k'],d['v'],d['position'],part,lse,1024,d['splits'],d['block'],num_warps=d['warps'])
                compare(label+'/part','attention_partial',part,d['part']);compare(label+'/lse','attention_lse',lse,d['lse'])
                actual=reduce_quant(d['part'],d['lse']);kind='attention_reduce'
                if record==next(r for r in manifest['records'] if r['kind']=='attention'):
                    for ext in ('ptx','ttgir'):(RESULTS/f'compiler_attention_{tag}.{ext}').write_text(compiled.asm[ext])
            else:raise ValueError(kind)
            compare(label,kind,actual,d['expected'])
        torch.cuda.current_stream().wait_stream(stream)
        del d,actual
    totals={}
    for row in rows:
        total=totals.setdefault(row['kind'],{'checks':0,'mismatches':0,'elements':0,'max_abs':0.0})
        total['checks']+=1;total['mismatches']+=row['mismatches'];total['elements']+=row['elements'];total['max_abs']=max(total['max_abs'],row['max_abs'])
    result={'torch':torch.__version__,'triton':triton.__version__,'capture':str(folder),'private_stream':True,'totals':totals,'rows':rows}
    (RESULTS/f'compiler_audit_{tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(totals,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('capture','replay'));p.add_argument('--folder',default=str(RESULTS/'compiler_capture_v1'));p.add_argument('--tag',default='');a=p.parse_args()
    if a.mode=='capture':capture(Path(a.folder))
    else:
        if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
        replay(Path(a.folder),a.tag)


if __name__=='__main__':main()
