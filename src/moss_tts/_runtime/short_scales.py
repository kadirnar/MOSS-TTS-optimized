"""Experimental exact BF16 scale storage with explicit floating layouts."""
import torch
from .dp4a_layout import linear


PLAN={'out':{'rows':4,'warps':4,'integer_groups':1,'integer_rows':1},
      'down':{'rows':4,'warps':2,'integer_groups':1,'integer_rows':1}}


def project(module,name,x,quantized):
    fn=linear;options={}
    config=getattr(module,'_projection_pdl',None)
    if config:
        if config['scale_prefetch']:
            from .dp4a_layout_pdl_prefetch import linear as fn
            options['prefetch']=1
        else:
            from .dp4a_layout_pdl import linear as fn
        options['trigger_mode']=config['projection_trigger']
    if name=='down' and getattr(module,'_bulk_down_linear',None):fn=module._bulk_down_linear
    if name=='out' and getattr(module,'_async_out_linear',None):fn=module._async_out_linear
    return fn(x,getattr(module,'_quant_'+name),getattr(module,'_quant_'+name+'_scale'),
              prequantized=quantized,**PLAN[name],**options)


@torch.inference_mode()
def enable(llm):
    if llm.graph is not None or llm.prefill_graphs:raise RuntimeError('Install short scales before graph capture')
    if not getattr(llm,'norm_projection_fused',None) or getattr(llm,'compressed_scales',None):
        raise ValueError('Requires the G32 norm-projection preset without compressed FP32 scales')
    replacements=[]
    for layer in llm.model.language_model.layers:
        if not getattr(layer.self_attn,'_quantize_attention_output',False):raise ValueError('Short scales require fused attention-output quantization')
        for module,name in ((layer.self_attn,'out'),(layer.mlp,'down')):
            if getattr(module,'_dp4a_group',None)!=32 or not getattr(module,'_scaled_dp4a',None) or getattr(module,'_group128_plan',None):
                raise ValueError('Selected G32 scaled-DP4A modules required')
            key='_quant_'+name+'_scale';old=getattr(module,key)
            if old.dtype!=torch.float32:raise ValueError('Original FP32 scale storage required')
            new=old.bfloat16()
            if not torch.equal(new.float(),old):raise ValueError('Scales are not exactly representable as BF16')
            replacements.append((module,key,name,new))
    for module,key,name,new in replacements:
        setattr(module,key,new);module._short_scale_projection=name
    llm.short_scales={'plan':PLAN,'buffers':len(replacements),
                      'scale_bytes':sum(t.numel()*t.element_size() for _,_,_,t in replacements),
                      'all_values_exact':True}
    return llm.short_scales
