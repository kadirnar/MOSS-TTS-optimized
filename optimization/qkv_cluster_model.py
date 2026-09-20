"""Optional per-model decode callable using isolated clustered QKV cubins.

It never changes global dispatch or serving defaults. Prefill delegates to the
model's original method. Install before warming or capturing any model graph.
"""
import torch
from .kernels import embedding_sum,add_rmsnorm
from .dp4a_norm_projection import SELECTED
from .dp4a_packing import project
from .attention_pdl import launch as attention
from .attention_quant_pdl import reduce_quant


def enable(fast):
    import hashlib
    import json
    import triton
    from .common import RESULTS
    from .qkv_cluster_binary import load_bundle
    if fast.graph is not None or fast.prefill_graphs or getattr(fast,'qkv_cluster',None):
        raise RuntimeError('Install clustered QKV once, before graph capture')
    if triton.__version__!='3.7.1':raise ValueError('The qualified host arithmetic requires Triton 3.7.1')
    if not getattr(fast,'output_weight_prefetch',None):raise ValueError('Selected register-preload output required')
    folder=RESULTS/'qkv_cluster_bundle_v6';name='c8_t2_exact'
    choices,launchers=load_bundle(folder)
    config=choices[name]
    if config!={'ctas':8,'divisor':16,'trigger_mode':2,'legacy_projection':True,'legacy_norm':True}:
        raise ValueError('Unexpected cluster configuration')
    dispatch=Hidden(fast,launchers[name],config)
    manifest=(folder/'manifest.json').read_bytes();data=json.loads(manifest)
    fast.qkv_cluster={'codebooks':32,'config':name,'options':dict(config),'bundle':str(folder),
        'manifest_sha256':hashlib.sha256(manifest).hexdigest(),'compiler_triton':data['triton'],
        'host_triton':triton.__version__}
    fast.hidden=dispatch
    return dict(fast.qkv_cluster)


class Hidden:
    def __init__(self,fast,launcher,options):
        if fast.cfg.n_vq!=32 or fast.norm_projection_fused!=SELECTED:
            raise ValueError('Selected 32-codebook G32 model required')
        if not fast.fused_residual or not fast.bulk_prefetch or not fast.projection_pdl or not fast.attention_pdl:
            raise ValueError('Selected residual, bulk-prefetch and PDL paths required')
        for layer in fast.model.language_model.layers:
            a=layer.self_attn
            if not a._native_attention or not a._quantize_attention_output or a._decode_backend is not None or (a._decode_block,a._decode_warps)!=(32,4):
                raise ValueError('Native B32/W4 attention and G32 output quantization required')
        self.fast=fast;self.original=fast.hidden;self.launch=launcher;self.options=dict(options)

    def __call__(self,ids,position,mask):
        if ids.shape[1]!=1:return self.original(ids,position,mask)
        fast=self.fast;backbone=fast.model.language_model
        source=embedding_sum(ids,fast.model.get_input_embeddings().weight,fast.audio_embeddings)
        cos,sin=backbone.rotary_emb(source,position[None]);pending=None
        for layer in backbone.layers:
            a=layer.self_attn;norm=layer.input_layernorm;cache=fast.cache.layers[a.layer_idx]
            residual,_,q=self.launch(source,pending,norm.weight,norm.variance_epsilon,a._quant_qkv,a._quant_qkv_scale,
                a.q_norm.weight,a.k_norm.weight,cos,sin,cache.keys,cache.values,position,a.q_norm.variance_epsilon,**self.options)
            capacity=getattr(a,'_decode_capacity',None)
            if capacity is None:capacity=cache.keys.shape[-2]
            if not 0<capacity<=cache.keys.shape[-2] or capacity%32:raise ValueError('Invalid attention capacity')
            splits=capacity//32
            partial=torch.empty((32,splits,128),device=q.device,dtype=torch.float32)
            lse=torch.empty((32,splits),device=q.device,dtype=torch.float32)
            attention(q,cache.keys,cache.values,position,partial,lse,pdl=True,trigger=a._attention_pdl['attention'])
            states,quantized=reduce_quant(partial,lse,pdl=True,trigger=a._attention_pdl['reduce'])
            output=project(a,'out',states,quantized)
            norm=layer.post_attention_layernorm;mlp=layer.mlp
            residual,states,quantized=fast._bulk_norm_linear(output,residual,norm.weight,norm.variance_epsilon,
                mlp._quant_up,mlp._quant_up_scale,fused=True,**SELECTED['up'],trigger_mode=fast.projection_pdl['norm_trigger'])
            source=project(mlp,'down',states,quantized);pending=residual
        return add_rmsnorm(source,pending,backbone.norm.weight,backbone.norm.variance_epsilon)[1]
