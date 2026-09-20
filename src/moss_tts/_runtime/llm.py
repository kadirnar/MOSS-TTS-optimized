"""Batch-one LLM: fused projections, Triton embedding/norm, static KV and CUDA graphs."""
import types
import torch
import torch.nn.functional as F
from transformers.cache_utils import StaticCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from .kernels import embedding_sum, rmsnorm, add_rmsnorm, qk_rope_decode, linear_decode, silu_mul


def sample_topk(logits, temperature=1.7, top_k=25, top_p=0.8):
    if temperature <= 0:
        return logits.argmax(-1)
    values, indexes = torch.topk(logits.float()/temperature, min(top_k, logits.shape[-1]), dim=-1)
    probs = values.softmax(-1)
    remove = probs.cumsum(-1)-probs > top_p
    probs = probs.masked_fill(remove, 0)
    # Sampling only retained top-k entries preserves the upstream distribution.
    sampled = torch.multinomial(probs, 1)
    return indexes.gather(-1, sampled).squeeze(-1)


def _norm(self, x):
    return rmsnorm(x, self.weight, self.variance_epsilon)


def _mlp(self, x, dp4a_input=None):
    if x.numel()!=x.shape[-1] and getattr(self,'_prefill_silu_config',None) is not None:
        from .prefill_pointwise import silu_mul as prefill_silu_mul
        return self.down_proj(prefill_silu_mul(F.linear(x,self._gate_up),**self._prefill_silu_config))
    if getattr(self,'_use_dp4a',False) and x.numel()==x.shape[-1]:
        from .int4_dp4a import int4_dp4a
        reciprocal=getattr(self,'_dp4a_reciprocal',False)
        grouped=getattr(self,'_dp4a_grouped_activation',False)
        if getattr(self,'_dp4a_packing',None):
            from .dp4a_packing import project
            if getattr(self,'_fused_gateup_quant',False):
                if getattr(self,'_scaled_dp4a',None):
                    from .dp4a_scaled import linear as scaled_linear
                    hidden,quantized=scaled_linear(x,self._quant_up,self._quant_up_scale,
                        **self._scaled_dp4a['up'],paired=True,fused=True,prequantized=dp4a_input,scale_mode=4)
                else:
                    from .dp4a_gateup_quant import gateup_quant
                    hidden,quantized=gateup_quant(x,self._quant_up,self._quant_up_scale,
                        rows=32,warps=4,direct=True,prequantized=dp4a_input,scale_mode=4)
                return project(self,'down',hidden,quantized)
            return project(self,'down',project(self,'up',x,dp4a_input))
        if getattr(self,'_fused_dp4a_gateup',False):
            from .dp4a_gateup import gateup
            hidden=gateup(x,self._quant_up,self._quant_up_scale,self._dp4a_group,1,dp4a_input)
            return int4_dp4a(hidden,self._quant_down,self._quant_down_scale,self._dp4a_group,2,reciprocal,grouped_activation=grouped)
        up=int4_dp4a(x,self._quant_up,self._quant_up_scale,self._dp4a_group,1,reciprocal,dp4a_input,grouped)
        if getattr(self,'_fused_dp4a',False):
            from .dp4a_fusions import silu_quant
            hidden,quantized=silu_quant(up,self._dp4a_group if grouped else 0)
            return int4_dp4a(hidden,self._quant_down,self._quant_down_scale,self._dp4a_group,2,reciprocal,quantized,grouped)
        return int4_dp4a(silu_mul(up),self._quant_down,self._quant_down_scale,self._dp4a_group,2,reciprocal,grouped_activation=grouped)
    if getattr(self,'_fused_gateup',False) and x.numel()==x.shape[-1]:
        from .fused_mlp import fp8_silu_decode
        hidden=fp8_silu_decode(x,self._gate_up,None)
        return linear_decode(hidden,self.down_proj.weight)
    if getattr(self,'_triton_gemv',False):
        up=linear_decode(x,self._gate_up)
        return linear_decode(silu_mul(up),self.down_proj.weight)
    return self.down_proj(silu_mul(F.linear(x,self._gate_up)))




def _attention(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, dp4a_input=None, projected_qkv=None, **kwargs):
    shape = hidden_states.shape[:-1]
    dp4a=getattr(self,'_use_dp4a',False) and hidden_states.numel()==hidden_states.shape[-1]
    if dp4a:
        from .int4_dp4a import int4_dp4a
        from .dp4a_packing import project
    if projected_qkv is not None:
        if not dp4a or projected_qkv.shape!=(*shape,sum(self._qkv_sizes)):
            raise ValueError('Precomputed QKV requires the one-row DP4A decode path')
        projected=projected_qkv
    elif dp4a:
        if getattr(self,'_dp4a_packing',None):
            projected=project(self,'qkv',hidden_states,dp4a_input)
        else:projected=int4_dp4a(hidden_states,self._quant_qkv,self._quant_qkv_scale,self._dp4a_group,1,getattr(self,'_dp4a_reciprocal',False),dp4a_input,getattr(self,'_dp4a_grouped_activation',False))
    else:
        projected=linear_decode(hidden_states,self._qkv) if getattr(self,'_triton_gemv',False) else F.linear(hidden_states,self._qkv)
    if hidden_states.shape[1]==1 and getattr(self,'_triton_decode',False):
        cos,sin=position_embeddings
        quantize_output=dp4a and getattr(self,'_quantize_attention_output',False)
        x=qk_rope_decode(projected,self.q_norm.weight,self.k_norm.weight,cos,sin,past_key_values,self.layer_idx,cache_position,self.q_norm.variance_epsilon,getattr(self,'_decode_backend',None),self._decode_block,self._decode_warps,getattr(self,'_decode_capacity',None),quantize_output,getattr(self,'_native_attention',False),getattr(self,'_attention_pdl',None))
        output_quantized=None
        if quantize_output:x,output_quantized=x
        if dp4a:
            if getattr(self,'_dp4a_packing',None):return project(self,'out',x,output_quantized),None
            return int4_dp4a(x,self._quant_out,self._quant_out_scale,self._dp4a_group,1,getattr(self,'_dp4a_reciprocal',False),output_quantized,grouped_activation=getattr(self,'_dp4a_grouped_activation',False)),None
        return (linear_decode(x,self.o_proj.weight) if getattr(self,'_triton_gemv',False) else self.o_proj(x)),None
    if hidden_states.shape[1]>1 and getattr(self,'_fused_prefill_qkv',False) and past_key_values is not None:
        from .prefill_qkv import project as prepare_qkv
        storage=past_key_values.layers[self.layer_idx]
        q=prepare_qkv(projected,self.q_norm.weight,self.k_norm.weight,*position_embeddings,
            storage.keys,storage.values,cache_position,self.q_norm.variance_epsilon)
        k,v=storage.keys,storage.values
    else:
        q, k, v = projected.split(self._qkv_sizes, dim=-1)
        q = self.q_norm(q.reshape(*shape, -1, self.head_dim)).transpose(1,2)
        k = self.k_norm(k.reshape(*shape, -1, self.head_dim)).transpose(1,2)
        v = v.reshape(*shape, -1, self.head_dim).transpose(1,2)
        cos, sin = position_embeddings
        q,k = apply_rotary_pos_emb(q,k,cos,sin)
        if past_key_values is not None:
            # FastLLM owns explicit positions and resets by overwriting its prefix.
            # New Transformers StaticCache.update appends at an internal counter,
            # ignoring cache_position; repeated graph warmups would overflow it.
            storage=past_key_values.layers[self.layer_idx]
            storage.keys.index_copy_(2,cache_position,k)
            storage.values.index_copy_(2,cache_position,v)
            k,v=storage.keys,storage.values
    if hidden_states.shape[1]>1:
        # A prefix-cache experiment can start after a restored causal prefix.
        # The offset is static while a suffix graph is captured, never inferred
        # by reading a CUDA scalar during capture.
        n=hidden_states.shape[1]
        offset=getattr(self,'_prefill_offset',0)
        k,v=k[:,:,:offset+n],v[:,:,:offset+n]
        if offset:
            from torch.nn.attention.bias import causal_lower_right
            attention_mask=causal_lower_right(n,offset+n)
        else:attention_mask=None
    x = F.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask, dropout_p=0.,
        is_causal=attention_mask is None and q.shape[-2]>1, scale=self.scaling, enable_gqa=True)
    x=x.transpose(1,2).reshape(*shape,-1)
    return self.o_proj(x), None


@torch.inference_mode()
def fuse_backbone(model):
    for module in model.language_model.modules():
        if module.__class__.__name__ == "Qwen3RMSNorm":
            module.forward = types.MethodType(_norm, module)
    for layer in model.language_model.layers:
        a = layer.self_attn
        a.register_buffer("_qkv", torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight],0))
        a._qkv_sizes = [a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features]
        a.forward = types.MethodType(_attention,a)
        m=layer.mlp
        m.register_buffer("_gate_up",torch.cat([m.gate_proj.weight,m.up_proj.weight],0))
        m.forward=types.MethodType(_mlp,m)


class FastLLM:
    def __init__(self, model, *, max_length=1024, fused=True, graph=True, greedy=False, triton_attention=True, triton_gemv=True, attention_block=128, attention_warps=4, fused_residual=False, fused_gateup=False):
        self.weight_quantization='none'
        self.model=model
        self.max_length=max_length
        self.graph_enabled=graph
        self.greedy=greedy
        self.fused_residual=fused_residual
        if fused_residual and not fused:
            raise ValueError('Residual fusion requires the fused backbone')
        if fused_gateup and not fused:
            raise ValueError('Gate/up fusion requires a BF16 or 8-bit fused backbone')
        self.cfg=model.config
        lc=self.cfg.language_config
        if (lc.hidden_size,lc.head_dim,lc.num_attention_heads,lc.num_key_value_heads,self.cfg.n_vq,self.cfg.audio_vocab_size)!=(4096,128,32,8,32,1024):
            raise ValueError('These specialized kernels require the MOSS-TTS-v1.5 Qwen3-8B architecture')
        if max_length<128 or max_length%128:
            raise ValueError('max_length must be a positive multiple of 128')
        if attention_block not in (16,32,64,128,256) or max_length%attention_block or attention_warps not in (4,8):
            raise ValueError('Unsupported attention tile or warp count')
        if fused:
            fuse_backbone(model)
            for layer in model.language_model.layers:
                layer.self_attn._triton_decode=triton_attention
                layer.self_attn._decode_block=attention_block
                layer.self_attn._decode_warps=attention_warps
                layer.mlp._triton_gemv=triton_gemv
                layer.mlp._fused_gateup=fused_gateup
                layer.mlp._fp8_prefill=False
                layer.self_attn._fp8_prefill=False
                layer.self_attn._triton_gemv=triton_gemv
                layer.mlp._use_quantized=False
                layer.self_attn._use_quantized=False
                layer.mlp._use_marlin=False
                layer.self_attn._use_marlin=False
                layer.mlp._use_dp4a=False
                layer.self_attn._use_dp4a=False
                layer.mlp._quantized_tiled=False
                layer.self_attn._quantized_tiled=False
                layer.self_attn._decode_backend=None
        self.audio_embeddings=torch.stack([e.weight for e in model.emb_ext])
        self.audio_heads=torch.cat([h.weight[:1024] for h in model.lm_heads[1:]],0).contiguous()
        self.text_ids=torch.tensor([self.cfg.audio_assistant_gen_slot_token_id,self.cfg.audio_assistant_delay_slot_token_id],device='cuda')
        self.text_head=model.lm_heads[0].weight[self.text_ids].contiguous()
        self.cache=StaticCache(model.config.language_config,max_cache_len=max_length)
        self.cache.early_initialization(1,model.config.language_config.num_key_value_heads,model.config.language_config.head_dim,torch.bfloat16,torch.device('cuda'))
        self.ids=torch.full((1,1,33),1024,device='cuda',dtype=torch.long)
        self.ids[...,0]=self.cfg.audio_start_token_id
        self.position=torch.zeros(1,device='cuda',dtype=torch.long)
        self.audio_length=torch.ones(1,device='cuda',dtype=torch.long)
        self.delay_length=torch.full((1,),-1,device='cuda',dtype=torch.long)
        self.kv_index=torch.arange(max_length,device='cuda')
        self.channels=torch.arange(32,device='cuda')
        self.graph=None
        self.prefill_graphs={}

    def _prefill_forward(self,ids,position,last_index):
        mask=(self.kv_index[None,:]<=position[:,None]).view(1,1,ids.shape[1],-1)
        h=self.hidden(ids,position,mask).index_select(1,last_index).squeeze(1)
        return F.linear(h,self.model.lm_heads[0].weight), F.linear(h,self.audio_heads).view(32,1024)

    @torch.inference_mode()
    def capture_prefill(self,buckets=(128,256,512)):
        for n in buckets:
            if n>=self.max_length:continue
            ids=torch.full((1,n,33),1024,dtype=torch.long,device='cuda')
            ids[...,0]=self.cfg.pad_token_id
            pos=torch.arange(n,device='cuda')
            last=torch.tensor([n-1],device='cuda')
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self._prefill_forward(ids,pos,last)
            torch.cuda.current_stream().wait_stream(stream)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=self._prefill_forward(ids,pos,last)
            # Every externally allocated graph input must remain alive.
            self.prefill_graphs[n]=(g,ids,pos,last,out)

    def hidden(self, ids, position, mask):
        emb=embedding_sum(ids,self.model.get_input_embeddings().weight,self.audio_embeddings)
        if self.fused_residual and ids.shape[1]==1:
            backbone=self.model.language_model
            rope=backbone.rotary_emb(emb,position[None])
            if getattr(self,'norm_projection_fused',None):
                from .dp4a_norm_projection import hidden
                return hidden(self,emb,rope,position,mask)
            residual=emb
            fused_dp4a=getattr(self,'fused_dp4a',False)
            a=backbone.layers[0].self_attn
            quant_group=a._dp4a_group if getattr(a,'_dp4a_grouped_activation',False) else 0
            quantized=None
            if fused_dp4a:
                from .dp4a_fusions import norm_quant
                layout_limit=getattr(self,'dp4a_norm_layout_limit',0)
                norm=backbone.layers[0].input_layernorm
                _,hidden,quantized=norm_quant(emb,None,norm.weight,norm.variance_epsilon,quant_group,layout_limit)
            else:hidden=backbone.layers[0].input_layernorm(emb)
            for i,layer in enumerate(backbone.layers):
                attention,_=layer.self_attn(hidden_states=hidden,position_embeddings=rope,
                    attention_mask=mask,past_key_values=self.cache,cache_position=position,dp4a_input=quantized)
                norm=layer.post_attention_layernorm
                if fused_dp4a:
                    residual,hidden,quantized=norm_quant(attention,residual,norm.weight,norm.variance_epsilon,quant_group,layout_limit)
                    mlp=layer.mlp(hidden,dp4a_input=quantized)
                else:
                    residual,hidden=add_rmsnorm(attention,residual,norm.weight,norm.variance_epsilon)
                    mlp=layer.mlp(hidden)
                norm=backbone.layers[i+1].input_layernorm if i+1<len(backbone.layers) else backbone.norm
                if fused_dp4a and i+1<len(backbone.layers):
                    residual,hidden,quantized=norm_quant(mlp,residual,norm.weight,norm.variance_epsilon,quant_group,layout_limit)
                else:residual,hidden=add_rmsnorm(mlp,residual,norm.weight,norm.variance_epsilon)
            return hidden
        if ids.shape[1]>1 and getattr(self,'_prefill_residual_fused',False):
            from .prefill_pointwise import hidden
            return hidden(self,emb,position,mask)
        return self.model.language_model(inputs_embeds=emb, past_key_values=self.cache,
            cache_position=position,position_ids=position[None],attention_mask={"full_attention":mask},
            use_cache=True,output_hidden_states=False).last_hidden_state

    def _decode(self):
        mask=(self.kv_index<=self.position).view(1,1,1,-1)
        h=self.hidden(self.ids,self.position,mask)[:, -1]
        text=F.linear(h,self.text_head)
        head_count=getattr(self,'_audio_head_count',32)
        if head_count==32:
            audio=F.linear(h,self.audio_heads).view(32,1024)
        else:
            # Only graph managers whose CPU schedule covers every active
            # codebook may choose a prefix. Keep the sampler's full shape/RNG.
            audio=torch.zeros((32,1024),device=h.device,dtype=h.dtype)
            audio[:head_count].copy_(F.linear(h,self.audio_heads[:head_count*1024]).view(head_count,1024))
        text_index=sample_topk(text,0 if self.greedy else 1.5,2,1.0)
        text_token=self.text_ids[text_index]
        text_token=torch.where(self.delay_length>=0,self.cfg.audio_assistant_delay_slot_token_id,text_token)
        text_token=torch.where(self.delay_length==32,self.cfg.audio_end_token_id,text_token)
        audio_tokens=sample_topk(audio,0 if self.greedy else 1.7)
        valid=(self.channels<self.audio_length)&((self.delay_length<0)|(self.channels>=self.delay_length))
        audio_tokens=torch.where(valid,audio_tokens,1024)
        return torch.cat([text_token,audio_tokens]).reshape(1,1,33), text, audio

    @torch.inference_mode()
    def warmup(self):
        for _ in range(3): self._decode()
        torch.cuda.synchronize()
        if self.graph_enabled:
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): self._decode()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.next_ids,self.text_logits,self.audio_logits=self._decode()

    @torch.inference_mode()
    def prefill(self, ids):
        if ids.shape[0]!=1 or ids.shape[1]>=self.max_length:
            raise ValueError("Batch must be one and prompt must fit the KV capacity")
        n=ids.shape[1]
        buckets=[b for b in self.prefill_graphs if b>=n]
        if buckets:
            g,static_ids,position,last,out=self.prefill_graphs[min(buckets)]
            static_ids[:,:n].copy_(ids)
            last.fill_(n-1)
            g.replay()
            return out
        pos=torch.arange(n,device=ids.device)
        mask=(self.kv_index[None,:]<=pos[:,None]).view(1,1,n,-1)
        h=self.hidden(ids,pos,mask)[:,-1]
        return F.linear(h,self.model.lm_heads[0].weight), F.linear(h,self.audio_heads).view(32,1024)

    @torch.inference_mode()
    def full_step(self, ids, position):
        self.position.fill_(position)
        mask=(self.kv_index<=self.position).view(1,1,1,-1)
        h=self.hidden(ids,self.position,mask)[:,-1]
        return F.linear(h,self.model.lm_heads[0].weight)

    @torch.inference_mode()
    def step(self, ids, position, audio_length, delay_length):
        if not 0<=position<self.max_length:
            raise ValueError("KV capacity exceeded")
        self.ids.copy_(ids)
        self.position.fill_(position)
        self.audio_length.fill_(audio_length)
        self.delay_length.fill_(delay_length)
        if self.graph is not None:
            self.graph.replay()
            return self.next_ids.clone()
        return self._decode()[0]
