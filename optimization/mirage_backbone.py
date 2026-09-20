"""Experimental Mirage online_notoken adapter for the unchanged MOSS backbone."""
from pathlib import Path
import torch


class MirageBackbone:
    @torch.inference_mode()
    def __init__(self, model, *, max_length=1024, output_dir=None):
        from mirage import PersistentKernel, MirageModelConfig
        from mirage.mpk.models.qwen3.builder import Qwen3Builder
        from mirage.utils import get_configurations_from_gpu
        cfg = model.config.language_config
        self.model = model
        self.max_length = max_length
        self.input_embedding = torch.zeros((1,cfg.hidden_size),device='cuda',dtype=torch.bfloat16)
        shape = (cfg.num_hidden_layers,1,max_length,cfg.num_key_value_heads,cfg.head_dim)
        self.keys = torch.zeros(shape,device='cuda',dtype=torch.bfloat16)
        self.values = torch.zeros_like(self.keys)
        def tensor(data,dtype=torch.int32):
            return torch.tensor(data,device='cuda',dtype=dtype)
        self.meta = {'step':tensor([0]),'tokens':torch.zeros((1,max_length),device='cuda',dtype=torch.int64),
            'input_tokens':tensor([[0]],torch.int64),'output_tokens':tensor([[0]],torch.int64),
            'num_new_tokens':tensor([1]),'prompt_lengths':tensor([0]),
            'qo_indptr_buffer':tensor([0,1]),'paged_kv_indptr_buffer':tensor([0,1]),
            'paged_kv_indices_buffer':tensor([0]),'paged_kv_last_page_len_buffer':tensor([1]),
            'paged_kv_indices_snapshot':tensor([0])}
        workers,schedulers = get_configurations_from_gpu(0)
        self.kernel = PersistentKernel(mode='online_notoken',world_size=1,mpi_rank=0,
            num_workers=workers,num_local_schedulers=schedulers,num_remote_schedulers=0,
            max_seq_length=max_length,max_num_batched_requests=1,max_num_batched_tokens=1,
            max_num_pages=1,page_size=max_length,meta_tensors=self.meta,profiler_tensor=None,
            trace_name=None,spec_decode_config=None,use_cutlass_kernel=True)
        positions = torch.arange(max_length,device='cuda')[None]
        rope = model.language_model.rotary_emb(self.input_embedding,positions)
        weights = {'model.'+name:value for name,value in model.language_model.state_dict().items()}
        # The stock builder begins with an embedding task even in notoken mode.
        # A one-row table receives the exact MOSS sum of text + 32 audio embeddings.
        weights['model.embed_tokens.weight'] = self.input_embedding
        config = MirageModelConfig(hidden_size=cfg.hidden_size,intermediate_size=cfg.intermediate_size,
            vocab_size=1,local_num_q_heads=cfg.num_attention_heads,local_num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,num_layers=cfg.num_hidden_layers,k_cache=self.keys,v_cache=self.values,
            position_embeddings=rope,state_dict=weights,with_lm_head=False)
        self.builder = Qwen3Builder(self.kernel)
        self.builder.build_from_config(config)
        if output_dir:
            Path(output_dir).mkdir(parents=True,exist_ok=True)
        if output_dir and list(Path(output_dir).glob('mpk_launcher_rank0.cpython-*.so')):
            self.kernel.load_mpk_kernel(output_dir)
        else:
            self.kernel.compile(output_dir=output_dir)

    @torch.inference_mode()
    def load_prefix(self,cache,length):
        for i,layer in enumerate(cache.layers):
            self.keys[i,0,:length].copy_(layer.keys[0,:,:length].transpose(0,1))
            self.values[i,0,:length].copy_(layer.values[0,:,:length].transpose(0,1))

    @torch.inference_mode()
    def __call__(self, embedding, position):
        self.input_embedding.copy_(embedding.reshape_as(self.input_embedding))
        self.meta['step'].fill_(position)
        self.meta['paged_kv_last_page_len_buffer'].fill_(position+1)
        self.kernel()
        # online_notoken launches its worker/scheduler on private streams and
        # does not join the caller stream. Its explicit wait is required before
        # PyTorch consumes the hidden state (offline mode joins internally).
        self.kernel.wait()
        return self.builder.returned_hidden_state

    def close(self):
        self.kernel.finalize()
