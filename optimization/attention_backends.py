"""Adapters for actual SGLang and vLLM decode kernels, not their schedulers."""
import torch


class FlashInferDecodeBackend:
    """One graph-safe HND page matches the existing static cache without copying."""
    def __init__(self,length,tensor_cores=False):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        self.workspace=torch.zeros(128*1024*1024,device='cuda',dtype=torch.uint8)
        self.indptr=torch.tensor([0,1],device='cuda',dtype=torch.int32)
        self.indices=torch.zeros(1,device='cuda',dtype=torch.int32)
        self.last=torch.full((1,),length,device='cuda',dtype=torch.int32)
        self.wrapper=BatchDecodeWithPagedKVCacheWrapper(self.workspace,kv_layout='HND',
            use_cuda_graph=True,use_tensor_cores=tensor_cores,
            paged_kv_indptr_buffer=self.indptr,paged_kv_indices_buffer=self.indices,
            paged_kv_last_page_len_buffer=self.last,backend='fa2')
        self.wrapper.plan(self.indptr,self.indices,self.last,32,8,128,length,
            q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,disable_split_kv=True)

    def __call__(self,q,k,v,position):
        torch.add(position,1,out=self.last)
        return self.wrapper.run(q.reshape(1,32,128),(k,v)).reshape(1,1,4096)


class DecodeBackend:
    def __init__(self,name,length):
        self.name=name
        self.length=length
        self.indices=torch.arange(length,device='cuda',dtype=torch.int32)
        self.zero=torch.zeros(1,device='cuda',dtype=torch.int32)
        self.splits=torch.full((1,),8,device='cuda',dtype=torch.int32)
        self.scale=torch.ones(1,device='cuda',dtype=torch.float32)
        self.tables=torch.arange(length//32,device='cuda',dtype=torch.int32)[None]
        if name=='sglang':
            from .vendor.sglang_decode_attention import decode_attention_fwd
            self.forward=decode_attention_fwd
        elif name=='vllm':
            import vllm._C
        else:raise ValueError(name)

    def __call__(self,q,k,v,position):
        query=q.reshape(1,32,128)
        out=torch.empty_like(query)
        lengths=(position+1).to(torch.int32)
        if self.name=='sglang':
            ptr=torch.cat([self.zero,lengths])
            partial=torch.empty((1,32,8,128),device=q.device,dtype=torch.float32)
            lse=torch.empty((1,32,8),device=q.device,dtype=torch.float32)
            self.forward(query,k[0].transpose(0,1),v[0].transpose(0,1),out,ptr,self.indices,
                partial,lse,self.splits,8,128**-0.5)
        else:
            blocks=self.length//32
            keys=k[0].view(8,blocks,32,16,8).permute(1,0,3,2,4).contiguous()
            vals=v[0].view(8,blocks,32,128).permute(1,0,3,2).contiguous()
            torch.ops._C.paged_attention_v1(out,query,keys,vals,8,128**-0.5,
                self.tables,lengths,32,self.length,None,'auto',self.scale,self.scale,0,0,0,64,0)
        return out.reshape(1,1,4096)
