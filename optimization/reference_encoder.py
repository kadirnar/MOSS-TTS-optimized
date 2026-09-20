"""CUDA graph buckets for uncached voice-reference encoding (FP32)."""
import torch
import torchaudio


class ReferenceEncoder:
    def __init__(self,processor,buckets=(16,32,40,64,128,192)):
        self.processor=processor
        self.codec=processor.audio_tokenizer
        self.buckets=buckets
        self.graphs={}

    @torch.inference_mode()
    def warmup(self):
        for frames in self.buckets:
            wave=torch.zeros((1,1,frames*1920),device='cuda',dtype=torch.float32)
            lengths=torch.full((1,),frames*1920,device='cuda',dtype=torch.long)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):self.codec._encode_frame(wave,lengths,n_quantizers=32)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=self.codec._encode_frame(wave,lengths,n_quantizers=32)
            self.graphs[frames]=(graph,wave,lengths,out)

    @torch.inference_mode()
    def encode(self,wave,sample_rate):
        if wave.ndim!=2:raise ValueError('wave must be [channels,samples]')
        if wave.shape[0]>1:wave=wave.mean(0,keepdim=True)
        if sample_rate!=24000:
            wave=torchaudio.functional.resample(wave,sample_rate,24000)
        wave=self.processor.loudness_normalize(wave.float().to('cuda').squeeze(0))
        n=wave.numel()
        buckets=[b for b in self.graphs if b*1920>=n]
        if not buckets:raise ValueError('Reference exceeds captured encoder capacity')
        graph,static_wave,lengths,out=self.graphs[min(buckets)]
        static_wave.zero_()
        static_wave[0,0,:n].copy_(wave)
        lengths.fill_(n)
        graph.replay()
        # Upstream length propagation floors through patching even though the
        # waveform buffer is padded; preserve that exact reference contract.
        valid=int(out.audio_codes_lengths[0].item())
        return out.audio_codes[:,0,:valid].T.contiguous().cpu()
