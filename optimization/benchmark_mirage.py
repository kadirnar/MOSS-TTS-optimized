"""Validate the persistent backbone against upstream before streaming integration."""
import json
import time
import torch
from .common import load_models, RESULTS, stats
from .kernels import embedding_sum
from .mirage_backbone import MirageBackbone


@torch.inference_mode()
def main():
    model,codec,processor = load_models()
    fixture = torch.load(RESULTS/'fixture.pt',weights_only=True)
    ids = fixture['inputs']['input_ids'].cuda()
    prefix = model(input_ids=ids,use_cache=True)
    mpk = MirageBackbone(model,output_dir=str(RESULTS/'mirage_build'))
    mpk.load_prefix(prefix.past_key_values,ids.shape[1])
    emb_weights = torch.stack([e.weight for e in model.emb_ext])
    errors=[]
    top1=[]
    for i in range(36):
        token = fixture['upstream_ids'][i:i+1][None].cuda()
        ref = model(input_ids=token,past_key_values=prefix.past_key_values,use_cache=True)
        embedding = embedding_sum(token,model.get_input_embeddings().weight,emb_weights)
        actual = mpk(embedding,ids.shape[1]+i).clone()
        logits = torch.cat([torch.nn.functional.linear(actual,head.weight[:1024]) for head in model.lm_heads[1:]],0)
        expected = torch.cat([head[:,-1,:1024] for head in ref.logits[1:]],0)
        rel = ((logits.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
        agree = (logits[:min(i+1,32)].argmax(-1)==expected[:min(i+1,32)].argmax(-1)).float().mean().item()
        errors.append(rel); top1.append(agree)
        print('VALIDATE',i,rel,agree,flush=True)
        assert torch.isfinite(logits).all(), 'Nonfinite MPK output'
    times=[]
    for i in range(23):
        torch.cuda.synchronize()
        start=time.perf_counter()
        mpk(embedding,ids.shape[1]+35)
        torch.cuda.synchronize()
        if i>=3:times.append((time.perf_counter()-start)*1000)
    result={'backend':'Mirage persistent Qwen3 backbone','codebooks':32,'dtype':'bfloat16',
        'projection_check_passed':max(errors)<0.05,
        'decode_backbone':stats(times),'max_relative_rms_audio_logits':max(errors),
        'active_codebook_mean_top1_agreement':sum(top1)/len(top1),
        'per_step_relative_rms':errors,'scope':'Backbone only; excludes MOSS embedding sum, audio heads, sampling, prefill and codec. Not TTFA.',
        'quality_gate':'Diagnostic, not deployment acceptance'}
    (RESULTS/'mirage_backbone.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    mpk.close()


if __name__=='__main__':main()
