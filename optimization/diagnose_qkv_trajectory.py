"""Stop at the first clustered projection difference on a cloned-voice request."""
import argparse
import json
from pathlib import Path
import types
import torch
from .common import RESULTS
from .benchmark_first_audio_graph import build_engine
from .qkv_cluster_binary import load_bundle
from .qkv_cluster_model import Hidden
from .dp4a_norm_projection import SELECTED
from .llm import FastLLM


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--binary-bundle',type=Path,required=True)
    p.add_argument('--config',default='c8_t2_legacy');a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    folder=RESULTS/f'qkv_trajectory_{a.tag}';folder.mkdir(exist_ok=False)
    engine,_=build_engine(bulk=True,prefill_qkv=True,prefill_pointwise=True,output_weight_prefetch=True)
    fast=engine.llm;choices,launchers=load_bundle(a.binary_bundle);count=0;failure=None
    def audit(*args,**kwargs):
        nonlocal count,failure
        actual=launchers[a.config](*args,**kwargs,debug=True)
        expected=fast._bulk_norm_linear(*args[:6],**SELECTED['qkv'],trigger_mode=fast.projection_pdl['norm_trigger'])
        mismatches=[int((x.reshape(-1).view(torch.uint8)!=y.reshape(-1).view(torch.uint8)).sum()) for x,y in zip(actual[:2],expected,strict=True)]
        if any(mismatches):
            failure={'call':count,'layer':count%36,'decode_call':count//36,'byte_mismatches':mismatches,
                'qkv_values':[[int(i),float(expected[1].flatten()[i]),float(actual[1].flatten()[i])] for i in (actual[1].flatten()!=expected[1].flatten()).nonzero().flatten()[:20]]}
            tensors={name:value.cpu() if torch.is_tensor(value) else value for name,value in zip(
                ('x','residual','norm_weight','eps','w','s','qw','kw','cos','sin','kc','vc','position','head_eps'),args,strict=True)}
            tensors['expected']=[t.cpu() for t in expected];tensors['actual']=[t.cpu() for t in actual]
            torch.save(tensors,folder/'mismatch.pt')
            print('MISMATCH',failure,flush=True)
            raise ArithmeticError('Clustered QKV differs from selected arithmetic')
        count+=1
        if count%360==0:print('CHECKED',count,'projections',flush=True)
        return actual
    fast.hidden=Hidden(fast,audit,choices[a.config]);fast.graph=None;fast._audio_head_buckets=None
    fast._audio_head_count=32;fast.step=types.MethodType(FastLLM.step,fast)
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True);torch.manual_seed(6999)
    try:
        list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=400))
    except ArithmeticError:
        pass
    finally:
        (folder/'result.json').write_text(json.dumps({'checked':count,'failure':failure,'codebooks':32,'config':a.config,'bundle':str(a.binary_bundle)},indent=2)+'\n')
        engine.codec.close()


if __name__=='__main__':main()
