"""Export calibrated INT4 projections without modifying the HF checkpoint."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
from .common import RESULTS,TTS_REVISION
from .calibrated_quant import gptq,dequantize,static_scales


def load_projection(snapshot,index,layer,projection):
    names={'qkv':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
           'out':['self_attn.o_proj'],'up':['mlp.gate_proj','mlp.up_proj'],'down':['mlp.down_proj']}[projection]
    weights=[]
    for name in names:
        key=f'language_model.layers.{layer}.{name}.weight'
        with safe_open(str(snapshot/index[key]),framework='pt',device='cuda') as handle:
            weights.append(handle.get_tensor(key))
    return torch.cat(weights,0) if len(weights)>1 else weights[0]


def split_rows(manifest):
    train=[];valid=[];offset=0
    for record in manifest['records']:
        target=valid if record['id'].endswith('_2') else train
        target.extend(range(offset,offset+record['decode_tokens']))
        offset+=record['decode_tokens']
    return torch.tensor(train),torch.tensor(valid)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='gptq_v1_g32_d01')
    parser.add_argument('--layers',type=int,nargs='+',default=list(range(36)))
    parser.add_argument('--projections',nargs='+',choices=('qkv','out','up','down'),default=['qkv','out','up','down'])
    parser.add_argument('--group',type=int,choices=(32,64,128),default=32)
    parser.add_argument('--damping',type=float,default=.01)
    parser.add_argument('--scale-search',choices=('max','mse','diagonal'),default='max')
    args=parser.parse_args()
    if not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Unsafe tag')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    folder=RESULTS/args.tag;folder.mkdir(exist_ok=True)
    calibration=RESULTS/'calibration_v1'
    manifest=json.loads((calibration/'manifest.json').read_text())
    train,valid=split_rows(manifest)
    config={'group':args.group,'damping':args.damping,'actorder':True,'static_bf16_scales':True,
        'code_range':[-7,7],'tts_revision':TTS_REVISION,'calibration_sha256':hashlib.sha256((calibration/'manifest.json').read_bytes()).hexdigest(),
        'training_rows':train.numel(),'heldout_rows':valid.numel(),'torch':torch.__version__,
        'training_ids':[r['id'] for r in manifest['records'] if not r['id'].endswith('_2')],
        'heldout_ids':[r['id'] for r in manifest['records'] if r['id'].endswith('_2')]}
    if args.scale_search!='max':config['scale_search']={'mode':args.scale_search,'steps':33,'minimum_clip_fraction':.5,'loss':'BF16-rounded static reconstruction squared error, optionally weighted by training activation energy'}
    config_path=folder/'config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Configuration differs from existing export'
    else:config_path.write_text(json.dumps(config,indent=2)+'\n')
    snapshot=Path(snapshot_download('OpenMOSS-Team/MOSS-TTS-v1.5',revision=TTS_REVISION,local_files_only=True))
    index=json.loads((snapshot/'model.safetensors.index.json').read_text())['weight_map']
    records=[]
    for layer in args.layers:
        for projection in args.projections:
            name=f'{layer:02d}_{projection}'
            path=folder/(name+'.pt')
            stats_path=folder/(name+'.json')
            if path.exists() and stats_path.exists():
                row=json.loads(stats_path.read_text());records.append(row);print('EXISTING',name,flush=True);continue
            weight=load_projection(snapshot,index,layer,projection)
            samples=torch.load(calibration/(name+'.pt'),weights_only=True)
            train_x=samples[train].cuda();valid_x=samples[valid].cuda()
            torch.cuda.synchronize();start=time.perf_counter()
            searched=None
            if args.scale_search!='max':
                from .calibration_scales import search_scales
                searched=search_scales(weight,train_x,args.group,args.scale_search)
            signed,scales=gptq(weight,train_x,group=args.group,damping=args.damping,scales_override=searched)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            quantized=dequantize(signed,scales,args.group)
            rtn=(weight.float().view(weight.shape[0],-1,args.group)/scales.float()[:,:,None]).round().clamp(-7,7).to(torch.int8).reshape_as(weight)
            rounded=dequantize(rtn,scales,args.group)
            row={'name':name,'shape':list(weight.shape),'seconds':seconds,'training_rows':train.numel(),'heldout_rows':valid.numel()}
            for split,x in [('train',train_x),('heldout',valid_x)]:
                reference=torch.nn.functional.linear(x.float(),weight.float())
                def error(w):return ((torch.nn.functional.linear(x.float(),w.float())-reference).square().mean()/reference.square().mean()).item()
                row[split+'_gptq_mse']=error(quantized)
                row[split+'_rtn_mse']=error(rounded)
            unsigned=signed.to(torch.uint8)&15
            packed=(unsigned[:,::2]|(unsigned[:,1::2]<<4)).contiguous()
            torch.save({'packed':packed.cpu(),'scales':scales.cpu(),'shape':list(weight.shape),'group':args.group},path.with_suffix('.tmp'))
            path.with_suffix('.tmp').replace(path)
            row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            stats_path.write_text(json.dumps(row,indent=2)+'\n')
            records.append(row)
            print(row,flush=True)
            del weight,samples,train_x,valid_x,signed,scales,quantized,rtn,rounded,unsigned,packed,reference
    print('COMPLETE',len(records),'requested projections',flush=True)


if __name__=='__main__':main()
