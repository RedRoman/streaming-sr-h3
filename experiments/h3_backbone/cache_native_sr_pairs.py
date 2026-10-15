"""Cache real train-video pairs for native H3 SR adaptation; never test fitting."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import types
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--comfy',type=Path,required=True);p.add_argument('--vae',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['train','val'],default='train');a=p.parse_args()
    sys.path.insert(0,str(a.comfy));import comfy.sd,comfy.utils
    source=a.comfy/'comfy_extras/nodes_minimax_h3.py';node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_resize')
    namespace={'comfy':types.SimpleNamespace(utils=comfy.utils)};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace);resize=namespace['_resize']
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.4)
    state,metadata=comfy.utils.load_torch_file(str(a.vae),return_metadata=True);vae=comfy.sd.VAE(sd=state,device=torch.device('cuda'),dtype=torch.float16,metadata=metadata);del state
    cases=[c for c in json.loads(a.manifest.read_text())['cases'] if c['split']==a.split]
    assert len(cases)==(8 if a.split=='train' else 4) and len({c['sequence'] for c in cases})==(4 if a.split=='train' else 2)
    scope=f'Frozen native H3 real {a.split} SR pairs only; validation is supervision/evaluation only and forbidden for optimizer updates; no test fitting'
    a.out.mkdir(parents=True,exist_ok=True);records=[];start=time.perf_counter()
    load=lambda f:torch.from_numpy(np.array(Image.open(f).convert('RGB'))).permute(2,0,1).float()/255
    with torch.inference_mode():
        for case in cases:
            paths=[a.data/p for p in case['lr_paths']];hrpaths=[a.data/p for p in case['hr_paths']]
            observed_sha=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            assert observed_sha==case['observed_lr_sha256'] and len(paths)==22
            low=torch.stack([load(p) for p in paths]);target=torch.stack([load(p) for p in hrpaths])
            up=F.interpolate(low,size=case['shape_hw'],mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            pixels=resize(up.permute(0,2,3,1),case['shape_hw'][1],case['shape_hw'][0],'center')
            guide=vae.encode(pixels).cpu().contiguous();truth=vae.encode(target.permute(0,2,3,1)).cpu().contiguous()
            assert guide.shape==truth.shape and guide.shape[2]==7 and torch.isfinite(guide).all() and torch.isfinite(truth).all()
            guidepath=a.out/(case['id']+'_lr_guide.safetensors');targetpath=a.out/(case['id']+'_hr_supervision_target.safetensors')
            save_file({'guide':guide},str(guidepath));save_file({'guide':truth},str(targetpath))
            record={**case,'guide':str(guidepath),'target':str(targetpath),'shape':list(guide.shape),'guide_sha256':hashlib.sha256(guidepath.read_bytes()).hexdigest(),'target_sha256':hashlib.sha256(targetpath.read_bytes()).hexdigest(),
                    'target_HR_used_only_for_supervision':True,'conditioning_HR':False,'input_prefix':[0,21],'bounded_block_wait_frames':21,'seconds_total':time.perf_counter()-start}
            records.append(record);(a.out/'pairs_manifest.json').write_text(json.dumps({'scope':scope,'cases':records,'vae_storage':'canonicalFP16','guide_preprocess':'TorchAA bicubic followedexactofficiallanczoscenteruint8helper','target':'HRstandardVAEencode supervision only','complete':False},indent=2)+'\n')
            print(json.dumps({'event':'train_pair_cached','id':case['id'],'shape':list(guide.shape)}),flush=True)
    result={'scope':scope,'cases':records,'vae_storage':'canonicalFP16','guide_preprocess':'TorchAA bicubic followedexactofficiallanczoscenteruint8helper','target':'HRstandardVAEencode supervision only','complete':True,'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    (a.out/'pairs_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'complete','cases':len(records),'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
