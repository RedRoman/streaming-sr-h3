"""Decode fixed native H3 pilot readouts using the canonical frozen FP16 VAE."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for key in ['comfy','vae','run','val']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=['unadapted','lora_only','guide_only','joint'],default=['unadapted','lora_only','guide_only','joint'])
    a=p.parse_args();gpu=os.environ.get('CUDA_VISIBLE_DEVICES','');assert gpu.isdigit()
    free,util=map(int,subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
    assert free>30000 and util==0,(free,util)
    assert sha(a.vae)=='7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522'
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35);torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0,str(a.comfy));import comfy.sd,comfy.utils
    state,meta=comfy.utils.load_torch_file(str(a.vae),return_metadata=True)
    vae=comfy.sd.VAE(sd=state,device=torch.device('cuda'),dtype=torch.float16,metadata=meta);del state
    cases=json.loads(a.val.read_text())['cases'];assert len(cases)==4 and all(c['split']=='val' for c in cases)
    records=[]
    with torch.no_grad():
        for case in cases:
            base=a.run/'readouts'/'vae_only'/case['id'];base.mkdir(parents=True,exist_ok=True)
            jobs=[(base,Path(case['guide']),'guide')]
            for mode in a.modes:
                folder=a.run/'readouts'/mode/case['id'];result=json.loads((folder/'sample_result.json').read_text());assert result['status']=='complete'
                jobs.append((folder,folder/'sampled_av_latents.safetensors','video'))
            for folder,latent,key in jobs:
                target=folder/'decode_result.json'
                if target.exists() and json.loads(target.read_text()).get('status')=='complete':continue
                began=time.perf_counter();pixels=vae.decode(load_file(str(latent))[key]).float().cpu()
                if pixels.ndim==5:pixels=pixels[0]
                assert pixels.shape==(22,*case['shape_hw'],3) and torch.isfinite(pixels).all()
                files={}
                for i,frame in enumerate(pixels):
                    file=folder/f'output_{i:02d}.png';Image.fromarray(np.uint8(frame.numpy().clip(0,1)*255+.5)).save(file);files[file.name]=sha(file)
                result={'status':'complete','scope':'Frozen canonicalFP16VAE decode; no HR pixels or supervision read during decode','case':case['id'],'mode':folder.parent.name,
                        'shape':list(pixels.shape),'range':[float(pixels.min()),float(pixels.max())],'vae_precision':'FP16storageandcompute','torch':torch.__version__,'python':sys.version,
                        'vae_sha256':'7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522','latent_sha256':sha(latent),'output_png_sha256':files,'seconds':time.perf_counter()-began}
                target.write_text(json.dumps(result,indent=2)+'\n');records.append(result);print(json.dumps({'event':'pilot_decode_complete','case':case['id'],'mode':folder.parent.name,'seconds':result['seconds']}),flush=True)
    result={'status':'complete','modes_this_execution':a.modes,'jobs_newly_decoded':len(records),'expected_output_cases':4*(1+len(a.modes)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    (a.run/'decode_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
