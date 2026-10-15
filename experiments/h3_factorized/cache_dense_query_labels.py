"""TRAIN-only query-LR neutral counterparts, auxiliary labels not runtime state."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from native_vae import load_vae,encode_repeated
from cache_revision import inverse_photo


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());sources=[x for x in manifest['items'] if x['split']=='train'];assert len(sources)==32;vae=load_vae(a.vae,a.comfy);start=time.time();records={};calls=0
    with torch.no_grad():
        for i,source in enumerate(sources):
            path=out/(source['id']+'.pt')
            if not path.exists():
                item=torch.load(cache/(source['id']+'.pt'),map_location='cpu',weights_only=False);ys=item['classical_inputs']['query_LR'].cuda().float();qs=item['supervision']['query_q'].cuda().float();zs=[]
                for y,q in zip(ys,qs):
                    up=F.interpolate(y[None],size=(256,256),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1);neutral=inverse_photo(up,q);zs.append(encode_repeated(vae,neutral[None])[0].cpu().half());calls+=1
                torch.save(dict(source=source['id'],query_neutral_latents=torch.stack(zs),provenance='Training-only nuisance-supervised inverse photometry of this current actual query LR, then native E. No HR texture, no historical near packet or generated observation; not an inference input.'),path)
            records[source['id']]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest());print(json.dumps(dict(source=source['id'],step=i+1,steps=32,seconds=time.time()-start)),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(config=vars(a),seconds=time.time()-start,actual_E_calls_this_run=calls,sources=records,scope='Only32TRAIN; known true training nuisance is supervision only. Mainruntime stillLR-only.',test_accessed=False),indent=2))


if __name__=='__main__':main()
