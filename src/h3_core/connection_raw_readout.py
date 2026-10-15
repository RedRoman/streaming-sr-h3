"""Direct official conditional-output / VAE-guide readout, no transplantation.

Structural diagnostic: withdrawing private carrier is explicitly marked. No
training, gain change or extra DiT call. This separates poor latent transport
coordinates from a genuinely poor actual H3 conditional candidate.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from h3_interface import load_frozen_vae
from observation_quotient import ObservationQuotient
from connection_data import SeparableCamera,VIEWS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--comfy',required=True);p.add_argument('--vae',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(Path(a.comfy).resolve()));root=Path(a.root);vae,_=load_frozen_vae(a.vae);q=ObservationQuotient();cams={k:SeparableCamera(v,q) for k,v in VIEWS.items()};result={'scope':'Directnativeconditional/VAEguide strongcontrols; privatecarrier withdrawnexplicitly, noextraDiTqueries/nooptimization. GTonlymetrics.','sources':{}}
    with torch.no_grad():
        for source in json.loads((root/'manifest.json').read_text())['sources']:
            folder=root/source['id'];item=torch.load(folder/'learning_inputs.pt',map_location='cpu',weights_only=False);ensemble=torch.load(folder/'ensemble_control.pt',map_location='cpu',weights_only=False);target=torch.load(folder/'supervision.pt',map_location='cpu',weights_only=False)['target'].cuda().permute(2,0,1)[None].float()/255;reference=item['rgb_reference'].cuda();obs={k:v.cuda() for k,v in item['observations'].items()};private=item['private_mask'].cuda();globalquery=item['global_query_mask'].cuda()
            def decode(z):return q.bounded(vae.decode(z.cuda().half())[:,:,21].cuda().float(),obs['far'])
            def metrics(rgb):
                e=(rgb-target).square();lr=torch.stack([(cams[k].observe(rgb)-obs[k]).square().mean() for k in ['A','B']]).mean();delta=q.null(rgb-reference)
                return {'psnr_full_db':float(-10*torch.log10(e.mean().clamp_min(1e-12))),'psnr_global_query_db':float(-10*torch.log10(e[:,:,globalquery].mean().clamp_min(1e-12))),'observed_prefix_lr_rmse':float(lr.sqrt()),'private_query_revision_rms':float(delta[:,:,private].square().mean().sqrt()),'highfrequency_rms':float(q.null(rgb).square().mean().sqrt())}
            raw=[decode(z[None]) for z in ensemble['latents']['AB']];guide=load_file(str(folder/'response_AB_seed101/guide.safetensors'))['guide'];lr_scores=[sum((cams[k].observe(x)-obs[k]).square().mean() for k in ['A','B']) for x in raw];chosen=int(torch.stack(lr_scores).argmin());result['sources'][source['id']]={'direct_H3_current_mean_RGB':metrics(sum(raw)/2),'direct_H3_current_minLR':metrics(raw[chosen]),'direct_VAE_current_guide':metrics(decode(guide)),'direct_H3_empty_mean':metrics(decode(ensemble['latents']['empty'].mean(0,keepdim=True)))};(root/'direct_readout_controls.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
