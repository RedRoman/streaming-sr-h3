"""Actual equal-budget H3 condition clips from the projected memory checkpoint.

Canonical RGB is only a readout, never a new observation. Pose/visibility is
oracle in this controlled-source interface. Common current-LR appearance fits
affect rendered guides, not the persistent content coefficients.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from h3_data import LRPacket
from h3_interface import load_frozen_vae,memory_clip_guide,stream_blocks,guide_payload
from memory_adapter import SpectralDetailTransport
from nonlinear_memory import NonlinearMemoryHead
from observation_quotient import ObservationQuotient
from train_quotient import ProjectedPrior


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--item',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(Path(args.comfy).resolve()))
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);item=torch.load(args.item,map_location='cpu',weights_only=False);z={k:v.cuda() if torch.is_tensor(v) else v for k,v in item['inputs'].items()}
    prior=ProjectedPrior(item['teacher']['p0']).cuda();prior.load_state_dict(checkpoint['prior']);prior.eval().requires_grad_(False)
    heads={k:NonlinearMemoryHead(k,rank=32,hidden=27,initialize=True).cuda().eval() for k in checkpoint['heads']}
    for kind,head in heads.items():head.load_state_dict(checkpoint['heads'][kind]);head.requires_grad_(False)
    vae,info=load_frozen_vae(args.vae);q=ObservationQuotient();out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    manifest={'config':vars(args),'vae':info,'scope':'Actual H3 E/D guide export, no DiT sampling result; oraclecamera/visibility','appearance':'same currentLR gain/bias fit for every canonical-guide control, not written into identity','blocks':[]}
    with torch.no_grad():
        mu,context=prior(z['anchor'],z['roi']);matrix=torch.eye(32,device='cuda')[None];scale=torch.ones(1,32,device='cuda');noise=torch.randn(1,32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(checkpoint['config']['seed']+50000))
        common=(matrix,mu,context,z['grams'][None],z['rhs'][None],z['packet_counts'][None]);predictions={k:h(*common,noise,scale)[0] for k,h in heads.items() if k!='current_only'}
        if 'current_only' in heads:
            if 'packet_grams' not in z:raise ValueError('Current-only guide requires corrected per-packet sufficient statistics')
            predictions['current_only']=heads['current_only'](matrix,mu,context,z['packet_grams'][None],z['packet_rhs'][None],z['packet_counts'][None],noise,scale,already_incremental=True)[0]
        predictions['map']=SpectralDetailTransport(rank=32).cuda()(*common,mode='map',identity_noise=noise)[0];predictions['anchor_only']=torch.zeros_like(predictions['map'])
        packets=[LRPacket(v['index'],v['index']/24,v['rgb'].numpy(),v['world_rectangle'],v['visible_mask'].numpy(),v['observation_id']) for v in item['inputs']['packets']]
        for block in stream_blocks():
            start=time.time();known=packets[block.start:block.stop];frames=[]
            for packet in known:
                lr=torch.as_tensor(packet.rgb.copy(),device='cuda').permute(2,0,1)[None].float()/255;frames.append(F.interpolate(lr,(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1))
            guides={'lr_only':vae.encode((2*torch.stack(frames,2)-1).half().clone())};canonical_outputs={};details={}
            for kind,coefficients in predictions.items():
                decoded=vae.decode(prior.embed(z['anchor'],coefficients[:,block.stop-1],z['roi']).half())[:,:,21].cuda()
                canonical=q.bounded(decoded,z['coarse'][block.stop-1:block.stop]);canonical_outputs[kind]=canonical.cpu()
                guides[kind],details[kind]=memory_clip_guide(vae,canonical,item['source']['canonical_rectangle'],known,appearance=True)
            tokens={k:v.shape[2]*(v.shape[-2]//2)*(v.shape[-1]//2) for k,v in guides.items()};assert len(set(tokens.values()))==1
            for latent in guides.values():guide_payload([(0,latent,block.stop-1)],block)
            path=out/f'block_{block.start}_{block.stop}.pt';torch.save({'guides':{k:v.cpu() for k,v in guides.items()},'canonical_rgb':canonical_outputs,'keyframes':{k:[{'resolved_frame_index':0,'latent':v.cpu()}] for k,v in guides.items()},'block':vars(block),'details':details},path)
            entry={'start':block.start,'stop':block.stop,'emit_start':block.emit_start,'max_input_wait_frames':21,'guide_tokens':tokens,'guide_clips_per_method':1,'path':str(path),'seconds':time.time()-start};manifest['blocks'].append(entry);print(json.dumps(entry),flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
