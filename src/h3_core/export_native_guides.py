"""Export equal-token native H3 LR and canonical-memory condition clips.

Only packets available at each block endpoint enter either guide. The memory
state is read, decoded, warped into the observed camera and re-encoded. The
generated clip never writes the evidence ledger. Geometry/visibility is oracle
in this controlled-camera export, not inferred real-video correspondence.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from h3_data import NaturalCameraSequence
from h3_interface import load_frozen_vae,memory_clip_guide,stream_blocks,guide_payload
from memory_adapter import CanonicalChart,SpectralDetailTransport
from nonlinear_memory import NonlinearMemoryHead


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--cache-item',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True)
    p.add_argument('--appearance',action='store_true');args=p.parse_args();sys.path.insert(0,str(Path(args.comfy).resolve()));torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);item=torch.load(args.cache_item,map_location='cpu',weights_only=False)
    chart=CanonicalChart(temporal=7,basis_mode=checkpoint['config']['basis_mode']).cuda();chart.load_state_dict(checkpoint['chart']);chart.eval().requires_grad_(False)
    heads={k:NonlinearMemoryHead(k).cuda().eval() for k in checkpoint['heads']}
    for kind,head in heads.items():head.load_state_dict(checkpoint['heads'][kind]);head.requires_grad_(False)
    z={k:v.cuda() if torch.is_tensor(v) else v for k,v in item['inputs'].items()};y,x=z['roi']
    vae,vae_info=load_frozen_vae(args.vae);sequence=NaturalCameraSequence(item['source']['path'],size=256,frames=39,protocol=item['source']['protocol'],seed=item['source']['sequence_seed'])
    manifest={'config':vars(args),'scope':'Actual frozen H3 E/D + native condition payload export; no DiT sampling or restoration claim','vae':vae_info,'oracle_inputs':['camera rectangle','visibility'],'blocks':[]}
    with torch.no_grad():
        matrix,prior,context=chart(z['anchor'][:,:,:,y:y+4,x:x+4]);scale=checkpoint['scale'].cuda()
        generator=torch.Generator(device='cuda').manual_seed(checkpoint['config']['seed']+50000);noise=torch.randn(1,8,generator=generator,device='cuda')
        common=(matrix,prior,context,z['grams'][None],z['rhs'][None],z['packet_counts'][None])
        predictions={k:h(*common,noise,scale)[0] for k,h in heads.items()}
        predictions['map']=SpectralDetailTransport().cuda()(*common,mode='map',identity_noise=noise)[0]
        predictions['anchor_only']=torch.zeros_like(predictions['map'])
        for block in stream_blocks():
            start=time.time();packets=[sequence.render(t)[0] for t in range(block.start,block.stop)]
            frames=[]
            for packet in packets:
                lr=torch.as_tensor(packet.rgb.copy(),device='cuda').permute(2,0,1)[None].float()/255
                frames.append(F.interpolate(lr,size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1))
            baseline=vae.encode((2*torch.stack(frames,2)-1).half().clone())
            guides={'lr_only':baseline};details={}
            for kind,prediction in predictions.items():
                canonical=chart.embed(z['anchor'],matrix,prediction[:,block.stop-1],z['roi'])
                rgb=vae.decode(canonical.half())[:,:,item['source']['output_frame']].cuda()
                latent,detail=memory_clip_guide(vae,rgb,item['source']['canonical_rectangle'],packets,appearance=args.appearance)
                guides[kind]=latent;details[kind]=detail
            tokens={k:v.shape[2]*(v.shape[-2]//2)*(v.shape[-1]//2) for k,v in guides.items()}
            if len(set(tokens.values()))!=1:raise ValueError('Guide token budgets differ')
            # All native payloads anchor the same full guide clip at target0.
            payloads={k:guide_payload([(0,v,block.stop-1)],block) for k,v in guides.items()}
            path=out/f'block_{block.start}_{block.stop}.pt'
            torch.save({'guides':{k:v.cpu() for k,v in guides.items()},'keyframes':{k:[{'resolved_frame_index':0,'latent':v.cpu()}] for k,v in guides.items()},'block':vars(block),'details':details},path)
            entry={'start':block.start,'stop':block.stop,'emit_start':block.emit_start,'max_input_wait_frames':block.max_input_wait_frames,'state_prefix_stop':block.stop-1,'guide_tokens':tokens,'guide_clips_per_method':1,'path':str(path),'compute_seconds':time.time()-start}
            manifest['blocks'].append(entry);print(json.dumps(entry),flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
