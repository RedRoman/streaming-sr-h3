"""Unchanged checkpoint RGB evidence for qualitative/independent metric audit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from connection_model import ConnectionHead,rollout,embed
from train_connection import KINDS
from h3_interface import load_frozen_vae
from observation_quotient import ObservationQuotient


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--comfy',required=True);p.add_argument('--vae',required=True);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(Path(args.comfy).resolve()))
    root=Path(args.root);manifest=json.loads((root/'manifest.json').read_text());saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);vae,_=load_frozen_vae(args.vae);q=ObservationQuotient();heads={k:ConnectionHead(v).cuda().eval() for k,v in KINDS.items()}
    for k,head in heads.items():head.load_state_dict(saved['heads'][k])
    def cuda(value):
        if torch.is_tensor(value):return value.cuda()
        if isinstance(value,dict):return {k:cuda(v) for k,v in value.items()}
        return value
    result={}
    with torch.no_grad():
        for source in manifest['sources']:
            if source['split']!='val':continue
            folder=root/source['id'];item=cuda(torch.load(folder/'learning_inputs.pt',map_location='cpu',weights_only=False));initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);ensemble=torch.load(folder/'ensemble_control.pt',map_location='cpu',weights_only=False);target=torch.load(folder/'supervision.pt',map_location='cpu',weights_only=False)['target'].permute(2,0,1)[None].float()/255
            outputs={};panel=[('GT',target),('X0',item['rgb_reference'].cpu()),('Classical joint near',initial['classical_rgb']['AB']),('H3 response mean',sum(ensemble['rgb_candidates']['AB'])/2)]
            for name,head in heads.items():
                state0,ab=rollout(head,item,['A','B']);_,ba=rollout(head,item,['B','A']);rgbs={}
                for key,state in {**ab,**ba}.items():
                    decoded=vae.decode(embed(item,state,state0).half())[:,:,21].cuda().float();rgbs[key]=q.bounded(decoded,item['observations']['far']).cpu()
                outputs[name]=rgbs;panel.append((name,rgbs['AB']))
            tensor={'source':source,'target_evaluation_only':target,'reference':item['rgb_reference'].cpu(),'classical_AB':initial['classical_rgb']['AB'],'H3_response_mean_AB':panel[3][1],'outputs':outputs,'observations':{k:v.cpu() for k,v in item['observations'].items()},'private_mask':item['private_mask'].cpu(),'global_query_mask':item['global_query_mask'].cpu()};torch.save(tensor,folder/'eval_readouts.pt')
            image=Image.new('RGB',(1024,568),'white');draw=ImageDraw.Draw(image)
            for n,(label,rgb) in enumerate(panel):
                x=(n%4)*256;y=(n//4)*284;draw.text((x+8,y+8),label,fill='black');pixels=(rgb[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8);image.paste(Image.fromarray(pixels),(x,y+28))
            image.save(folder/'comparison_panel.png');result[source['id']]={'panel':str(folder/'comparison_panel.png'),'readouts':str(folder/'eval_readouts.pt')}
    (root/'exported_evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
