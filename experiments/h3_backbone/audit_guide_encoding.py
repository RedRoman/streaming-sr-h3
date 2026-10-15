"""Official AddGuide parity and clearly labeled HR-oracle guide diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--comfy',type=Path,required=True)
    p.add_argument('--vae',type=Path,required=True)
    p.add_argument('--context',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.comfy))
    from comfy.cli_args import args as comfy_args
    comfy_args.reserve_vram=20.0
    import comfy.sd,comfy.utils
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide,_empty_av_latent,_resize,FPS,AUDIO_LATENT_FPS
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN,FRAME_RESCALE,VISUAL_COND_TIMESTEP
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.5)
    torch.cuda.reset_peak_memory_stats()
    began=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=True)
    rgb=np.array(Image.open(args.image).convert('RGB').resize((256,256),Image.Resampling.BICUBIC))
    hr=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255
    lr=F.interpolate(hr,size=(64,64),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    up=F.interpolate(lr,size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    cache=torch.load(args.context,map_location='cpu',weights_only=True)
    base=[[cache['prompt_embeds'][None],{'minimax_token_tags':cache['text_token_tags']}]]
    sd,metadata=comfy.utils.load_torch_file(str(args.vae),return_metadata=True)
    vae=comfy.sd.VAE(sd=sd,device=torch.device('cuda'),dtype=torch.float16,metadata=metadata)
    del sd
    cases=[]
    for frames,quality,mode in [(5,'lr','full'),(5,'hr_oracle','full'),(22,'lr','full'),(22,'hr_oracle','full'),(22,'lr','first'),(22,'hr_oracle','first')]:
        name=f'f{frames}_{quality}_{mode}'
        out=args.output/name;out.mkdir(exist_ok=True)
        pixels=(up if quality=='lr' else hr)[0].permute(1,2,0)[None]
        if mode=='full':pixels=pixels.repeat(frames,1,1,1)
        latent,actual_frames=_empty_av_latent(256,256,frames)
        with torch.no_grad():
            official=MiniMaxH3AddGuide.execute(positive=base,latent=latent,frame_idx=0,vae=vae,image=pixels).result[0]
            z=official[0][1]['minimax_keyframes'][0]['latent']
            resized=_resize(pixels,256,256,'center')
            direct=vae.encode(resized)
        difference=float((z.float()-direct.float()).abs().max())
        identity_resize=float((pixels-resized).abs().max())
        print(json.dumps({"event":"parity_probe","case":name,"latent_maxabs":difference,"resize_maxabs":identity_resize}),flush=True)
        assert difference<1e-5
        save_file({'guide':z.contiguous().cpu()},str(out/'guide.safetensors'))
        Image.fromarray(rgb).save(out/'source_hr_for_evaluation_only.png')
        Image.fromarray((lr[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(out/'observed_lr.png')
        Image.fromarray((up[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(out/'lr_upsampled_guide.png')
        result={'case':name,'scope':'Guide interface diagnosis only. HR-oracle is forbidden as a causal SR method input.',
                'target_frames':actual_frames,'guide_input_frames':len(pixels),'guide_shape':list(z.shape),
                'guide_quality':quality,'guide_mode':mode,'oracle_HR_guide':quality!='lr',
                'official_AddGuide_execute_vs_direct_encode_max_abs':difference,'official_lanczos_center_same_canvas_max_abs':identity_resize,
                'guide_encoding':'Actual MiniMaxH3AddGuide.execute with standard comfy.sd.VAE; raw0..1RGB, native wrapper normalization; latent formatscale1',
                'frame_idx':0,'fps':FPS,'audio_latent_fps':AUDIO_LATENT_FPS,'frame_per_token':FRAME_PER_TOKEN,
                'frame_rescale':FRAME_RESCALE,'visual_cond_timestep':VISUAL_COND_TIMESTEP,
                'context_sha256':hashlib.sha256(args.context.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),
                'source_split':'BSDS500 official train','source_image':str(args.image),'guide_sha256':hashlib.sha256((out/'guide.safetensors').read_bytes()).hexdigest()}
        (out/'guide_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        cases.append(result);print(json.dumps(result),flush=True)
    source_hashes={str(p.relative_to(args.comfy)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.comfy/'comfy_extras/nodes_minimax_h3.py',args.comfy/'comfy/model_base.py',args.comfy/'comfy/latent_formats.py',args.comfy/'comfy/ldm/minimax/model.py',args.comfy/'comfy/ldm/minimax/vae.py']}
    summary={'status':'complete','cases':cases,'source_hashes':source_hashes,'source_commit':None,'comfy_version':'0.39.0','seconds':time.perf_counter()-began,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    (args.output/'encoding_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'status':'complete','cases':len(cases),'seconds':summary['seconds']}),flush=True)


if __name__=='__main__':main()
