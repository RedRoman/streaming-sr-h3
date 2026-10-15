"""Cache actual H3 VAE latents and decoder camera responses, then stop.

No DiT, future HR, or generated guide writes the evidence ledger. Ground truth
is separately saved as supervision. The first stage is an explicit small pilot.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from h3_data import NaturalCameraSequence
from camera_operator import camera_footprint,nuisance_project
from memory_adapter import CanonicalChart,PacketLedger
from h3_interface import load_frozen_vae,decoder_teacher


def tensor_rgb(array,device):
    return torch.from_numpy(np.asarray(array).copy()).to(device).permute(2,0,1)[None].float()/255


def main():
    p=argparse.ArgumentParser();p.add_argument('--vae',required=True);p.add_argument('--comfy',default='external/ComfyUI_H3')
    p.add_argument('--data',default='data/BSR/BSDS500/data/images');p.add_argument('--out',default='results/h3_core/cache')
    p.add_argument('--counts',nargs=3,type=int,default=[8,4,8]);p.add_argument('--size',type=int,default=128)
    p.add_argument('--frames',type=int,default=39);p.add_argument('--atoms',type=int,default=32);p.add_argument('--epsilon',type=float,default=.1)
    p.add_argument('--effective-noise',type=float,default=.005);p.add_argument('--nuisance',action='store_true')
    p.add_argument('--context-frames',type=int,choices=[1,5,22],default=1)
    p.add_argument('--output-frame',type=int,default=0)
    p.add_argument('--span')
    p.add_argument('--teacher-dtype',choices=['float16','float32'],default='float16')
    p.add_argument('--strict-fp32',action='store_true',help='Disable TF32 for distributed small decoder perturbations; independently audited, no weight change')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(args.comfy).resolve()));torch.set_num_threads(4)
    if args.strict_fp32:
        if args.teacher_dtype!='float32':raise ValueError('Strict FP32 requires float32 teacher compute')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    vae,vae_info=load_frozen_vae(args.vae)
    decoder_dtype=torch.float32 if args.teacher_dtype=='float32' else torch.float16
    # Keep the actual native encoder at FP16; only the decoder derivative
    # teacher computes at FP32 using the SAME FP16-stored weight values.
    vae.decoder.to(dtype=decoder_dtype);vae.post_quant_conv.to(dtype=decoder_dtype)
    vae_info['encoder_compute']='float16';vae_info['decoder_teacher_compute']=args.teacher_dtype
    vae_info['tf32']={'matmul':torch.backends.cuda.matmul.allow_tf32,'cudnn':torch.backends.cudnn.allow_tf32}
    manifest={'config':vars(args),'vae':vae_info,'protocol':f'Native {args.context_frames}-frame encoding of copies of the already observed anchor (no future input); canonical 4x4-token central ROI; rank32 fixed decoder span; camera+visibility oracle; no DiT rollout yet',
              'limits':['Decoder finite differences are a local approximation.','PIL vs torch degradation mismatch is included in effective noise, but must be measured.','39 frames is a short pilot, not long revisit evidence.'], 'items':[]}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    temporal={1:1,5:2,22:7}[args.context_frames]
    chart=CanonicalChart(atoms=args.atoms,temporal=temporal).cuda();start=time.time()
    if args.span:
        span=torch.load(args.span,map_location='cpu',weights_only=False)
        if span['p0'].shape!=chart.p0.shape:raise ValueError('Span dimensions differ from the declared temporal ROI')
        chart.p0.copy_(span['p0'].cuda());manifest['span']={'path':args.span,'train_energy_fraction':span['train_energy_fraction']}
    all_ids={split:sorted((Path(args.data)/split).glob('*.jpg')) for split in ['train','val','test']}
    assert not {x.name for x in all_ids['train']}&{x.name for x in all_ids['test']}
    assert not {x.name for x in all_ids['train']}&{x.name for x in all_ids['val']}
    with torch.no_grad():
        for split,count in zip(['train','val','test'],args.counts):
            for source_number,path in enumerate(all_ids[split][:count]):
                item_id=f'{split}_{path.stem}';destination=out/f'{item_id}.pt'
                if destination.exists():raise FileExistsError(f'Refusing to silently replace teacher cache: {destination}')
                protocol=['zoom_revisit','zoom_lighting','revisit'][source_number%3]
                sequence=NaturalCameraSequence(path,size=args.size,frames=args.frames,protocol=protocol,seed=17100+source_number)
                first_packet,first_gt,_=sequence.render(0)
                anchor_rgb=F.interpolate(tensor_rgb(first_packet.rgb,'cuda'),size=(args.size,args.size),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
                anchor_video=(2*anchor_rgb-1).unsqueeze(2).repeat(1,1,args.context_frames,1,1)
                target_video=(2*tensor_rgb(first_gt,'cuda')-1).unsqueeze(2).repeat(1,1,args.context_frames,1,1)
                anchor=vae.encode(anchor_video.half().clone()).float()
                target=vae.encode(target_video.half().clone()).float()
                latent_size=anchor.shape[-1];roi=((latent_size-4)//2,)*2
                base,responses=decoder_teacher(vae,anchor.to(decoder_dtype),chart.p0,roi,args.epsilon,args.output_frame)
                roundtrip_mse=float((base-anchor_rgb).square().mean())
                roundtrip={'input_range':[float(anchor_rgb.min()),float(anchor_rgb.max())],
                           'latent_range':[float(anchor.min()),float(anchor.max())],
                           'decoded_range':[float(base.min()),float(base.max())],
                           'psnr_db':float(-10*np.log10(max(roundtrip_mse,1e-12))),
                           'native_input':'RGB -> 2*RGB-1; native encode internally ImageNet normalizes and standardizes latents',
                           'native_output':'native decode restores latent mean/std, ImageNet mean/std, then clamps RGB [0,1]'}
                # Half-step audit of one direction checks finite-difference scale.
                half_base,half_response=decoder_teacher(vae,anchor.to(decoder_dtype),chart.p0[:,:1],roi,args.epsilon/2,args.output_frame)
                fd_relative=float((responses[:1]-half_response).norm()/half_response.norm().clamp_min(1e-8))
                audit_kwargs=dict(canonical_rectangle=(0,0,args.size,args.size),view_rectangle=(0,0,args.size,args.size),view_size=(args.size,args.size),lr_size=(args.size//4,)*2)
                lr_response=camera_footprint(responses[:1],**audit_kwargs);lr_half=camera_footprint(half_response,**audit_kwargs)
                fd_lr_relative=float((lr_response-lr_half).norm()/lr_half.norm().clamp_min(1e-8))
                ledger=PacketLedger(args.atoms,device='cuda');gram_prefix=[];rhs_prefix=[];counts=[];indices=[];mismatch=[]
                original_rect=first_packet.world_rectangle
                for packet in sequence.packets():
                    kwargs=dict(canonical_rectangle=original_rect,view_rectangle=packet.world_rectangle,view_size=(args.size,args.size),lr_size=(args.size//4,)*2)
                    predicted=camera_footprint(base,**kwargs)
                    design_images=camera_footprint(responses,**kwargs)
                    coverage=camera_footprint(torch.ones_like(base[:,:1]),**kwargs)
                    visible=torch.from_numpy(packet.visible_mask.copy()).cuda().float()/255
                    valid=(coverage[0,0]>.999)&(coverage[0,0]<1.001)&(visible>.5)
                    mask=valid[None].expand(3,-1,-1)
                    observed=tensor_rgb(packet.rgb,'cuda')[0]
                    residual=(observed-predicted[0])[mask]
                    design=design_images[:,mask].T
                    if args.nuisance:
                        channel_bias=torch.eye(3,device='cuda')[:,:,None,None].expand(-1,-1,*valid.shape)
                        channel_gain=channel_bias*predicted
                        nuisance=torch.cat([channel_bias,channel_gain],0)[:,mask].T
                        design,residual=nuisance_project(design,residual,nuisance)
                    ledger.write(design,residual,packet.observation_id,packet.index,args.effective_noise**2)
                    snapshot=ledger.snapshot();gram_prefix.append(snapshot.gram.cpu());rhs_prefix.append(snapshot.rhs.cpu());counts.append(snapshot.packets);indices.append(packet.index)
                    # Oracle HR is used here solely to AUDIT the simulator/operator mismatch.
                    _,hr,_=sequence.render(packet.index)
                    if packet.index==0:
                        correct=camera_footprint(tensor_rgb(first_gt,'cuda'),**kwargs)[0]
                        mismatch.append(float((correct-observed).square().mean().sqrt()))
                y,x=roi
                target_delta=(target-anchor)[:,:,:,y:y+4,x:x+4].flatten(1)
                target_p0=target_delta@chart.p0
                packed={'inputs':{'anchor':anchor.cpu(),'roi':roi,'grams':torch.stack(gram_prefix),'rhs':torch.stack(rhs_prefix),'packet_counts':torch.tensor(counts),'indices':torch.tensor(indices)},
                        'supervision':{'target_latent':target.cpu(),'target_p0':target_p0.cpu(),'canonical_hr':torch.from_numpy(first_gt.copy())},
                        'teacher':{'base_rgb':base.cpu(),'decoder_responses':responses.cpu(),'p0':chart.p0.cpu(),'finite_difference_relative_halfstep':fd_relative,'lr_finite_difference_relative_halfstep':fd_lr_relative,'anchor_lr_operator_rmse':mismatch,'roundtrip':roundtrip},
                        'source':{'path':str(path),'split':split,'source_hash':sequence.source_hash,'protocol':protocol,'sequence_seed':sequence.seed,'canonical_rectangle':original_rect,'output_frame':args.output_frame,'context_frames':args.context_frames}}
                torch.save(packed,destination)
                item={'id':item_id,'path':str(destination),'fd_relative_halfstep':fd_relative,'lr_fd_relative_halfstep':fd_lr_relative,'anchor_lr_operator_rmse':mismatch[0],'roundtrip_psnr_db':roundtrip['psnr_db'],'seconds':time.time()-start}
                manifest['items'].append(item);(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
                print(json.dumps({'event':'cached',**item}),flush=True)
    print(json.dumps({'event':'complete','items':len(manifest['items']),'seconds':time.time()-start}),flush=True)


if __name__=='__main__':main()
