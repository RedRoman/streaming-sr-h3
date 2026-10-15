"""Train fixed-budget H3-VAE coordinate adapters; audit with real decoder.

With 1/1/1 source files this is a pipeline test only. This does not fine-tune or
evaluate the H3 DiT. Model state is written only from cached LR sufficient stats.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from h3_data import NaturalCameraSequence
from camera_operator import camera_footprint
from memory_adapter import CanonicalChart,SpectralDetailTransport,PosteriorDetailHead
from h3_interface import load_frozen_vae
from train_nonlinear import masks,rgb_metrics


def get_inputs(item,device):
    return {key:value.to(device) if torch.is_tensor(value) else value for key,value in item['inputs'].items()}


def solve(chart,heads,item,seed,device):
    z=get_inputs(item,device);y,x=z['roi'];matrix,prior,context=chart(z['anchor'][:,:,:,y:y+4,x:x+4])
    generator=torch.Generator(device=device).manual_seed(seed)
    noise=torch.randn(1,8,generator=generator,device=device)
    fresh=torch.randn(1,z['grams'].shape[0],8,generator=generator,device=device)
    fresh[:,0]=noise
    args=(matrix,prior,context,z['grams'][None],z['rhs'][None],z['packet_counts'][None])
    outputs={kind:head(*args,identity_noise=noise) for kind,head in heads.items() if kind!='transport'}
    outputs['transport']=heads['transport'](*args,identity_noise=noise)[0]
    outputs['fixed_reference']=(prior+noise)[None].transpose(0,1).expand(-1,z['grams'].shape[0],-1)
    outputs['anchor_only']=torch.zeros_like(outputs['fixed_reference'])
    for mode in ['map','analytic_transport','shared_seed','fresh_noise','bures_transport']:
        outputs[mode]=heads['transport'](*args,mode=mode,identity_noise=noise,fresh_identity_noises=fresh)[0]
    return outputs,matrix,prior,context,z


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default='results/h3_core/cache_smoke');p.add_argument('--out',default='results/h3_core/adapter_smoke')
    p.add_argument('--steps',type=int,default=200);p.add_argument('--basis-steps',type=int,default=100);p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43]);p.add_argument('--device',default='cuda')
    p.add_argument('--vae');p.add_argument('--comfy',default='external/ComfyUI_H3')
    p.add_argument('--basis-mode',choices=['learned','fixed_pca','random'],default='learned')
    p.add_argument('--eval-only');p.add_argument('--source-root');args=p.parse_args()
    restored=torch.load(args.eval_only,map_location='cpu',weights_only=False) if args.eval_only else None
    actual_steps=args.steps;actual_basis_steps=args.basis_steps
    if restored is not None:
        args.seeds=[restored['seed']];actual_steps=restored['config']['steps'];actual_basis_steps=restored['config']['basis_steps']
        args.steps=0;args.basis_steps=0
    torch.set_num_threads(4);device=torch.device(args.device);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    files=sorted(Path(args.cache).glob('*.pt'));items=[torch.load(p,map_location='cpu',weights_only=False) for p in files]
    cache_manifest=json.loads((Path(args.cache)/'manifest.json').read_text())
    probe_epsilon=cache_manifest['config']['epsilon']
    if args.source_root:
        for item in items:
            source=Path(args.source_root)/item['source']['split']/Path(item['source']['path']).name
            if hashlib.sha256(source.read_bytes()).hexdigest()!=item['source']['source_hash']:raise ValueError('Relocated source hash differs from cached source')
            item['source']['path']=str(source)
    splits={s:[v for v in items if v['source']['split']==s] for s in ['train','val','test']}
    if any(not x for x in splits.values()):raise ValueError('Need disjoint train, val and test source caches')
    manifest={'config':vars(args),'split_sources':{s:[v['source']['source_hash'] for v in vals] for s,vals in splits.items()},
              'scope':'Actual frozen H3 VAE teacher; H3 DiT is NOT trained/evaluated here. Small source counts are pipeline evidence only.',
              'cache_files':[str(p) for p in files]}
    for a,b in [('train','val'),('train','test')]:assert not set(manifest['split_sources'][a])&set(manifest['split_sources'][b])
    vae=None
    if args.vae:
        sys.path.insert(0,str(Path(args.comfy).resolve()));vae,info=load_frozen_vae(args.vae,device=str(device));manifest['vae']=info
    all_results={}
    for seed in args.seeds:
        torch.manual_seed(seed);random.seed(seed)
        atoms=splits['train'][0]['inputs']['grams'].shape[-1]
        chart=CanonicalChart(atoms=atoms,temporal=splits['train'][0]['inputs']['anchor'].shape[2],basis_mode=args.basis_mode).to(device)
        if 'p0' in splits['train'][0]['teacher']:chart.p0.copy_(splits['train'][0]['teacher']['p0'].to(device))
        if restored is not None:chart.load_state_dict(restored['chart'])
        optimizer=torch.optim.Adam(chart.parameters(),lr=.001)
        for step in range(args.basis_steps):
            item=random.choice(splits['train']);z=get_inputs(item,device);y,x=z['roi']
            matrix,prior,context=chart(z['anchor'][:,:,:,y:y+4,x:x+4])
            predicted=(matrix@prior[...,None])[...,0]
            target=item['supervision']['target_p0'].to(device)
            projected=(matrix@(matrix.transpose(1,2)@target[...,None]))[...,0]
            # Basis coverage and prior prediction are separate objectives.
            # This still needs a fixed-PCA/random-basis held-out comparison;
            # QR alone proves neither usefulness nor a nonlinear advantage.
            loss=F.mse_loss(predicted,target)+F.mse_loss(projected,target)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        chart.eval().requires_grad_(False)
        heads={'transport':SpectralDetailTransport().to(device),**{kind:PosteriorDetailHead(kind).to(device) for kind in ['stateless','gru','single_packet']}}
        if restored is not None:
            for kind,head in heads.items():head.load_state_dict(restored['heads'][kind])
        optimizers={kind:torch.optim.Adam(head.parameters(),lr=.001) for kind,head in heads.items()}
        parameters={kind:sum(p.numel() for p in head.parameters()) for kind,head in heads.items()}
        print(json.dumps({'event':'start','seed':seed,'parameters_extra':parameters,'basis_parameters':sum(p.numel() for p in chart.parameters())}),flush=True)
        for step in range(1,args.steps+1):
            item=random.choice(splits['train']);target=item['supervision']['target_p0'].to(device)
            z=get_inputs(item,device);y,x=z['roi']
            with torch.no_grad():matrix,prior,context=chart(z['anchor'][:,:,:,y:y+4,x:x+4])
            noise=torch.randn(1,8,device=device)
            head_args=(matrix,prior,context,z['grams'][None],z['rhs'][None],z['packet_counts'][None])
            losses={}
            for kind,head in heads.items():
                prediction=head(*head_args,identity_noise=noise)
                if kind=='transport':prediction=prediction[0]
                reconstructed=torch.einsum('bkr,btr->btk',matrix,prediction)
                loss=F.mse_loss(reconstructed,target[:,None].expand_as(reconstructed))
                optimizers[kind].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizers[kind].step();losses[kind]=float(loss.detach())
            if step%50==0 or step==args.steps:print(json.dumps({'event':'train','seed':seed,'step':step,'loss':losses}),flush=True)
        result={'seed':seed,'basis_steps':actual_basis_steps,'head_steps':actual_steps,'parameters_extra':parameters,'evaluation':{}}
        with torch.no_grad():
            for split in ['val','test']:
                for number,item in enumerate(splits[split]):
                    predictions,matrix,prior,context,z=solve(chart,heads,item,seed+50000,device)
                    target=item['supervision']['target_p0'].to(device)
                    metrics={};rgb_outputs={}
                    basis_coverage=float((matrix.transpose(1,2)@target[...,None]).square().sum()/target.square().sum().clamp_min(1e-12))
                    for kind,prediction in predictions.items():
                        represented=torch.einsum('bkr,btr->btk',matrix,prediction)
                        errors=(represented-target[:,None]).square().mean(-1)[0]
                        metrics[kind]={'coefficient_mse_first':float(errors[0]),'coefficient_mse_final':float(errors[-1]),'coefficient_mse_per_t':errors.tolist(),
                                       'conditional_basis_target_energy_fraction':basis_coverage,
                                       'canonical_code_drift_occluded':float((prediction[:,30]-prediction[:,24]).square().mean()),
                                       'max_code_norm':float(prediction.norm(dim=-1).max()),
                                       'max_code_norm_over_fd_probe':float(prediction.norm(dim=-1).max())/probe_epsilon}
                        if vae is not None:
                            gt=item['supervision']['canonical_hr'].to(device).permute(2,0,1)[None].float()/255
                            nominal_roi,teacher_support=masks(item,device)
                            sequence=NaturalCameraSequence(item['source']['path'],size=gt.shape[-1],frames=39,protocol=item['source']['protocol'],seed=item['source'].get('sequence_seed',17100))
                            audits=[]
                            for t in [0,17,38]:
                                guide=chart.embed(z['anchor'],matrix,prediction[:,t],z['roi'])
                                rgb=vae.decode(guide.half())[:,:,item['source'].get('output_frame',0)].to(device=device,dtype=torch.float32)
                                pixel_error=float((rgb-gt).square().mean())
                                linear=item['teacher']['base_rgb'].to(device)+torch.einsum('k,kchw->chw',represented[0,t],item['teacher']['decoder_responses'].to(device))[None]
                                packet=sequence.render(t)[0]
                                kwargs=dict(canonical_rectangle=item['source']['canonical_rectangle'],view_rectangle=packet.world_rectangle,view_size=gt.shape[-2:],lr_size=packet.rgb.shape[:2])
                                projected=camera_footprint(rgb,**kwargs)
                                observed=torch.from_numpy(packet.rgb.copy()).to(device).permute(2,0,1)[None].float()/255
                                valid=torch.from_numpy(packet.visible_mask.copy()).to(device)>0
                                image_coverage=camera_footprint(torch.ones_like(rgb[:,:1]),**kwargs)[0,0]
                                valid=valid&(image_coverage>.999)&(image_coverage<1.001)
                                lr_error=float((projected-observed)[:,:,valid].square().mean())
                                audits.append({'frame':t,'actual_decoder_psnr_canonical_db':-10*np.log10(max(pixel_error,1e-12)),
                                               **rgb_metrics(rgb,gt,nominal_roi,teacher_support),
                                               'actual_lr_reprojection_rmse':float(np.sqrt(lr_error)),
                                               'latent_perturbation_norm':float(prediction[:,t].norm()),
                                               'latent_norm_over_fd_probe':float(prediction[:,t].norm())/probe_epsilon,
                                               'decoder_linearization_rmse':float((rgb-linear).square().mean().sqrt())})
                                if t==38:rgb_outputs[kind]=rgb.cpu()
                            metrics[kind]['actual_decoder_audits']=audits
                    key=f"{split}/{Path(item['source']['path']).stem}";result['evaluation'][key]=metrics
                    print(json.dumps({'event':'evaluation','seed':seed,'item':key,'final_coefficient_mse':{k:round(v['coefficient_mse_final'],6) for k,v in metrics.items()}}),flush=True)
                    if rgb_outputs:
                        image_arrays=[item['supervision']['canonical_hr'].numpy()]
                        for kind in ['map','shared_seed','gru','transport']:
                            image_arrays.append((rgb_outputs[kind][0].permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8))
                        Image.fromarray(np.concatenate(image_arrays,1)).save(out/f'comparison_seed{seed}_{split}{number}.png')
                        torch.save({'reference_kind':'canonical frame-0 coordinates; warp to current view before guide injection','guides':{k:chart.embed(z['anchor'],matrix,p[:,-1],z['roi']).cpu() for k,p in predictions.items()}},out/f'guides_seed{seed}_{split}{number}.pt')
        checkpoint={'chart':chart.state_dict(),'heads':{k:h.state_dict() for k,h in heads.items()},'seed':seed,'config':vars(args)}
        if restored is None:torch.save(checkpoint,out/f'checkpoint_seed{seed}.pt')
        (out/f'seed{seed}.json').write_text(json.dumps(result,indent=2));all_results[str(seed)]=result
        (out/'all_results.json').write_text(json.dumps(all_results,indent=2));(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'event':'complete','seeds':list(all_results),'actual_h3_decoder_evaluated':vae is not None}),flush=True)


if __name__=='__main__':main()
