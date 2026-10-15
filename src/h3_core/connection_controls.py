"""Matched physical/VAE/current-H3 and exact finite-prior controls.

Exact Bayes weights apply only to fixed far-only natural prior atoms. Current
conditional proposals already used the cue, so their residual reweighting is
explicitly a heuristic score, not a calibrated Bayesian posterior.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from connection_data import SeparableCamera,VIEWS,SUBSETS
from h3_interface import load_frozen_vae
from observation_quotient import ObservationQuotient


def two_atom_ot(old,new,cost):
    low=max(0.,float(old[0]+new[0]-1));high=min(float(old[0]),float(new[0]));slope=float(cost[0,0]-cost[0,1]-cost[1,0]+cost[1,1]);p=high if slope<=0 else low
    return torch.tensor([[p,float(old[0])-p],[float(new[0])-p,1-float(old[0]+new[0])+p]],dtype=torch.float64).clamp_min(0)


def transport_source(source,old,new,joint):
    mode=0 if source<float(old[0]) else 1;mass=float(old[mode]);local=(source-(0 if mode==0 else float(old[0])))/max(mass,1e-30);threshold=float(joint[mode,0])/max(mass,1e-30)
    next_mode=0 if local<threshold else 1
    local=local/max(threshold,1e-30) if next_mode==0 else (local-threshold)/max(1-threshold,1e-30)
    return min(1-1e-12,max(0.,(0 if next_mode==0 else float(new[0]))+local*float(new[next_mode])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--comfy',required=True);p.add_argument('--vae',required=True);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(Path(args.comfy).resolve()))
    root=Path(args.root);manifest=json.loads((root/'manifest.json').read_text());vae,_=load_frozen_vae(args.vae);q=ObservationQuotient();cams={k:SeparableCamera(v,q) for k,v in VIEWS.items()};result={'scope':'Fixeddevelopment controls, exact finiteprior onlyconditionalonfixedfar-onlyatoms/specifedlikelihood; currentcue-generatedproposalscores areheuristic, notBayesposterior','sample_budget':'Same56H3samples, noextraDiTqueries; actualD controlsadditionalreadouts countedseparately','sources':{}}
    with torch.no_grad():
        for source in manifest['sources']:
            folder=root/source['id'];item=torch.load(folder/'learning_inputs.pt',map_location='cpu',weights_only=False);initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);ensemble=torch.load(folder/'ensemble_control.pt',map_location='cpu',weights_only=False);target=torch.load(folder/'supervision.pt',map_location='cpu',weights_only=False)['target'].cuda().permute(2,0,1)[None].float()/255
            z0=item['z0'].cuda();basis=item['basis'].cuda();reference=item['rgb_reference'].cuda();obs={k:v.cuda() for k,v in item['observations'].items()};private=item['private_mask'].cuda();globalquery=item['global_query_mask'].cuda()
            def decoded(delta):
                shared=torch.einsum('ck,bcthw->bkthw',basis,delta);z=z0+torch.einsum('ck,bkthw->bcthw',basis,shared);return q.bounded(vae.decode(z.half())[:,:,21].cuda().float(),obs['far'])
            def metrics(rgb,keys):
                error=(rgb-target).square();delta=q.null(rgb-reference);lr=torch.stack([(cams[k].observe(rgb)-obs[k]).square().mean() for k in keys]).mean() if keys else torch.tensor(0.,device='cuda')
                return {'psnr_full_db':float(-10*torch.log10(error.mean().clamp_min(1e-12))),'psnr_global_query_db':float(-10*torch.log10(error[:,:,globalquery].mean().clamp_min(1e-12))),'observed_prefix_lr_rmse':float(lr.sqrt()),'private_query_revision_rms':float(delta[:,:,private].square().mean().sqrt()),'highfrequency_rms':float(q.null(rgb).square().mean().sqrt())}
            def weights(rgb,keys):
                if not keys:return torch.ones(2,dtype=torch.float64,device='cuda')/2
                values=torch.stack([sum((cams[k].observe(candidate)-obs[k]).double().square().sum() for k in keys) for candidate in rgb]);return torch.softmax(-.5*(values-values.min())/.005**2,0)
            rows={'initial':metrics(reference,['A','B']),'subsets':{},'conditional_OT':{},'private_protection':'Every latentcontrol uses sameU/X0complement as neuralheads; classicalpixelMAP is separatelystrongconstraintcontrol.'};guide0=load_file(str(folder/'response_empty_seed101/guide.safetensors'))['guide'].cuda().float();conditional={};cw={}
            for subset,keys in SUBSETS.items():
                candidates=[v.cuda() for v in ensemble['rgb_candidates'][subset]];conditional[subset]=candidates;cw[subset]=weights(candidates,keys);mean=sum(candidates)/2;scored=sum(w*x for w,x in zip(cw[subset],candidates));chosen=candidates[int(cw[subset].argmax())];fixed=candidates[0 if .37<float(cw[subset][0]) else 1]
                guide=load_file(str(folder/f'response_{subset}_seed101/guide.safetensors'))['guide'].cuda().float();vaeonly=decoded(guide-guide0);response=item['responses'][subset].cuda();guide_response=guide-guide0
                full_response=q.bounded(vae.decode((z0+response.mean(0,keepdim=True)).half())[:,:,21].cuda().float(),obs['far'])
                full_guide=q.bounded(vae.decode((z0+guide_response).half())[:,:,21].cuda().float(),obs['far'])
                rows['subsets'][subset]={'classical_joint_LR':metrics(initial['classical_rgb'][subset].cuda(),keys),'VAEguide_response_same_U_private':metrics(vaeonly,keys),'H3conditional_uniform_mean':metrics(mean,keys),'H3conditional_residual_scored_mean':metrics(scored,keys),'H3conditional_minLR_candidate':metrics(chosen,keys),'H3conditional_fixedsource':metrics(fixed,keys),'structural_no_private_carrier_full24_H3response':metrics(full_response,keys),'structural_no_private_carrier_full24_VAEguide_response':metrics(full_guide,keys),'conditional_score_weights_not_calibrated_posterior':cw[subset].tolist(),'H3_response_minus_guide_response_relative':float((response-guide_response).norm()/response.norm().clamp_min(1e-20))}
            # Exact2x2 minimum-cost coupling on current conditional-candidate
            # marginals. Their data-dependent weights remain heuristic scores.
            for path in [('A','B'),('B','A')]:
                old_subset='empty';source_u=.37
                for step,cue in enumerate(path):
                    subset=cue if step==0 else 'AB';cost=torch.tensor([[float((q.null(x-y)[:,:,private]).square().mean()) for y in conditional[subset]] for x in conditional[old_subset]],dtype=torch.float64);joint=two_atom_ot(cw[old_subset].cpu(),cw[subset].cpu(),cost);source_u=transport_source(source_u,cw[old_subset].cpu(),cw[subset].cpu(),joint);old_subset=subset
                atom=0 if source_u<float(cw['AB'][0]) else 1;rows['conditional_OT'][''.join(path)]={'source_u':source_u,'final_atom':atom,'metrics':metrics(conditional['AB'][atom],['A','B'])}
            if source['split']=='val':
                prior=[reference,decoded(initial['second_natural_initial']['z0'].cuda()-z0)];priorrows={};posterior={}
                for subset,keys in SUBSETS.items():
                    w=weights(prior,keys);posterior[subset]=w;mean=sum(weight*atom for weight,atom in zip(w,prior));fixed=prior[0 if .37<float(w[0]) else 1];priorrows[subset]={'exact_weights':w.tolist(),'effective_atoms':float(1/w.square().sum()),'exact_mean':metrics(mean,keys),'fixedsource':metrics(fixed,keys)}
                cost=torch.tensor([[float((x-y).square().mean()) for y in prior] for x in prior],dtype=torch.float64);otrows={}
                for path in [('A','B'),('B','A')]:
                    previous='empty';u=.37
                    for step,cue in enumerate(path):
                        subset=cue if step==0 else 'AB';joint=two_atom_ot(posterior[previous].cpu(),posterior[subset].cpu(),cost);u=transport_source(u,posterior[previous].cpu(),posterior[subset].cpu(),joint);previous=subset
                    atom=0 if u<float(posterior['AB'][0]) else 1;otrows[''.join(path)]={'final_atom':atom,'metrics':metrics(prior[atom],['A','B'])}
                rows['exact_fixed_natural_far_prior']={'scope':'Bothatoms genuinefar-onlyH3initialseed101/202, matchedU/private. ExactspecifiedfiniteBayes, not claimcalibratedpopulationposterior.','subsets':priorrows,'OT':otrows}
            result['sources'][source['id']]=rows;(root/'strong_controls.json').write_text(json.dumps(result,indent=2));print(json.dumps({'event':'connection_controls','source':source['id'],'classical_global':rows['subsets']['AB']['classical_joint_LR']['psnr_global_query_db'],'VAEguide_global':rows['subsets']['AB']['VAEguide_response_same_U_private']['psnr_global_query_db'],'H3global':rows['subsets']['AB']['H3conditional_uniform_mean']['psnr_global_query_db']}),flush=True)
    (root/'strong_controls.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
