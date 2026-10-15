"""Fixed LR-only operator-family guides and actual H3 response cache.

This is a conditional-function compression pilot after far/A/B have arrived,
not a claim to have processed each earlier prefix with future-free H3 calls.
"""
import argparse, json, sys, time
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from connection_data import SeparableCamera,VIEWS,classical_map
from h3_interface import load_frozen_vae
from observation_quotient import ObservationQuotient

COORDS=[(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.),(0.,0.)]
SUPPORT=[0,1,2]


def cameras_for(g,q):
    x,y=g;old=VIEWS['A'];rect=(old[0]+4*x,old[1]+4*y,old[2]+4*x,old[3]+4*y)
    return {'A':SeparableCamera(rect,q),'B':SeparableCamera(VIEWS['B'],q)}


def joint_weights(rgb,obs,q):
    # Fixed far-only natural atoms x operator hypotheses: exact finite-prior
    # likelihood. This is NOT Bayes weighting of cue-conditioned C_g outputs.
    early=[];late=[]
    for u in range(2):
        for g in COORDS:
            cams=cameras_for(g,q)
            a=(cams['A'].observe(rgb[u:u+1])-obs['A']).double().square().sum()
            b=(cams['B'].observe(rgb[u:u+1])-obs['B']).double().square().sum()
            early.append(-.5*a/.005**2);late.append(-.5*(a+b)/.005**2)
    loglate=torch.stack(late).reshape(2,5)
    return torch.softmax(torch.stack(early),0).reshape(2,5),torch.softmax(loglate.flatten(),0).reshape(2,5),torch.softmax(loglate,1)


def robust_inverse(base,obs,q,cams):
    ridge=.01;weights={k:torch.ones_like(obs[k]) for k in ['A','B']};value=torch.zeros_like(base)
    for _ in range(3):
        def normal(x):
            nx=q.null(x)
            return q.null(sum(cams[k].adjoint(weights[k]*cams[k].observe(nx)) for k in ['A','B']))+ridge*x
        rhs=q.null(sum(cams[k].adjoint(weights[k]*(obs[k]-cams[k].observe(base))) for k in ['A','B']))
        value=torch.zeros_like(base);r=rhs.clone();p=r.clone();rr=r.square().sum()
        for _ in range(48):
            ap=normal(p);alpha=rr/(p*ap).sum().clamp_min(1e-30);value+=alpha*p;r-=alpha*ap;new=r.square().sum();p=r+new/rr.clamp_min(1e-30)*p;rr=new
        for k in weights:
            residual=(cams[k].observe(base+q.null(value))-obs[k]).abs()
            weights[k]=(.01/residual.clamp_min(.01)).clamp_min(.05)
    rgb=q.bounded(base+q.null(value),obs['far'])
    errors=[(cams[k].observe(rgb)-obs[k]).abs() for k in ['A','B']]
    def robust_penalty(e):
        middle=.02*e-.0001
        return torch.where(e<.01,e.square(),torch.where(e<.2,middle,.05*e.square()+.0019))
    score=ridge*(rgb-base).square().sum()+sum(robust_penalty(e).sum() for e in errors)
    return rgb,float(score)


def prepare(a,root,old,vae,q):
    began=time.monotonic();manifest=json.loads((old/'manifest.json').read_text());jobs=[];enc=dec=0
    with torch.no_grad():
        for source in manifest['sources']:
            sid=source['id'];where=root/sid;where.mkdir(exist_ok=True,parents=True);previous=old/sid
            data=torch.load(previous/'inputs.pt',map_location='cpu',weights_only=False);initial=torch.load(previous/'initial.pt',map_location='cpu',weights_only=False);ensemble=torch.load(previous/'ensemble_control.pt',map_location='cpu',weights_only=False)
            obs={k:v.cuda() for k,v in data['observations'].items()};basez=ensemble['latents']['empty'].cuda().float();basergb=[]
            for u in range(2):basergb.append(q.bounded(vae.decode(basez[u:u+1].half())[:,:,21].cuda().float(),obs['far']));dec+=1
            basergb=torch.cat(basergb);early,late,conditional=joint_weights(basergb,obs,q);gbar=late.sum(0)@torch.tensor(COORDS,dtype=late.dtype,device='cuda')
            reference=initial['rgb_reference'].cuda();guide0=load_file(str(previous/'response_empty_seed101/guide.safetensors'))['guide'];guides=[];maps=[]
            for j,g in enumerate(COORDS):
                cams=cameras_for(g,q);rgb,_=classical_map(reference,obs,cams,['A','B'],q);maps.append(rgb.cpu())
                if j<4:
                    guide=vae.encode((2*rgb-1)[:,:,None].repeat(1,1,22,1,1).half().clone()).float().cpu().contiguous();enc+=1
                    for seed in [101,202]:
                        folder=where/f'offset{j}_seed{seed}';folder.mkdir(exist_ok=True);save_file({'guide':guide},str(folder/'guide.safetensors'));jobs.append(dict(id=f'{sid}/offset{j}/seed{seed}',source=sid,directory=str(folder),seed=seed,evidence_subset=['far','A','B'],operator_A_HR_shift=[4*g[0],4*g[1]]))
                else:guide=load_file(str(previous/'response_AB_seed101/guide.safetensors'))['guide']
                guides.append(guide)
            # Full joint robust geometry scan is a separate strong classical
            # control; all source/carrier hypotheses and all five g are tried.
            robust=[];robust_audits=[]
            for u in range(2):
                hypotheses=[robust_inverse(basergb[u:u+1],obs,q,cameras_for(g,q)) for g in COORDS];idx=min(range(5),key=lambda j:hypotheses[j][1]);robust.append(hypotheses[idx][0].cpu());robust_audits.append(dict(selected=idx,objective=[v[1] for v in hypotheses],IRLS=3,CG=48,ridge=.01,huber=.01))
            torch.save(dict(source=source,observations=data['observations'],packet_ids=data['packet_ids'],basez=basez.cpu(),basergb=basergb.cpu(),guides=torch.cat(guides),guide0=guide0,classical_maps=torch.cat(maps),joint_early_weights=early.cpu(),joint_late_weights=late.cpu(),conditional_late_weights=conditional.cpu(),inferred_query=gbar.cpu().float(),robust_inverse=torch.cat(robust),robust_audit=robust_audits,old_folder=str(previous)),where/'prepared.pt')
            print(json.dumps(dict(event='operator_guides_prepared',source=sid,gbar=gbar.tolist(),early=early.tolist(),late=late.tolist())),flush=True)
    value=dict(scope='Fixed existing4train2val development; post-arrival conditional family, not fullstreaming success',sources=manifest['sources'],coordinates=COORDS,HR_shift_multiplier=4,support_indices=SUPPORT,held_composition_index=3,held_interior_index=4,actual_new_H3_samples=48,full50_DiT_forwards=2352,source_carrier='Actual existing C_empty seed101/202 full24; not artificial modes',nativeE_calls=enc,nativeD_calls=dec,preparation_seconds=time.monotonic()-began,geometry_inference='Exact finite far-only atom/operator Gaussian likelihood sigma.005; shared across all methods; no GT geometry input',robust_control='5geometry x2sourcecarrier,3IRLSx48CG per image; noHRselection')
    (root/'manifest.json').write_text(json.dumps(value,indent=2));(root/'jobs.json').write_text(json.dumps(jobs,indent=2))


def assemble(a,root,old,vae,q):
    manifest=json.loads((root/'manifest.json').read_text());calls=0
    with torch.no_grad():
        for source in manifest['sources']:
            where=root/source['id'];data=torch.load(where/'prepared.pt',map_location='cpu',weights_only=False);previous=Path(data['old_folder']);base=data['basez'];observations={k:v.cuda() for k,v in data['observations'].items()};family=[];targets=[];vaergbs=[]
            for j,g in enumerate(COORDS):
                zs=[];rgbs=[];vrgbs=[]
                for u,seed in enumerate([101,202]):
                    folder=where/f'offset{j}_seed{seed}' if j<4 else previous/f'response_AB_seed{seed}';record=json.loads((folder/'sample_result.json').read_text());assert record['status']=='complete' and len(record['forward_events'])==49
                    z=load_file(str(folder/'sampled_av_latents.safetensors'))['video'].float();zs.append(z)
                    rgbs.append(q.bounded(vae.decode(z.cuda().half())[:,:,21].cuda().float(),observations['far']).cpu());calls+=1
                    zv=base[u:u+1]+data['guides'][j:j+1]-data['guide0'];vrgbs.append(q.bounded(vae.decode(zv.cuda().half())[:,:,21].cuda().float(),observations['far']).cpu());calls+=1
                family.append(torch.cat(zs));targets.append(torch.cat(rgbs));vaergbs.append(torch.cat(vrgbs))
            data.update(response=torch.stack(family,1)-base[:,None],teacher_rgb=torch.stack(targets,1),VAE_response=(data['guides']-data['guide0'])[None].expand(2,-1,-1,-1,-1,-1).clone(),VAE_rgb=torch.stack(vaergbs,1))
            y,x=torch.meshgrid(torch.arange(256),torch.arange(256),indexing='ij');data['private_mask']=((x<32)|(x>=224))&((y<32)|(y>=224));data['target']=torch.load(previous/'supervision.pt',map_location='cpu',weights_only=False)['target'].permute(2,0,1)[None].float()/255
            assert data['response'].shape==(2,5,24,7,16,16);torch.save(data,where/'family.pt');print(json.dumps(dict(event='operator_family_ready',source=source['id'],response_RMS=float(data['response'].square().mean().sqrt()))),flush=True)
    (root/'assembly.json').write_text(json.dumps(dict(nativeD_calls=calls,status='complete',sources=len(manifest['sources'])),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','assemble']);p.add_argument('--root',required=True);p.add_argument('--old',required=True);p.add_argument('--comfy',required=True);p.add_argument('--vae',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;sys.path.insert(0,str(Path(a.comfy).resolve()));root=Path(a.root);root.mkdir(exist_ok=True,parents=True);vae,_=load_frozen_vae(a.vae);q=ObservationQuotient();(prepare if a.phase=='prepare' else assemble)(a,root,Path(a.old),vae,q)


if __name__=='__main__':main()
