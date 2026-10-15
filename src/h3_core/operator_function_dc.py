"""Separate DC-version jobs, physical checks, and matched carrier rebasing.

No near guide bytes change. Sampling projection is the actual declared
Gaussian/AA far camera, not the real-video bicubic-only baseline operator.
"""
import argparse,json,shutil,sys,time
from pathlib import Path
import torch
from safetensors.torch import load_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from operator_function_data import COORDS,cameras_for,joint_weights,robust_inverse
from observation_quotient import ObservationQuotient
from h3_interface import load_frozen_vae
from camera_operator import camera_footprint


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['jobs','audit','rebase']);p.add_argument('--root',required=True);p.add_argument('--raw',default='results/h3_core/operator_function_v1');p.add_argument('--comfy');p.add_argument('--vae');a=p.parse_args();root=Path(a.root);raw=Path(a.raw);root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.phase=='audit':
        q=ObservationQuotient(device='cpu');generator=torch.Generator().manual_seed(725);x=.2+.6*torch.rand((2,3,256,256),generator=generator,dtype=torch.float64);independent=camera_footprint(x,(0,0,256,256),(0,0,256,256));linear=q.observe(x);error=float((linear-independent).abs().max());fixed=float((q.bounded(x,linear,steps=25)-x).abs().max());assert error<1e-12 and fixed<1e-12
        result=dict(status='passed',actual_operator='Exact shared reflectGaussian.6+AA bicubic4x separable matrix',independent_camera_forward_maxabs=error,actual_P25_feasible_RGB_fixedpoint_maxabs=fixed,pixels='Float RGB; no newPIL/PNGquantization',independent_far_packets=1,repeated_native_context_frames=22,HR_input=False,scope='CPU physical/operator audit; not H3 restoration efficacy');(root/'physical_operator_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    manifest=json.loads((raw/'manifest.json').read_text())
    if a.phase=='jobs':
        jobs=[];guard=[]
        for source in manifest['sources']:
            sid=source['id'];where=root/sid;where.mkdir(exist_ok=True);prepared=torch.load(raw/sid/'prepared.pt',map_location='cpu',weights_only=False);original=Path(prepared['old_folder']);shutil.copy2(original/'supervision.pt',where/'supervision.pt')
            for role in ['empty','AB','offset0','offset1','offset2','offset3']:
                for seed in [101,202]:
                    name=f'response_{role}_seed{seed}' if role in ['empty','AB'] else f'{role}_seed{seed}';oldwhere=original/name if role in ['empty','AB'] else raw/sid/name;dest=where/name;dest.mkdir(exist_ok=True);shutil.copy2(oldwhere/'guide.safetensors',dest/'guide.safetensors');job=dict(id=f'{sid}/{role}/seed{seed}',source=sid,directory=str(dest),seed=seed,role=role,evidence_subset=['far'] if role=='empty' else ['far','A','B'],far_inputs=str(original/'inputs.pt'),guide_copied_byte_identical_from=str(oldwhere/'guide.safetensors'))
                    jobs.append(job)
                    if sid=='train_100075' and role=='offset0' and seed==101:
                        gd=root/'identity_guard';gd.mkdir(exist_ok=True);shutil.copy2(oldwhere/'guide.safetensors',gd/'guide.safetensors');guard.append(dict(**{**job,'id':'TRAIN_identity_guard','directory':str(gd),'mode':'identity','raw_reference':str(oldwhere/'sampled_av_latents.safetensors')}))
        assert len(jobs)==72 and len(guard)==1
        manifest.update(scope='Independentfar-samplingDCversion, all12carriers+12center+48offsetregenerated; samecamera/LR/near-guidebytes/headbudget asraw',actual_new_H3_samples=72,full50_DiT_forwards=3528,source_carrier='Actual newfar-only DC C_empty seed101/202 full24',nativeE_calls=0,nativeD_calls=0,sampling_projection='Own exactGaussian.6 AA far operator P25; unit E(P(D))-E(D) at28/45/48',raw_version=str(raw),no_new_identity_claim=True)
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2));(root/'jobs.json').write_text(json.dumps(jobs,indent=2));(root/'guard_jobs.json').write_text(json.dumps(guard,indent=2));print(json.dumps(dict(jobs=72,guard_jobs=1)),flush=True);return
    sys.path.insert(0,str(Path(a.comfy).resolve()));vae,_=load_frozen_vae(a.vae);q=ObservationQuotient();calls=0;began=time.monotonic()
    with torch.no_grad():
        for source in manifest['sources']:
            sid=source['id'];where=root/sid;data=torch.load(raw/sid/'prepared.pt',map_location='cpu',weights_only=False);obs={k:v.cuda() for k,v in data['observations'].items()};z=torch.cat([load_file(str(where/f'response_empty_seed{seed}/sampled_av_latents.safetensors'))['video'].float() for seed in [101,202]]).cuda();rgb=[]
            for u in range(2):rgb.append(q.bounded(vae.decode(z[u:u+1].half())[:,:,21].cuda().float(),obs['far']));calls+=1
            rgb=torch.cat(rgb);early,late,conditional=joint_weights(rgb,obs,q);gbar=late.sum(0)@torch.tensor(COORDS,device='cuda',dtype=late.dtype);robust=[];audits=[]
            for u in range(2):
                hs=[robust_inverse(rgb[u:u+1],obs,q,cameras_for(g,q)) for g in COORDS];idx=min(range(5),key=lambda j:hs[j][1]);robust.append(hs[idx][0].cpu());audits.append(dict(selected=idx,objective=[v[1] for v in hs],IRLS=3,CG=48,ridge=.01,huber=.01))
            data.update(basez=z.cpu(),basergb=rgb.cpu(),joint_early_weights=early.cpu(),joint_late_weights=late.cpu(),conditional_late_weights=conditional.cpu(),inferred_query=gbar.cpu().float(),robust_inverse=torch.cat(robust),robust_audit=audits,old_folder=str(where),sampler_version='matchedfarinnerDC72');torch.save(data,where/'prepared.pt');print(json.dumps(dict(event='DCcarrier_rebased',source=sid,inferred_query=gbar.tolist())),flush=True)
    (root/'rebase.json').write_text(json.dumps(dict(status='complete',nativeD_calls=calls,seconds=time.monotonic()-began),indent=2))


if __name__=='__main__':main()
