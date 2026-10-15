"""Fixed-checkpoint val-only decoder/measurement mechanism diagnostics."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from model import FactorizedMemory,flatten_time,restore_time
from train import batch,write_live
from native_vae import load_vae,decode_last
from pixel_dc import Projection,project_visible
from cache_pairs import degrade


class SupportedDetail:
    """C=N_far P_near_emb N_far; linear contrast, not claimed novel."""
    def __init__(self,far,near):
        self.pf=far.right@far.matrix;self.pn=near.right@near.matrix
    def null_far(self,x):return x-self.pf@x@self.pf.T
    def near_support(self,x):
        result=torch.zeros_like(x);result[:,:,64:192,64:192]=self.pn@x[:,:,64:192,64:192]@self.pn.T;return result
    def __call__(self,x):return self.null_far(self.near_support(self.null_far(x)))


def psnr(x,y):return float(-10*torch.log10((x-y).square().mean().clamp_min(1e-12)))
def rms(x):return float(x.square().mean().sqrt())


def update_terms(model,old,observed,a,inputs,step):
    o,shape=flatten_time(old);z,_=flatten_time(observed);b,c,t,h,w=shape
    visible=F.interpolate(inputs['visible'][:,step],size=(h,w),mode='area');reveal=F.interpolate(inputs['reveal'][:,step],size=(h,w),mode='area');channels=torch.cat((visible,reveal,torch.full_like(visible,float(inputs['footprint'][step]))),1)
    channels=channels[:,None].expand(-1,t,-1,-1,-1).reshape(b*t,3,h,w);aa=a[:,None,:,None,None].expand(-1,t,-1,h,w).reshape(b*t,12,h,w)
    delta,gate=model.update_head.net(torch.cat((o,z,channels,aa),1)).chunk(2,1);gate=gate.sigmoid();obs=channels[:,:1]*gate*(z-o);correction=channels[:,:1]*gate*2*delta.tanh()
    return restore_time(obs,shape),restore_time(correction,shape),float(gate.mean())


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--observations',required=True);p.add_argument('--runs',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);start=time.time();vae=load_vae(args.vae,args.comfy);far=Projection(256,4);near=Projection(128,2);band=SupportedDetail(far,near)
    source_manifest=json.loads((Path(args.cache)/'manifest.json').read_text());items=[i for i in source_manifest['items'] if i['split']=='val'];assert len(items)==8
    models=[]
    for seed in [17,29,43]:
        c=torch.load(Path(args.runs)/f'factorized_swap_seed{seed}'/'best.pt',map_location='cpu',weights_only=False);m=FactorizedMemory('factorized_swap').cuda().eval();m.load_state_dict(c['model']);models.append((seed,m))
    rows=[];summary=[]
    with torch.no_grad():
        probe=torch.randn(1,3,256,256,device='cuda');nullcheck=rms(far.observe(band(probe)));coarsecheck=rms(band(probe-band.null_far(probe)));assert max(nullcheck,coarsecheck)<2e-6
        for i,prov in enumerate(items):
            item=torch.load(Path(args.cache)/(prov['id']+'.pt'),weights_only=False);inputs,targets=batch([item],'cuda');measurement=torch.load(Path(args.observations)/(prov['id']+'.pt'),weights_only=False)['inputs'];yn=measurement['near_y'][2:3].cuda();yf=measurement['far_y'][4:5].cuda()
            for seed,model in models:
                _,a,states=model.rollout(inputs);fixed_query=a[:,2];gt=targets['rgb'][:,1];target_band=band(gt);values={}
                for step,label in [(1,'pre_near'),(2,'post_near'),(3,'query_no_write'),(4,'post_return')]:
                    rgb=decode_last(vae,model.chart.read(states[step],fixed_query));be=(band(rgb)-target_band).square().mean();values[label]=float(be)
                    rows.append(dict(source=prov['id'],seed=seed,query='fixed_observed_near_appearance',case=label,rgb_psnr=psnr(rgb,gt),near_measurement_rmse=rms(degrade(rgb[:,:,64:192,64:192],2)-yn),far_measurement_rmse=None,detail_band_rmse=float(be.sqrt()),DC_rgb_psnr=None,obs_term_rms=None,correction_term_rms=None,gate_mean=None))
                canonical=model.chart.canonicalize(inputs['observation_latents'][:,4],a[:,4]);obsterm,corrterm,gate=update_terms(model,states[3],canonical,a[:,4],inputs,4);assert float((states[3]+obsterm+corrterm-states[4]).abs().max())<1e-5
                base=decode_last(vae,inputs['observation_latents'][:,4]);nearbase=decode_last(vae,inputs['observation_latents'][:,2]);neutral_gt=targets['rgb'][:,0]
                rows.append(dict(source=prov['id'],seed=seed,query='actual_near_input',case='current_LR_VAE_near',rgb_psnr=psnr(nearbase,gt),near_measurement_rmse=rms(degrade(nearbase[:,:,64:192,64:192],2)-yn),far_measurement_rmse=None,detail_band_rmse=rms(band(nearbase)-target_band),DC_rgb_psnr=None,obs_term_rms=None,correction_term_rms=None,gate_mean=None))
                retained=decode_last(vae,model.chart.read(states[2],a[:,4]));probe_rgb=base+band(retained-base)
                near_observed_canonical=model.chart.canonicalize(inputs['observation_latents'][:,2],a[:,2]);direct_near=decode_last(vae,model.chart.read(near_observed_canonical,a[:,4]))
                cases={'skip_far_write':retained,'observation_injection_only':decode_last(vae,model.chart.read(states[3]+obsterm,a[:,4])),'learned_correction_only':decode_last(vae,model.chart.read(states[3]+corrterm,a[:,4])),'normal_return':decode_last(vae,model.chart.read(states[4],a[:,4])),'current_LR_VAE_return':base,'classical_anchor_plus_supported_memory_probe':probe_rgb,'classical_anchor_plus_direct_near_code_probe':base+band(direct_near-base)}
                for label,rgb in cases.items():
                    projected,_=project_visible(rgb,measurement,4,far,near)
                    rows.append(dict(source=prov['id'],seed=seed,query='fixed_current_return_appearance',case=label,rgb_psnr=psnr(rgb,neutral_gt),near_measurement_rmse=None,far_measurement_rmse=rms(degrade(rgb,4)-yf),detail_band_rmse=rms(band(rgb)-band(neutral_gt)),DC_rgb_psnr=psnr(projected,neutral_gt),obs_term_rms=rms(obsterm),correction_term_rms=rms(corrterm),gate_mean=gate))
                summary.append(dict(source=prov['id'],seed=seed,near_band_MSE_pre=values['pre_near'],near_band_MSE_post=values['post_near'],near_band_MSE_return=values['post_return'],near_provided_positive_band_gain=values['pre_near']>values['post_near'],gain_retained_ratio=((values['pre_near']-values['post_return'])/(values['pre_near']-values['post_near'])) if values['pre_near']>values['post_near'] else None))
            with (out/'decoder_observation_steps.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            (out/'near_return_gain.json').write_text(json.dumps(summary,indent=2));write_live(out,status='fixed_checkpoint_val_mechanism_diagnostics',phase='val_only_decoder_observation_audit',step=i+1,steps=8,source=prov['id'],seconds=time.time()-start);print(json.dumps(dict(source=prov['id'],seconds=time.time()-start)),flush=True)
    config=dict(**vars(args),split='val_only',source_count=8,training_seeds=[17,29,43],HR_only_evaluation=True,checkpoints_frozen=True,new_training=False,band='C=N_far P_near_emb N_far, a linear near-supported/far-null contrast; not an orthogonal projector or a novel algorithm',far_null_test_rms=nullcheck,coarse_annihilation_test_rms=coarsecheck,actual_H3_D='native256x22 float16',chart='strict float32 noTF32',seconds=time.time()-start)
    (out/'complete.json').write_text(json.dumps(config,indent=2));write_live(out,status='val_mechanism_complete',phase='val_only_decoder_observation_audit',step=8,steps=8,seconds=time.time()-start)


if __name__=='__main__':main()
