"""Fixed last672 evidence-dependence of every head; no optimization/selection."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_dense import build_dense,DENSE_KINDS
from revision_utils import batch,Readout,FP
from native_vae import load_vae,differentiable_decoder_math


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--models',required=True);p.add_argument('--baseline',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());sources=[x for x in manifest['items'] if x['split']=='val'];assert len(sources)==8;vae=load_vae(a.vae,a.comfy);read=Readout(vae);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.time();base_rows=list(csv.DictReader(Path(a.baseline).open()));rows=[];gradients=[];models={}
    for kind in DENSE_KINDS:
        model=build_dense(kind).cuda().eval();cp=torch.load(Path(a.models)/f'{kind}_seed17'/'last.pt',map_location='cpu',weights_only=False);assert cp['step']==672;model.load_state_dict(cp['model']);models[kind]=model
    with torch.no_grad():
        for i,source in enumerate(sources):
            item=torch.load(cache/(source['id']+'.pt'),map_location='cpu',weights_only=False);other=torch.load(cache/(sources[(i+1)%8]['id']+'.pt'),map_location='cpu',weights_only=False);inputs,target=batch([item],'cuda');far=item['classical_inputs']['query_LR'].cuda().float()
            for kind,model in models.items():
                baseline=model.rollout(inputs)[0]
                for variant in ['no_measured_delta','shuffled_history_delta_and_proxy']:
                    vi={k:v.clone() for k,v in inputs.items()}
                    if variant=='no_measured_delta':vi['packet_delta'].zero_()
                    else:
                        for key in ['packet_delta','packet_proxy']:vi[key]=other['inputs'][key][None].cuda().float()
                    z=model.rollout(vi)[0]
                    if kind=='anchor_only':assert torch.equal(z,baseline)
                    for qi,step in enumerate(inputs['query_steps'][0].tolist()):
                        union=torch.zeros((1,1,256,256),device='cuda',dtype=torch.bool)
                        for j,arrival in enumerate(inputs['packet_steps'][0].tolist()):
                            if arrival<=step:top,left=FP[int(inputs['packet_fp'][0,j])];union[...,top:top+128,left:left+128]=True
                        if kind=='anchor_only':values={family:None for family in ['raw','DC']}
                        else:rgb=read.rgb(inputs,z[:,qi],qi);values=dict(raw=rgb.clamp(0,1),DC=read.far.apply(rgb,far[qi:qi+1]))
                        for family,rgb in values.items():
                            b=next(x for x in base_rows if x['source']==source['id'] and x['method']==kind and x['family']==family and int(x['query_step'])==step);psnr=float(b['union_psnr']) if rgb is None else float(-10*torch.log10((rgb-target['rgb'][:,qi]).square()[union.expand_as(rgb)].mean()));rows.append(dict(source=source['id'],method=kind,variant=variant,family=family,query_step=step,condition=b['condition'],baseline_union_PSNR=float(b['union_psnr']),intervention_union_PSNR=psnr,baseline_minus_intervention_dB=float(b['union_psnr'])-psnr,predicted_latent_change_RMSE=float((z[:,qi]-baseline[:,qi]).square().mean().sqrt())))
            print(json.dumps(dict(source=source['id'],step=i+1,steps=8,seconds=time.time()-start)),flush=True)
    # The first pretraining same-appearance probe analytically cancels affine
    # H(a)^-1/H(a), yielding rounding-sized no-equiv core gradients. A mixed
    # capture-group query is the meaningful test of D sensitivity to the core.
    train=next(x for x in manifest['items'] if x['split']=='train');item=torch.load(cache/(train['id']+'.pt'),map_location='cpu',weights_only=False);inputs,target=batch([item],'cuda')
    for kind,model in models.items():
        model.zero_grad(set_to_none=True);pred=model.rollout(inputs)[0]
        with differentiable_decoder_math():rgb=read.rgb(inputs,pred[:,4],4);loss=F.l1_loss(rgb,target['rgb'][:,4]);loss.backward()
        g={branch:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(branch) and p.grad is not None) for branch in ['encoder','decoder','core','appearance']};assert all(np.isfinite(v) for v in g.values());assert all(p.grad is None for p in vae.parameters());gradients.append(dict(method=kind,source=train['id'],query_step=11,loss='actualD_RGB_L1_only_noaux_no_update',gradients=g,frozen_VAE_grads=0))
    with (out/'all_head_history_interventions.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    paired=[];rng=np.random.default_rng(20261015);ii=rng.integers(0,8,(10000,8))
    for kind in DENSE_KINDS:
        for variant in ['no_measured_delta','shuffled_history_delta_and_proxy']:
            ss=[np.mean([r['baseline_minus_intervention_dB'] for r in rows if r['method']==kind and r['variant']==variant and r['family']=='DC' and r['source']==s['id']]) for s in sources];v=np.array(ss);ci=np.quantile(v[ii].mean(1),[.025,.975]);paired.append(dict(method=kind,variant=variant,source_mean_drop_dB=float(v.mean()),CI=ci.tolist(),sources_hurt_by_intervention=int((v>0).sum()),per_source=ss))
    (out/'summary.json').write_text(json.dumps(dict(paired=paired,mixed_query_actualD_gradients=gradients,seconds=time.time()-start,actual_D_backwards_no_optimizer=7,optimizer_updates=0,scope='Fixedlast672;8VAL source bootstrap; controller/metadata held fixed. Only evidence payload is changed. Mixed-query gradient probe is oneTRAIN source and nooptimization.'),indent=2))


if __name__=='__main__':main()
