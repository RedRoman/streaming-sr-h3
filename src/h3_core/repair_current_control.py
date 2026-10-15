"""Replace only the invalid rebased-difference current-packet control.

The other completed models are unchanged. Replay the original constructor,
Python-data and CUDA-noise schedule, use the saved shared prior, and train the
same400steps with genuine per-packet sufficient statistics. Original results
are preserved and marked legacy; this is a fairness bug repair, not tuning.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_quotient import ProjectedPrior,witness
from train_nonlinear import masks,rgb_metrics
from nonlinear_memory import NonlinearMemoryHead
from h3_interface import load_frozen_vae,differentiable_decoder_math
from observation_quotient import ObservationQuotient


def repair(cache,run,vae_path,comfy):
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False);seed=checkpoint['config']['seed'];torch.manual_seed(seed);random.seed(seed)
    vae,_=load_frozen_vae(vae_path);q=ObservationQuotient();items=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted(Path(cache).glob('*.pt'))]
    train=[v for v in items if v['source']['split']=='train'];val=[v for v in items if v['source']['split']=='val']
    prior=ProjectedPrior(items[0]['teacher']['p0']).cuda();heads={k:NonlinearMemoryHead(k,rank=32,hidden=27,initialize=True).cuda() for k in ['transport','stateless','gru','current_only']}
    prior.load_state_dict(checkpoint['prior']);prior.eval().requires_grad_(False);head=heads['current_only'];optimizer=torch.optim.Adam(head.parameters(),lr=1e-3)
    for _ in range(checkpoint['config']['prior_steps']):random.choice(train)
    matrix=torch.eye(32,device='cuda')[None];scale=torch.ones(1,32,device='cuda')
    def move(item):return {k:v.cuda() if torch.is_tensor(v) else v for k,v in item['inputs'].items()}
    def decoded(z,coefficient,t,backward=False):
        latent=prior.embed(z['anchor'],coefficient,z['roi'])
        if backward:
            with differentiable_decoder_math():rgb=vae.decode(latent.half())[:,:,21].cuda()
        else:rgb=vae.decode(latent.half())[:,:,21].cuda()
        return q.readout(rgb,z['coarse'][t:t+1])
    for step in range(checkpoint['config']['steps']):
        item=random.choice(train);z=move(item);t=random.choice([0,8,17,24,30,38]);noise=torch.randn(1,32,device='cuda')
        with torch.no_grad():mu,context=prior(z['anchor'],z['roi'])
        common=(matrix,mu,context,z['packet_grams'][None],z['packet_rhs'][None],z['packet_counts'][None])
        coefficient=head(*common,noise,scale,already_incremental=True)[0][:,t];rgb=decoded(z,coefficient,t,True)
        target=item['supervision']['canonical_hr'].cuda().permute(2,0,1)[None].float()/255;roi,support=masks(item,'cuda');e=(rgb-target).square();physical,_=witness(rgb,item,t)
        loss=100*e[:,:,roi].mean()+25*e[:,:,support].mean()+25*physical
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
        if step%100==0 or step==checkpoint['config']['steps']-1:print(json.dumps({'event':'repair_train','seed':seed,'step':step+1,'loss':float(loss.detach())}),flush=True)
    corrected={'seed':seed,'reason':'Difference of rebased accumulated h includes old-coarse recalibration; it is not a single-packet h. Original current_only results are invalid controls.','steps':checkpoint['config']['steps'],'noise_and_data_schedule':'Replayed constructor and original Python/CUDA schedule; no prior retraining','evaluation':{}}
    generator=torch.Generator(device='cuda').manual_seed(seed+50000)
    with torch.no_grad():
        for item in val:
            z=move(item);mu,context=prior(z['anchor'],z['roi']);noise=torch.randn(1,32,device='cuda',generator=generator)
            predictions=head(matrix,mu,context,z['packet_grams'][None],z['packet_rhs'][None],z['packet_counts'][None],noise,scale,already_incremental=True)[0]
            target=item['supervision']['canonical_hr'].cuda().permute(2,0,1)[None].float()/255;roi,support=masks(item,'cuda');audits=[];rgbs=[]
            for t in [0,17,38]:
                rgb=decoded(z,predictions[:,t],t);bounded=q.bounded(rgb,z['coarse'][t:t+1]);clean,raw=witness(bounded,item,t)
                audits.append({'frame':t,'affine':rgb_metrics(rgb,target,roi,support),'bounded':rgb_metrics(bounded,target,roi,support),'initialfootprint_lr_rmse':float((q.observe(bounded)-z['coarse'][t:t+1]).square().mean().sqrt()),'past_witness_clean_rmse':float(clean.sqrt()),'past_witness_raw_rmse':float(raw.sqrt()),'coefficient_norm':float(predictions[:,t].norm())});rgbs.append(bounded)
            key=f"val/{Path(item['source']['path']).stem}";corrected['evaluation'][key]={'audits':audits,'rgb_revision_17_38':float((rgbs[-1]-rgbs[1]).square().mean())}
    (run/'current_control_corrected.json').write_text(json.dumps(corrected,indent=2));checkpoint['heads']['current_only']=head.state_dict();checkpoint['current_only_correction']=corrected['reason'];torch.save(checkpoint,run/'checkpoint_current_corrected.pt')
    result=json.loads((run/'results.json').read_text());result['invalid_legacy_controls']=['current_only from originalresults.json; replacedonlyvia400stepretrainingwithdirectpacketstats'];result['current_only_correction']=corrected['reason']
    for source,metric in corrected['evaluation'].items():result['evaluation'][source]['current_only']=metric
    (run/'results_corrected_controls.json').write_text(json.dumps(result,indent=2));print(json.dumps({'event':'repair_complete','seed':seed}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--suite',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43]);args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(Path(args.comfy).resolve()))
    while True:
        try:state=json.loads((Path(args.suite)/'suite_state.json').read_text())
        except (OSError,json.JSONDecodeError):state={'jobs':[]}
        if len(state['jobs'])==3:break
        time.sleep(10)
    for seed in args.seeds:repair(args.cache,Path(args.suite)/f'seed{seed}',args.vae,args.comfy)


if __name__=='__main__':main()
