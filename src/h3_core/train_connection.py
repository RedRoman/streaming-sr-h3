"""Fixed4-source/2-val,4matched-head actualD connection pilot,300steps each."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from connection_model import ConnectionHead,rollout,embed
from connection_data import SeparableCamera,VIEWS
from h3_interface import load_frozen_vae,differentiable_decoder_math
from observation_quotient import ObservationQuotient


KINDS={'connection_loop':'transport','basic_sequential':'transport','gru':'gru','set':'set'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--comfy',required=True);p.add_argument('--vae',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=17);p.add_argument('--resume',action='store_true');p.add_argument('--eval-only',action='store_true');args=p.parse_args()
    began=time.monotonic();torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(Path(args.comfy).resolve()))
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True);root=Path(args.root);manifest=json.loads((root/'manifest.json').read_text());vae,vae_info=load_frozen_vae(args.vae);q=ObservationQuotient();cameras={k:SeparableCamera(v,q) for k,v in VIEWS.items()}
    items=[]
    def cuda(value):
        if torch.is_tensor(value):return value.cuda()
        if isinstance(value,dict):return {k:cuda(v) for k,v in value.items()}
        return value
    for source in manifest['sources']:
        folder=root/source['id'];item=cuda(torch.load(folder/'learning_inputs.pt',map_location='cpu',weights_only=False));item['target']=torch.load(folder/'supervision.pt',map_location='cpu',weights_only=False)['target'].cuda().permute(2,0,1)[None].float()/255;items.append(item)
    train=[v for v in items if v['source']['split']=='train'];val=[v for v in items if v['source']['split']=='val'];assert len(train)==4 and len(val)==2
    prototype=ConnectionHead('transport');weights=copy.deepcopy(prototype.state_dict());heads={key:ConnectionHead(kind).cuda() for key,kind in KINDS.items()}
    for head in heads.values():head.load_state_dict(weights)
    optimizers={k:torch.optim.Adam(h.parameters(),lr=1e-3) for k,h in heads.items()};step0=0;history=[]
    if args.resume or args.eval_only:
        saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);step0=saved['step'];history=saved['history']
        for k,h in heads.items():h.load_state_dict(saved['heads'][k]);optimizers[k].load_state_dict(saved['optimizers'][k])
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    def save(step):
        state={'config':vars(args),'step':step,'history':history,'heads':{k:h.state_dict() for k,h in heads.items()},'optimizers':{k:o.state_dict() for k,o in optimizers.items()},'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()};torch.save(state,out/'checkpoint_next.pt');(out/'checkpoint_next.pt').replace(out/'checkpoint.pt')
    def decode(item,state,initial,bounded=False):
        rgb=vae.decode(embed(item,state,initial).half())[:,:,21].cuda().float();return q.bounded(rgb,item['observations']['far']) if bounded else q.readout(rgb,item['observations']['far'])
    def all_states(head,item):
        initial,ab=rollout(head,item,['A','B']);_,ba=rollout(head,item,['B','A']);return initial,{**ab,**ba}
    def common_loss(rgb,item,prefix):
        private=item['private_mask'];error=(rgb-item['target']).square();reconstruction=error[:,:,~private].mean();identity=q.null(rgb-item['rgb_reference'])[:,:,private].square().mean();lr=torch.stack([(cameras[k].observe(rgb)-item['observations'][k]).square().mean() for k in sorted(set(prefix))]).mean();return 100*reconstruction+25*lr+100*identity
    if not args.eval_only:
        for step in range(step0,args.steps):
            item=random.choice(train);losses={}
            for name,head in heads.items():
                optimizers[name].zero_grad(set_to_none=True)
                with differentiable_decoder_math():
                    initial,states=all_states(head,item);rgbs={k:decode(item,state,initial) for k,state in states.items()};loss=torch.stack([common_loss(rgb,item,key) for key,rgb in rgbs.items()]).mean();loop=(rgbs['AB']-rgbs['BA']).square().mean();code_loop=(states['AB']-states['BA']).square().mean()
                    if name=='connection_loop':loss=loss+25*loop+.01*code_loop
                    loss.backward()
                assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in head.parameters());assert all(v.grad is None for v in vae.parameters());gradient=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.));optimizers[name].step();losses[name]={'loss':float(loss.detach()),'RGB_loop_mse':float(loop.detach()),'code_loop_mse':float(code_loop.detach()),'gradient_norm':gradient}
            if step%25==0 or step+1==args.steps:
                event={'step':step+1,'source':item['source']['id'],'heads':losses,'seconds':time.monotonic()-began};history.append(event);print(json.dumps({'event':'connection_train',**event}),flush=True)
            if (step+1)%25==0 or step+1==args.steps:save(step+1);(out/'progress.json').write_text(json.dumps({'status':'training','step':step+1,'steps_per_head':args.steps,'head_count':4,'seconds':time.monotonic()-began},indent=2))
    result={'scope':'Fixed4train2val developmentpilot; not testconfirmation or H3semanticfactoridentification','config':vars(args),'native_response_budget':'56officialH3samples/2744full50forwards shared; headtraining actualD only','head_parameters':{k:sum(p.numel() for p in h.parameters()) for k,h in heads.items()},'shared_U_private':'SameimmutableX0channelPCA8/privatecarrier for allheads; not semanticallyidentified','steps_per_head':args.steps,'loss':'100RGBoutside4cornerprivatequeries+25observedLR+100reference-privateHF; proposedonly+25RGBloop+.01codeloop','source_hashes':{v['source']['id']:v['source']['sha256'] for v in items},'vae':vae_info,'training_history':history,'evaluation':{},'protocol_checks':{}}
    def metric(rgb,item,prefix):
        error=(rgb-item['target']).square();private=item['private_mask'];globalquery=item['global_query_mask'];delta=q.null(rgb-item['rgb_reference']);lr=torch.stack([(cameras[k].observe(rgb)-item['observations'][k]).square().mean() for k in sorted(set(prefix))]).mean()
        gradient=(rgb[:,:,:,1:]-rgb[:,:,:,:-1]).abs().mean()+(rgb[:,:,1:]-rgb[:,:,:-1]).abs().mean()
        return {'psnr_full_db':float(-10*torch.log10(error.mean().clamp_min(1e-12))),'psnr_global_query_db':float(-10*torch.log10(error[:,:,globalquery].mean().clamp_min(1e-12))),'observed_prefix_lr_rmse':float(lr.sqrt()),'far_lr_rmse':float((q.observe(rgb)-item['observations']['far']).square().mean().sqrt()),'private_query_revision_rms':float(delta[:,:,private].square().mean().sqrt()),'global_query_revision_rms':float(delta[:,:,globalquery].square().mean().sqrt()),'gradient_energy_l1':float(gradient),'highfrequency_rms':float(q.null(rgb).square().mean().sqrt())}
    with torch.no_grad():
        for item in val+train:
            rows={}
            for name,head in heads.items():
                initial,states=all_states(head,item);rgbs={k:decode(item,state,initial,True) for k,state in states.items()};rows[name]={'prefixes':{k:metric(rgb,item,k) for k,rgb in rgbs.items()},'RGB_path_mse':float((rgbs['AB']-rgbs['BA']).square().mean()),'latent_path_mse':float((states['AB']-states['BA']).square().mean())}
            result['evaluation'][item['source']['id']]=rows;print(json.dumps({'event':'connection_eval','source':item['source']['id'],'final_global_db':{k:.5*(v['prefixes']['AB']['psnr_global_query_db']+v['prefixes']['BA']['psnr_global_query_db']) for k,v in rows.items()},'path_mse':{k:v['RGB_path_mse'] for k,v in rows.items()}}),flush=True)
            (out/'results.json').write_text(json.dumps(result,indent=2))
        probe=val[0];changed=copy.deepcopy(probe);changed['responses']['AB']+=100;changed['cue_features']['B']+=100
        for name,head in heads.items():
            initial,a=rollout(head,probe,['A']);_,b=rollout(head,changed,['A']);_,duplicate=rollout(head,probe,['A','A']);result['protocol_checks'][name]={'future_prefix_maxabs':float((a['A']-b['A']).abs().max()),'duplicate_maxabs':float((a['A']-duplicate['A']).abs().max())};assert a['A'].equal(b['A']) and a['A'].equal(duplicate['A'])
    result.update(seconds=time.monotonic()-began,total_optimizer_steps=4*args.steps,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());(out/'results.json').write_text(json.dumps(result,indent=2));(out/'progress.json').write_text(json.dumps({'status':'complete','steps_per_head':args.steps,'total_optimizer_steps':4*args.steps},indent=2));print(json.dumps({'event':'connection_complete','seconds':result['seconds'],'parameters':result['head_parameters']}),flush=True)


if __name__=='__main__':main()
