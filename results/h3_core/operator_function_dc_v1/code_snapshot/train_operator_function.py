"""Fixed three-head operator-function pilot through the actual frozen H3 D."""
import argparse, csv, json, random, sys, time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from operator_function_model import KINDS,build,opposite_correlation_probe
from operator_function_data import COORDS,cameras_for
from h3_interface import load_frozen_vae,differentiable_decoder_math
from observation_quotient import ObservationQuotient


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--comfy',required=True);p.add_argument('--vae',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=17);p.add_argument('--smoke',action='store_true');p.add_argument('--eval-only',action='store_true');a=p.parse_args()
    began=time.monotonic();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(a.seed);random.seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;sys.path.insert(0,str(Path(a.comfy).resolve()));vae,_=load_frozen_vae(a.vae);q=ObservationQuotient();calls={'decode_train':0,'decode_evaluation':0,'decode_smoke_teacher':0};manifest=json.loads((root/'manifest.json').read_text())
    def cuda(v):
        if torch.is_tensor(v):return v.cuda()
        if isinstance(v,dict):return {k:cuda(x) for k,x in v.items()}
        return v
    items=[]
    for source in manifest['sources']:
        if a.smoke and items:break
        folder=root/source['id']
        if a.smoke:
            item=cuda(torch.load(folder/'prepared.pt',map_location='cpu',weights_only=False));item['response']=(item['guides']-item['guide0'])[None].expand(2,-1,-1,-1,-1,-1).clone();item['VAE_response']=item['response'].clone();item['target']=torch.load(Path(item['old_folder'])/'supervision.pt',map_location='cpu',weights_only=False)['target'].cuda().permute(2,0,1)[None].float()/255
            y,x=torch.meshgrid(torch.arange(256,device='cuda'),torch.arange(256,device='cuda'),indexing='ij');item['private_mask']=((x<32)|(x>=224))&((y<32)|(y>=224));teacher=[]
            with torch.no_grad():
                for u in range(2):
                    row=[]
                    for j in range(5):row.append(q.bounded(vae.decode((item['basez'][u:u+1]+item['response'][u:u+1,j]).half())[:,:,21].cuda().float(),item['observations']['far']));calls['decode_smoke_teacher']+=1
                    teacher.append(torch.cat(row))
            item['teacher_rgb']=torch.stack(teacher)
        else:item=cuda(torch.load(folder/'family.pt',map_location='cpu',weights_only=False))
        items.append(item)
    train=[v for v in items if v['source']['split']=='train'];assert len(train)==(1 if a.smoke else 4)
    prototype=build('mixed');encoder=prototype.encoder.state_dict();heads={k:build(k).cuda() for k in KINDS}
    for h in heads.values():h.encoder.load_state_dict(encoder)
    optimizers={k:torch.optim.Adam(h.parameters(),lr=.001) for k,h in heads.items()};history=[]
    if a.eval_only:
        saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        for k,h in heads.items():h.load_state_dict(saved['heads'][k])
    def decode(z,item,stage,bounded=False):
        calls[stage]+=1;rgb=vae.decode(z.half())[:,:,21].cuda().float()
        return q.bounded(rgb,item['observations']['far']) if bounded else q.readout(rgb,item['observations']['far'])
    def observe_loss(rgb,item,g):
        cams=cameras_for(g,q);return torch.stack([(cams[k].observe(rgb)-item['observations'][k]).square().mean() for k in ['A','B']]).mean()
    if not a.eval_only:
        for step in range(a.steps):
            item=random.choice(train);u=random.randrange(2);j=random.randrange(3);second=(j+1)%3;carrier=item['basez'][u:u+1];g=carrier.new_tensor(COORDS[j]);other=carrier.new_tensor(COORDS[second]);private=item['private_mask'];losses={}
            # No held/interior/LR-inferred coordinate enters training. HR loss
            # also excludes the union of all uncertain A footprints, so a wrong
            # operator is not forced to hallucinate the same nominal GT detail.
            known_region=(~private).clone();known_region[28:164,28:164]=False
            for kind,h in heads.items():
                optimizers[kind].zero_grad(set_to_none=True)
                with differentiable_decoder_math():
                    memory=h.memory(item['response'][u:u+1,:3],carrier);prediction=h.query(memory,carrier,g);otherprediction=h.query(memory,carrier,other)
                    rgb=decode(carrier+prediction,item,'decode_train');other_rgb=decode(carrier+otherprediction,item,'decode_train');teacher=item['teacher_rgb'][u:u+1,j];teacher2=item['teacher_rgb'][u:u+1,second];base=item['basergb'][u:u+1]
                    latent=.05*((prediction-item['response'][u:u+1,j]).square().mean()+(otherprediction-item['response'][u:u+1,second]).square().mean());teacherloss=50*((rgb[:,:,~private]-teacher[:,:,~private]).square().mean()+(other_rgb[:,:,~private]-teacher2[:,:,~private]).square().mean());identity=50*(q.null(rgb-base)[:,:,private].square().mean()+q.null(other_rgb-base)[:,:,private].square().mean())
                    reconstruction=50*((rgb[:,:,known_region]-item['target'][:,:,known_region]).square().mean()+(other_rgb[:,:,known_region]-item['target'][:,:,known_region]).square().mean());lr=12.5*(observe_loss(rgb,item,COORDS[j])+observe_loss(other_rgb,item,COORDS[second]));loss=latent+teacherloss+identity+reconstruction+lr;loss.backward()
                assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in h.parameters());assert all(v.grad is None for v in vae.parameters());encgrad=sum(float(v.grad.square().sum()) for v in h.encoder.parameters() if v.grad is not None)**.5;gradient=float(torch.nn.utils.clip_grad_norm_(h.parameters(),1.));optimizers[kind].step();losses[kind]=dict(loss=float(loss.detach()),latent=float(latent.detach()),teacher_RGB=float(teacherloss.detach()),identity=float(identity.detach()),unambiguous_GT_RGB=float(reconstruction.detach()),LR=float(lr.detach()),gradient_norm=gradient,encoder_gradient_norm=encgrad)
            if step%25==0 or step+1==a.steps:
                event=dict(step=step+1,source=item['source']['id'],carrier_seed=[101,202][u],operator=j,losses=losses,seconds=time.monotonic()-began);history.append(event);print(json.dumps(dict(event='operator_function_train',**event)),flush=True)
            if (step+1)%25==0 or step+1==a.steps:
                torch.save(dict(step=step+1,config=vars(a),heads={k:h.state_dict() for k,h in heads.items()},optimizers={k:v.state_dict() for k,v in optimizers.items()},history=history),out/'checkpoint.pt');(out/'progress.json').write_text(json.dumps(dict(status='training',step=step+1,steps=a.steps,seconds=time.monotonic()-began),indent=2))
    result=dict(config=vars(a),scope='Development conditional-function compression after far/A/B arrival; no motion/longrevisit/novelty claim',parameters={k:sum(v.numel() for v in h.parameters()) for k,h in heads.items()},persistent_memory_shape=[24,7,16,16],carrier_shape=[24,7,16,16],same_storage_all_heads=True,opposite_correlation_probe=opposite_correlation_probe(),training_history=history,calls=calls,rows=[])
    if a.smoke:
        assert all(history[-1]['losses'][k]['encoder_gradient_norm']>0 for k in KINDS)
        result.update(status='smoke_passed',smoke_response='VAE-guide response placeholder, not actual conditional H3 family; excluded from formal model initialization',base_weight_grad_count=0,seconds=time.monotonic()-began);(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(event='operator_function_smoke_complete',parameters=result['parameters'],calls=calls)),flush=True);return
    def metrics(rgb,item,u,g):
        error=(rgb-item['target']).square();private=item['private_mask'];base=item['basergb'][u:u+1];delta=q.null(rgb-base)
        return dict(psnr_full_db=float(-10*torch.log10(error.mean().clamp_min(1e-12))),private_HF_revision_RMS=float(delta[:,:,private].square().mean().sqrt()),HF_RMS=float(q.null(rgb).square().mean().sqrt()),hypothesis_LR_RMSE=float(observe_loss(rgb,item,g).sqrt()),nominal_A0B_LR_RMSE=float(observe_loss(rgb,item,(0.,0.)).sqrt()),far_LR_RMSE=float((q.observe(rgb)-item['observations']['far']).square().mean().sqrt()))
    exported={}
    with torch.no_grad():
        for item in items:
            sid=item['source']['id'];outputs={};queries=[('support0',COORDS[0]),('support1',COORDS[1]),('support2',COORDS[2]),('held_composition',COORDS[3]),('held_interior',COORDS[4]),('LR_inferred',item['inferred_query'].cpu().tolist())]
            for u in range(2):
                carrier=item['basez'][u:u+1]
                for kind,h in heads.items():
                    memory=h.memory(item['response'][u:u+1,:3],carrier);grid=[];grid_LR=[]
                    for query,g in queries:
                        response=h.query(memory,carrier,carrier.new_tensor(g));rgb=decode(carrier+response,item,'decode_evaluation',True);row=dict(source=sid,split=item['source']['split'],method=kind,carrier_seed=[101,202][u],query=query,**metrics(rgb,item,u,g));idx={'held_composition':3,'held_interior':4}.get(query)
                        if idx is not None:row['response_latent_MSE']=float((response-item['response'][u:u+1,idx]).square().mean());row['teacher_RGB_RMSE']=float((rgb-item['teacher_rgb'][u:u+1,idx]).square().mean().sqrt())
                        result['rows'].append(row)
                        if query!='LR_inferred':grid.append(rgb);grid_LR.append(row['hypothesis_LR_RMSE']**2)
                        else:outputs[f'{kind}_mean_operator_diagnostic_u{u}']=rgb.cpu()
                    conditional=item['conditional_late_weights'][u];combined=sum(conditional[j]*rgb for j,rgb in enumerate(grid));joint_row=dict(source=sid,split=item['source']['split'],method=kind,carrier_seed=[101,202][u],query='LR_joint_conditioned_function',**metrics(combined,item,u,item['inferred_query'].cpu().tolist()));joint_row['functional_expected_hypothesis_LR_RMSE']=float((conditional*conditional.new_tensor(grid_LR)).sum().sqrt());result['rows'].append(joint_row);outputs[f'{kind}_u{u}']=combined.cpu()
                    # Same trained readout with homologous VAE support is a
                    # fixed-checkpoint intervention, not a separately fitted head.
                    mv=h.memory(item['VAE_response'][u:u+1,:3],carrier);rv=h.query(mv,carrier,item['inferred_query']);rgbv=decode(carrier+rv,item,'decode_evaluation',True);result['rows'].append(dict(source=sid,split=item['source']['split'],method=kind+'_VAE_support_intervention',carrier_seed=[101,202][u],query='LR_inferred',**metrics(rgbv,item,u,item['inferred_query'].cpu().tolist())))
                conditional=item['conditional_late_weights'][u];controls={'source_carrier':item['basergb'][u:u+1],'joint_robust_inverse':item['robust_inverse'][u:u+1],'dense_conditional_H3_pushforward_not_posterior':(conditional[:,None,None,None]*item['teacher_rgb'][u]).sum(0,keepdim=True),'dense_VAE_family_pushforward':(conditional[:,None,None,None]*item['VAE_rgb'][u]).sum(0,keepdim=True)}
                # Exact finite-prior mean averages the fixed far-only atoms,
                # not the already cue-conditioned C_g candidates.
                marginal=item['joint_late_weights'].sum(1);controls['exact_finite_prior_joint_mean']=(marginal[:,None,None,None]*item['basergb']).sum(0,keepdim=True)
                # Same support-coordinate affine readout without learned H3
                # mechanism; this is a classical function interpolation control.
                design=carrier.new_tensor([[1,-1,-1],[1,-1,1],[1,1,-1]]);coef=torch.einsum('ij,jcthw->icthw',torch.linalg.inv(design),item['VAE_response'][u,:3]);gg=item['inferred_query'];vr=coef[0]+gg[0]*coef[1]+gg[1]*coef[2];controls['affine_VAE_family_at_LR_query']=decode(carrier+vr[None],item,'decode_evaluation',True)
                for name,rgb in controls.items():
                    result['rows'].append(dict(source=sid,split=item['source']['split'],method=name,carrier_seed=[101,202][u],query='LR_inferred',**metrics(rgb,item,u,item['inferred_query'].cpu().tolist())));outputs[f'{name}_u{u}']=rgb.cpu()
            torch.save(dict(source=item['source'],target_evaluation_only=item['target'].cpu(),outputs=outputs,private_mask=item['private_mask'].cpu(),joint_early_weights=item['joint_early_weights'].cpu(),joint_late_weights=item['joint_late_weights'].cpu(),inferred_query=item['inferred_query'].cpu()),out/f'{sid}_readouts.pt');print(json.dumps(dict(event='operator_function_eval',source=sid,rows=len(result['rows']))),flush=True)
    result.update(status='complete',optimizer_steps=0 if a.eval_only else 3*a.steps,seconds=time.monotonic()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());(out/'results.json').write_text(json.dumps(result,indent=2));fields=list(dict.fromkeys(k for row in result['rows'] for k in row))
    with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(result['rows'])
    (out/'progress.json').write_text(json.dumps(dict(status='complete',steps_each=a.steps,total_optimizer_steps=result['optimizer_steps'],calls=calls),indent=2));print(json.dumps(dict(event='operator_function_complete',seconds=result['seconds'],calls=calls)),flush=True)


if __name__=='__main__':main()
