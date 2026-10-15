"""Causal LR-only grouped evidence inputs; HR is separate supervision only."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cache_pairs import image,photo,degrade
from native_vae import load_vae,encode_repeated,decode_last,CANONICAL_SHA
from pixel_dc import Projection

FP=[(32,32),(32,96),(96,32),(96,96)]
PACKET_STEPS=[1,2,5,6,10,12,14,16]
PACKET_FP=[0,1,2,3,3,0,1,0]
PACKET_GROUP=[0,0,1,1,1,0,1,0]
QUERY=[3,4,7,8,11,13,17]
FAR={0:(0,0),3:(0,1),4:(0,0),7:(0,2),8:(0,0),11:(0,1),13:(1,2),17:(1,0)}


def modes(h,w,fp=None,size=256,device='cpu'):
    top,left,span=(0,0,size) if fp is None else (*fp,128)
    yy,xx=torch.meshgrid(torch.linspace(-1+2*top/(size-1),-1+2*(top+span-1)/(size-1),h,device=device),torch.linspace(-1+2*left/(size-1),-1+2*(left+span-1)/(size-1),w,device=device),indexing='ij')
    return torch.stack((xx,yy,xx*yy,xx.square()-1/3,yy.square()-1/3,torch.cos(torch.pi*xx/2)*torch.cos(torch.pi*yy/2)))


def inverse_photo(x,q,fp=None):
    m=modes(*x.shape[-2:],fp=fp,device=x.device);shade=(1+(q[6:,None,None]*m).sum(0)).clamp(.3,2)
    return ((x-q[3:6,None,None])/(1+q[:3,None,None]).clamp(.5,1.5)/shade).clamp(0,1)


def fit_photo(reference,y,fp=None):
    """Fixed robust LR-only regression; no true q/content/foreground mask."""
    if fp is None:x=reference;z=y
    else:
        top,left=fp;x=reference[:,top//4:top//4+32,left//4:left//4+32];z=F.interpolate(y[None],size=(32,32),mode='area')[0]
    m=modes(*x.shape[-2:],fp=fp,device=x.device);coeff=[];residuals=[];inliers=[]
    for c in range(3):
        xx=x[c].flatten();yy=z[c].flatten();design=torch.stack((torch.ones_like(xx),xx,*[xx*v.flatten() for v in m]),1).double();target=yy.double();weight=((xx>.01)&(xx<.99)&(yy>.01)&(yy<.99)).double();prior=torch.zeros(8,dtype=torch.float64,device=x.device);prior[1]=1
        penalty=torch.diag(torch.tensor([.0001,.001,*([.005]*6)],dtype=torch.float64,device=x.device))
        for _ in range(3):
            n=weight.sum().clamp_min(1);beta=torch.linalg.solve(design.T@(weight[:,None]*design)/n+penalty,design.T@(weight*target)/n+penalty@prior);res=target-design@beta;weight=(.025/res.abs().clamp_min(.025))*((xx>.01)&(xx<.99)&(yy>.01)&(yy<.99))
        coeff.append(beta.float());residuals.append(float(res.square().mean().sqrt()));inliers.append(float((res.abs()<.025).float().mean()))
    coeff=torch.stack(coeff);gain=coeff[:,1].clamp(.7,1.3);bias=coeff[:,0].clamp(-.05,.05);shade=(coeff[:,2:]/gain[:,None]).mean(0).clamp(-.15,.15);q=torch.cat((gain-1,bias,shade));quality=torch.tensor([float(fp is None),x.numel()/(3*4096),*residuals,float(np.mean(inliers))],device=x.device);return torch.cat((q,quality))


def confidence(reference,descriptor,fp,q_factor,q_far=None):
    top,left=fp
    if q_far is not None:reference=inverse_photo(reference,q_far)
    ref=F.interpolate(reference[:,top//4:top//4+32,left//4:left//4+32][None],size=(8,8),mode='area')[0]
    obs=inverse_photo(descriptor,q_factor,fp);err=(obs-ref).square().mean(0).sqrt();weight=torch.exp(-(err/.05).square()).clamp(.05,1)
    result=torch.zeros(1,16,16,device=reference.device);result[:,top//16:top//16+8,left//16:left//16+8]=weight;return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);p.add_argument('--counts',nargs=2,type=int,default=[32,8]);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);start=time.time()
    def live(**extra):
        value=dict(command=' '.join(sys.argv),status='B3R_cache_building',pid=os.getpid(),unix_time=time.time(),foreground=True,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),**extra);(out.parent/'LIVE_STATE.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
    live(step=0,steps=sum(args.counts));vae=load_vae(args.vae,args.comfy);far=Projection(256,4);near=Projection(128,2);pn=near.right@near.matrix;pf=far.right@far.matrix
    def null(x):return x-pf@x@pf.T
    def band(x,fp):
        top,left=fp;z=torch.zeros_like(x);z[...,top:top+128,left:left+128]=pn@x[...,top:top+128,left:left+128]@pn.T;return z
    def view(reference_up,y,fp):
        top,left=fp;v=reference_up.clone();crop=v[...,top:top+128,left:left+128];v[...,top:top+128,left:left+128]=near.affine(crop,y[None]);return v.clamp(0,1)
    def code_pair(v,fp):
        top,left=fp;proxy=v.clone();proxy[...,top:top+128,left:left+128]=F.interpolate(degrade(v[...,top:top+128,left:left+128],4),size=(128,128),mode='bicubic',align_corners=False,antialias=True).clamp(0,1);z=encode_repeated(vae,v);zp=encode_repeated(vae,proxy);return (z-zp)[0].cpu().half(),zp[0].cpu().half(),z[0].cpu().half()
    original=json.loads(Path(args.manifest).read_text());sources=sum(([x for x in original['items'] if x['split']==s][:n] for s,n in zip(['train','val'],args.counts)),[])
    manifest=dict(config=vars(args),native_H3_VAE=True,vae_sha256=CANONICAL_SHA,scope='Known nuisance capture groups, simulated photometry/foreground/content changes on natural source photos; LR-only runtime state; no DiT/test/video claim',capture_group_provenance='Simulator known locked-exposure segment tags, all controls share; not discovered by LR and contains no numerical q or content/visibility labels',causality='Group features/weights at each time use only arrived LR packets,far cues,8x8 retained descriptors/reference LR. HR and counterfactuals only under supervision.',QUERY=QUERY,packet_steps=PACKET_STEPS,packet_footprints=PACKET_FP,packet_groups=PACKET_GROUP,controller='Fixed IRLS12D photometric fit,coarse descriptor consistency exp(-(error/.05)^2),floor.05; no oracle change/occlusion mask',items=[])
    with torch.no_grad():
        for number,prov in enumerate(sources):
            destination=out/(prov['id']+'.pt')
            if destination.exists():manifest['items'].append(json.loads(destination.with_suffix('.json').read_text()));continue
            source=Path(prov['source']);donor=Path(prov['donor_source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==prov['source_sha256'];assert hashlib.sha256(donor.read_bytes()).hexdigest()==prov['donor_sha256'];a=image(source,256,'cuda');d=image(donor,256,'cuda');b=a.clone();b[:,64:128,64:128]=d[:,64:128,64:128];scenes=[a,b]
            rng=np.random.default_rng(prov['seed']+20261016);qs=[torch.zeros(12,device='cuda')]+[torch.tensor(np.r_[rng.uniform(-.18,.18,3),rng.uniform(-.025,.025,3),rng.uniform(-.09,.09,6)],device='cuda',dtype=torch.float32) for _ in range(2)];gen=torch.Generator(device='cuda').manual_seed(prov['seed']+20261016)
            def noisy(y):return (y+torch.randn(y.shape,device='cuda',generator=gen)*.003).clamp(0,1)
            yfar={t:noisy(degrade(photo(scenes[c],qs[g])[None],4)[0]) for t,(c,g) in FAR.items()};ref=yfar[0];refup=F.interpolate(ref[None],size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            near_y=[];near_desc=[];codes=[];proxies=[];fullcodes=[];neutralcodes=[];neutralproxies=[]
            for j,(t,fpi,g) in enumerate(zip(PACKET_STEPS,PACKET_FP,PACKET_GROUP)):
                scene=scenes[int(t>=12)].clone()
                if t==5:scene[:,112:144,64:96]=d[:,144:176,192:224]
                if t==14:scene[:,64:96,128:160]=d[:,160:192,32:64]
                top,left=FP[fpi];yi=noisy(degrade(photo(scene,qs[g+1])[None,:,top:top+128,left:left+128],2)[0]);near_y.append(yi);near_desc.append(F.interpolate(yi[None],size=(8,8),mode='area')[0]);z,zp,zfull=code_pair(view(refup,yi,FP[fpi]),FP[fpi]);codes.append(z);proxies.append(zp);fullcodes.append(zfull)
                # Counterfactual training label only, never runtime inputs.
                yn=degrade(scene[None,:,top:top+128,left:left+128],2)[0];zn,zpn,_=code_pair(view(refup,yn,FP[fpi]),FP[fpi]);neutralcodes.append(zn);neutralproxies.append(zpn)
            groupfeat=torch.zeros(2,18,device='cuda');group_is_far=[False,False];weights=torch.zeros(8,1,16,16,device='cuda');group_history=[];weight_history=[];present=[];query_features=[];query_z=[];query_base=[];query_lr=[];target_rgb=[];target_latent=[];query_q=[]
            for t in range(18):
                if t in PACKET_STEPS:
                    j=PACKET_STEPS.index(t);g=PACKET_GROUP[j];fp=FP[PACKET_FP[j]];feat=fit_photo(ref,near_y[j],fp)
                    if not group_is_far[g]:groupfeat[g]=feat
                    present.append(j);weights[j]=confidence(ref,near_desc[j],fp,groupfeat[g,:12])
                if t in FAR and t!=0:
                    c,appearance=FAR[t];feat=fit_photo(ref,yfar[t])
                    if appearance>0:groupfeat[appearance-1]=feat;group_is_far[appearance-1]=True
                    qf=feat[:12] if appearance>0 else torch.zeros(12,device='cuda')
                    for j in present:weights[j]=confidence(yfar[t],near_desc[j],FP[PACKET_FP[j]],groupfeat[PACKET_GROUP[j],:12],q_far=qf)
                    if t in QUERY:
                        up=F.interpolate(yfar[t][None],size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1);zl=encode_repeated(vae,up);query_z.append(zl[0].cpu().half());query_base.append(decode_last(vae,zl)[0].cpu().float());query_lr.append(yfar[t].cpu().float());query_features.append(feat.cpu().float());gt=photo(scenes[c],qs[appearance])[None];target_rgb.append(gt[0].cpu().half());query_q.append(qs[appearance].cpu())
                        total=torch.zeros_like(gt);count=torch.zeros_like(gt[:,:1]);contrast=null(gt-up)
                        for j in present:
                            fp=FP[PACKET_FP[j]];total+=band(contrast,fp);top,left=fp;count[...,top:top+128,left:left+128]+=1
                        teacher=(up+null(total/count.clamp_min(1))).clamp(0,1);zt=encode_repeated(vae,teacher);target_latent.append((zt-zl)[0].cpu().half())
                group_history.append(groupfeat.cpu().clone());weight_history.append(weights.cpu().clone())
            inputs=dict(packet_delta=torch.stack(codes),packet_proxy=torch.stack(proxies),group_features=torch.stack(group_history),weights=torch.stack(weight_history),query_features=torch.stack(query_features),query_latents=torch.stack(query_z),query_base_rgb=torch.stack(query_base),packet_steps=torch.tensor(PACKET_STEPS),packet_fp=torch.tensor(PACKET_FP),packet_groups=torch.tensor(PACKET_GROUP),query_steps=torch.tensor(QUERY),query_groups=torch.tensor([FAR[t][1]-1 for t in QUERY]),reference_LR=ref.cpu().float(),coarse_descriptors=torch.stack(near_desc).cpu().float())
            supervision=dict(detail_latents=torch.stack(target_latent),rgb=torch.stack(target_rgb),q_groups=torch.stack(qs[1:]).cpu(),query_q=torch.stack(query_q),neutral_delta=torch.stack(neutralcodes),neutral_proxy=torch.stack(neutralproxies))
            classical_inputs=dict(near_LR=torch.stack(near_y).cpu().float(),query_LR=torch.stack(query_lr),packet_full_latents=torch.stack(fullcodes),reference_LR=ref.cpu().float())
            record=dict(**prov,revision_cache_path=str(destination),known_nuisance_capture_groups=True,test_accessed=False);torch.save(dict(inputs=inputs,supervision=supervision,classical_inputs=classical_inputs,provenance=record),destination);destination.with_suffix('.json').write_text(json.dumps(record,indent=2));manifest['items'].append(record);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));live(step=number+1,steps=len(sources),source=prov['id'],seconds=time.time()-start)
    manifest.update(seconds=time.time()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,precision='native E/D FP16,strict FP32 simulator/teacher,noTF32',peak_GPU_bytes=torch.cuda.max_memory_allocated());(out/'manifest.json').write_text(json.dumps(manifest,indent=2));live(status_final='cache_complete',step=len(sources),steps=len(sources),seconds=manifest['seconds'])


if __name__=='__main__':main()
