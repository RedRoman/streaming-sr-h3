"""CPU natural-source check of the proposed complementary-footprint protocol.

Only neutral appearance/no occlusion: this establishes information headroom,
not a learned/H3 result. All reconstructions receive strictly prefix LR only.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from cache_pairs import image,degrade
from pixel_dc import Projection

FOOTPRINTS=[(32,32),(32,96),(96,32),(96,96)]


def cg(normal,b,iterations):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rr=r.square().sum()
    initial=float(rr)
    for _ in range(iterations):
        ap=normal(p);alpha=rr/(p.mul(ap).sum().clamp_min(1e-30));x=x+alpha*p;r=r-alpha*ap
        new=r.square().sum()
        if float(new)<max(initial*1e-14,1e-26):break
        p=r+(new/rr)*p;rr=new
    return x,float(r.square().sum().sqrt()/b.square().sum().sqrt().clamp_min(1e-15))


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--iterations',type=int,default=48);p.add_argument('--ridge',type=float,default=1e-4);p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--count',type=int,default=8);args=p.parse_args()
    torch.set_num_threads(4);start=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(args.manifest).read_text());sources=[x for x in manifest['items'] if x['split']==args.split][:args.count];assert len(sources)==args.count
    far=Projection(256,4,device='cpu');near=Projection(128,2,device='cpu')
    k0=far.matrix.double();u0=far.right.double();k=near.matrix.double();u=near.right.double()
    def null(x):return x-u0@(k0@x@k0.T)@u0.T
    def observe(x,fp):
        y,x0=fp;return k@x[...,y:y+128,x0:x0+128]@k.T
    def adjoint(y,fp):
        top,left=fp;z=torch.zeros((1,3,256,256),dtype=torch.float64);z[...,top:top+128,left:left+128]=k.T@y@k;return z
    def invert(base,packets):
        b=sum((adjoint(y-observe(base,fp),fp) for fp,y in packets),torch.zeros_like(base));b=null(b)
        def normal(v):return null(sum((adjoint(observe(null(v),fp),fp) for fp,_ in packets),torch.zeros_like(v)))+args.ridge*v
        h,res=cg(normal,b,args.iterations);return base+null(h),res
    def atlas(base,packets):
        total=torch.zeros_like(base);count=torch.zeros_like(base)
        for fp,y in packets:
            top,left=fp;total[...,top:top+128,left:left+128]+=u@(y-observe(base,fp))@u.T;count[...,top:top+128,left:left+128]+=1
        return base+null(total/count.clamp_min(1))
    rows=[];max_anchor=0.0
    def measure(prov,order,prefix,name,pred,gt,mask,base,yfar,cgres=None):
        nonlocal max_anchor
        # Box only for RGB quality. Physical anchor is audited before boxing.
        max_anchor=max(max_anchor,float((k0@(pred-base)@k0.T).square().mean().sqrt()))
        pred=pred.clamp(0,1);err=(pred-gt).square()
        rows.append(dict(source=prov['id'],order=order,prefix=prefix,method=name,psnr=float(-10*torch.log10(err.mean())),union_roi_psnr=float(-10*torch.log10(err[mask.expand_as(err)].mean())),current_far_rmse=float((k0@pred@k0.T-yfar).square().mean().sqrt()),cg_relative_residual=cgres))
    for index,prov in enumerate(sources):
        source=Path(prov['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==prov['source_sha256']
        gt=image(source,256,'cpu')[None].double();gen=torch.Generator().manual_seed(prov['seed']+3701)
        yfar=degrade(gt.float(),4).double();yfar=(yfar+torch.randn(yfar.shape,generator=gen,dtype=torch.float64)*.003).clamp(0,1)
        up=F.interpolate(yfar,size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1);base=up+u0@(yfar-k0@up@k0.T)@u0.T
        packets=[]
        for fp in FOOTPRINTS:
            y=observe(gt,fp);packets.append((fp,(y+torch.randn(y.shape,generator=gen,dtype=torch.float64)*.003).clamp(0,1)))
        for order,indices in [('forward',list(range(4))),('reverse',list(reversed(range(4))))]:
            history=[];mask=torch.zeros((1,1,256,256),dtype=torch.bool)
            for prefix,i in enumerate(indices,1):
                history.append(packets[i]);top,left=packets[i][0];mask[...,top:top+128,left:left+128]=True
                last,reslast=invert(base,history[-1:]);full,resfull=invert(base,history)
                for name,pred,res in [('current_far',base,None),('last_near_inverse',last,reslast),('last_near_direct_residual',atlas(base,history[-1:]),None),('aligned_residual_average',atlas(base,history),None),('all_history_linear_inverse',full,resfull)]:measure(prov,order,prefix,name,pred,gt,mask,base,yfar,res)
        with (out/'rgb_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        live=dict(status='complementary_natural_CPU_probe_running',source=prov['id'],step=index+1,steps=len(sources),seconds=time.time()-start,command=' '.join(sys.argv),GPU_used=False);(out/'LIVE_STATE.json').write_text(json.dumps(live,indent=2));print(json.dumps(live),flush=True)
    summary=[]
    for prefix in range(1,5):
        for method in ['current_far','last_near_inverse','last_near_direct_residual','aligned_residual_average','all_history_linear_inverse']:
            subset=[r for r in rows if r['prefix']==prefix and r['method']==method];summary.append(dict(prefix=prefix,method=method,n_sources=len(sources),n_orders=2,mean_psnr=float(np.mean([r['psnr'] for r in subset])),mean_union_roi_psnr=float(np.mean([r['union_roi_psnr'] for r in subset]))))
    deltas=[]
    for prov in sources:
        for prefix in [2,3,4]:
            full=[r['union_roi_psnr'] for r in rows if r['source']==prov['id'] and r['prefix']==prefix and r['method']=='all_history_linear_inverse'];last=[r['union_roi_psnr'] for r in rows if r['source']==prov['id'] and r['prefix']==prefix and r['method']=='last_near_inverse'];deltas.append(dict(source=prov['id'],prefix=prefix,full_minus_last_union_roi_dB=float(np.mean(full)-np.mean(last))))
    config=dict(**vars(args),sources=[{k:v for k,v in x.items() if k in ['id','split','source','source_sha256']} for x in sources],footprints=FOOTPRINTS,scope='Natural source photos, exact known neutral camera, no H3 and no neural training; establishes multi-view information headroom only',causal=True,HR_used_only_measurement_simulation_and_evaluation=True,ridge_fixed_before_running=True,prebox_current_anchor_difference_max_rms=max_anchor,seconds=time.time()-start,torch=torch.__version__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary=summary,source_paired_differences=deltas)
    (out/'complete.json').write_text(json.dumps(config,indent=2));(out/'LIVE_STATE.json').write_text(json.dumps(dict(status='complete',step=len(sources),steps=len(sources),seconds=config['seconds'],GPU_used=False),indent=2));print(json.dumps(dict(status='complete',seconds=config['seconds'],summary=summary)),flush=True)


if __name__=='__main__':main()
