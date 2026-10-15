#!/usr/bin/env python3
"""Aliasing identities, immutable episodes and exact compressed MHT controls.

No HR access in inference. The latent observations are sufficient statistics of
known 2x2 box-downsampled near views. Far 4x4 averages annihilate all latent modes.
The fork method is intentionally checked against standard exact Bayesian MHT.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'branch_memory'
SCENARIOS=['no_cue','weak_cue','true_change','duplicates','delayed_reveal','no_reveal']
METHODS=['single_map','hard_reject_reset','hard_reject_preserve','particle_mht_4','particle_mht_32','exact_mht','exact_mht_dedup','fork_merge']
K=4;D=48;N=16;SIGMA=.03;CHANGE_SD=.25;COMMIT=.95


def basis():
    B=np.zeros((N*N,D))
    patterns=np.array([[[1,1,-1,-1]]*4,[[1]*4,[1]*4,[-1]*4,[-1]*4],[[1,1,-1,-1],[1,1,-1,-1],[-1,-1,1,1],[-1,-1,1,1]]],float)/4
    for b in range(16):
        yy,xx=divmod(b,4)
        for j in range(3):
            z=np.zeros((N,N));z[yy*4:yy*4+4,xx*4:xx*4+4]=patterns[j]
            B[:,b*3+j]=z.ravel()
    assert np.max(np.abs(B.T@B-np.eye(D)))<1e-12
    assert np.max(np.abs(B.reshape(N,N,D).reshape(4,4,4,4,D).mean(axis=(1,3))))<1e-12
    return B


def gaussian_ll(y,mean,var):
    return -.5*np.sum(np.log(2*np.pi*var)+(y-mean)**2/var,axis=-1)


def init_branches(bank_mu,bank_v):
    mu=np.repeat(bank_mu,2,axis=0).copy();v=np.repeat(bank_v,2,axis=0).copy()
    v[1::2]+=CHANGE_SD**2
    lp=np.tile(np.log([.9,.1]),K)-np.log(K)
    return mu,v,lp


class EpisodeEngine:
    def __init__(self,kind,bank_mu,bank_v,rng):
        self.kind=kind;self.bank_mu=bank_mu.copy();self.bank_v=bank_v.copy();self.rng=rng
        self.archive_nodes=K;self.archives=[];self.alias_checks=[];self.idset=set();self.reset_episode()

    def reset_episode(self):
        self.prior_mu,self.prior_v,self.prior_lp=init_branches(self.bank_mu,self.bank_v)
        self.start_bank_mu=self.bank_mu.copy()
        self.mu=self.prior_mu.copy();self.v=self.prior_v.copy();self.lp=self.prior_lp.copy()
        self.g=np.zeros(D);self.b=np.zeros(D);self.c=np.zeros(D);self.count=np.zeros(D)
        self.cue_log=np.zeros(K);self.idset=set();self.informative=False;self.seen_unique=0
        self.merged=False;self.theta_last=self.bank_mu[0].copy();self.last_reset_id=None
        if self.kind.startswith('particle'):
            pcount=int(self.kind.split('_')[-1])
            self.particle_ids=self.rng.choice(K*2,size=pcount,p=np.exp(self.prior_lp))
            self.mu=self.prior_mu[self.particle_ids].copy();self.v=self.prior_v[self.particle_ids].copy()
            self.lp=np.full(pcount,-np.log(pcount))

    def compressed_posterior(self):
        if not np.any(self.count):
            lp=self.prior_lp+np.repeat(self.cue_log,2);lp-=logsumexp(lp)
            return self.prior_mu.copy(),self.prior_v.copy(),lp
        H=1/self.prior_v+self.g
        m=(self.prior_mu/self.prior_v+self.b)/H
        v=1/H
        ll=-.5*np.sum(self.c+self.prior_mu**2/self.prior_v-(self.prior_mu/self.prior_v+self.b)**2/H+np.log(self.prior_v*H)+self.count*np.log(2*np.pi*SIGMA**2),axis=1)
        lp=self.prior_lp+np.repeat(self.cue_log,2)+ll
        lp-=logsumexp(lp)
        return m,v,lp

    def candidate_view(self):
        if self.kind=='fork_merge':return self.compressed_posterior()
        return self.mu,self.v,self.lp-logsumexp(self.lp)

    def update(self,packet,prototypes,cue_sd):
        duplicate=packet['uid'] in self.idset
        dedup=self.kind in ('fork_merge','exact_mht_dedup')
        if not duplicate:self.seen_unique+=1
        self.idset.add(packet['uid'])
        idx=packet['idx'];y=packet['y'];cue=packet['cue']
        if not (duplicate and dedup):
            cue_ll=np.zeros(K) if cue is None else -.5*np.sum((cue-prototypes)**2,axis=1)/cue_sd**2
            self.cue_log+=cue_ll
            if len(idx):self.informative=True
            if self.kind=='fork_merge':
                self.g[idx]+=1/SIGMA**2;self.b[idx]+=y/SIGMA**2
                self.c[idx]+=y*y/SIGMA**2;self.count[idx]+=1
            elif self.kind in ('exact_mht','exact_mht_dedup') or self.kind.startswith('particle'):
                branch_cue=cue_ll[self.particle_ids//2] if self.kind.startswith('particle') else np.repeat(cue_ll,2)
                self.lp+=branch_cue
                if len(idx):
                    vp=self.v[:,idx]+SIGMA**2
                    self.lp+=gaussian_ll(y,self.mu[:,idx],vp)
                    gain=self.v[:,idx]/vp
                    self.mu[:,idx]+=gain*(y-self.mu[:,idx]);self.v[:,idx]*=1-gain
                self.lp-=logsumexp(self.lp)
                if self.kind.startswith('particle'):
                    w=np.exp(self.lp)
                    if 1/np.sum(w*w)<len(w)/2:
                        # Conventional bootstrap resampling may erase an identity
                        # before a revealing observation arrives.
                        inds=self.rng.choice(len(w),size=len(w),p=w)
                        self.particle_ids=self.particle_ids[inds]
                        self.mu=self.mu[inds].copy();self.v=self.v[inds].copy()
                        self.lp=np.full(len(w),-np.log(len(w)))
            else:
                # Single-state policies: no persistent competing identity state.
                bm,bv,blp=init_branches(self.bank_mu,self.bank_v)
                blp+=np.repeat(self.cue_log,2)
                if len(idx):blp+=gaussian_ll(y,bm[:,idx],bv[:,idx]+SIGMA**2)
                blp-=logsumexp(blp);best=int(np.argmax(blp));ident=best//2
                weights=np.exp(blp).reshape(K,2).sum(axis=1)
                if len(idx) and (self.kind=='single_map' or weights[ident]>=COMMIT):
                    if self.kind=='hard_reject_reset' and self.last_reset_id!=ident:
                        # A new single candidate is initialized from the current
                        # partial view, so unobserved confirmed content is lost.
                        self.bank_mu[ident]=0;self.bank_v[ident]=.4**2
                        prior_v=np.full(len(idx),.4**2)
                        self.last_reset_id=ident
                    else:prior_v=bv[best,idx]
                    gain=prior_v/(prior_v+SIGMA**2)
                    self.bank_mu[ident,idx]+=gain*(y-self.bank_mu[ident,idx])
                    self.bank_v[ident,idx]=prior_v*(1-gain)
                self.mu,self.v,self.lp=init_branches(self.bank_mu,self.bank_v)
                self.lp=blp
        mu,v,lp=self.candidate_view()
        weights=np.exp(lp)
        if self.kind.startswith('particle'):
            idprob=np.bincount(self.particle_ids//2,weights=weights,minlength=K)
            selected_id=int(np.argmax(idprob));cand=np.where(self.particle_ids//2==selected_id)[0]
            best=int(cand[np.argmax(weights[cand])]);selected_epoch=int(self.particle_ids[best]%2)
        else:
            idprob=weights.reshape(K,2).sum(axis=1)
            selected_id=int(np.flatnonzero(idprob>=np.max(idprob)-1e-12)[0])
            best=selected_id*2+int(weights[selected_id*2+1]>weights[selected_id*2]);selected_epoch=best%2
        self.theta_last=mu[best].copy();self.var_last=v[best].copy()
        if self.kind in ('single_map','hard_reject_reset','hard_reject_preserve'):
            self.theta_last=self.bank_mu[selected_id].copy();self.var_last=self.bank_v[selected_id].copy()
        self.selected_id=selected_id;self.selected_epoch=selected_epoch;self.idprob=idprob
        self.posterior_mu=mu;self.posterior_v=v;self.posterior_lp=lp
        return dict(selected_id=selected_id,selected_epoch=selected_epoch,confidence=float(idprob[selected_id]),duplicate=duplicate,active_bytes=self.active_bytes())

    def finish(self):
        if self.kind in ('single_map','hard_reject_reset','hard_reject_preserve'):return
        # No destructive registry write before both identity evidence and an
        # informative observation exist. An epoch change retains its old node.
        if self.informative and self.idprob[self.selected_id]>=COMMIT:
            if self.selected_epoch:
                self.archives.append((self.selected_id,self.bank_mu[self.selected_id].copy(),self.bank_v[self.selected_id].copy()))
            self.bank_mu[self.selected_id]=self.theta_last
            self.bank_v[self.selected_id]=self.var_last
            self.archive_nodes=K+len(self.archives)
            self.merged=True

    def active_bytes(self):
        bank=(K+len(self.archives))*D*2*8
        if self.kind=='fork_merge':dynamic=(4*D+K)*8
        elif self.kind.startswith('particle'):dynamic=len(self.lp)*(2*D+2)*8
        elif self.kind in ('exact_mht','exact_mht_dedup'):dynamic=K*2*(2*D+1)*8
        else:dynamic=K*8
        return bank+dynamic+len(self.idset)*16


def worlds(seed,episodes,scenario):
    rng=np.random.default_rng(seed+1000)
    theta=rng.normal(0,.35,(K,D))
    for i in range(K):
        theta[i]*=min(1,1.5/np.max(np.sum(np.abs(theta[i].reshape(16,3)),axis=1)))
    B=basis();base=np.full(N*N,.5)
    images=base[None,:]+theta@B.T
    far=images.reshape(K,4,4,4,4).mean(axis=(2,4))
    assert np.max(np.abs(far-far[0]))<1e-12
    enroll=theta+rng.normal(0,SIGMA/np.sqrt(3),(K,D))
    enrollv=np.full((K,D),SIGMA**2/3)
    prototypes=.30*np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
    cue_sd=1.0
    order=np.tile(rng.permutation(K),(episodes+K-1)//K)[:episodes]
    packets=[];uid=0
    for episode,identity in enumerate(order):
        changed=scenario=='true_change' and episode==episodes//2
        unchanged=np.ones(D,bool)
        if changed:
            delta=rng.normal(0,.40,12)
            theta[identity,:12]+=delta;unchanged[:12]=False
        truth=theta[identity].copy()
        far_count=12 if scenario=='delayed_reveal' else (8 if scenario=='no_reveal' else 3)
        cue_enabled=scenario not in ('no_cue','no_reveal')
        timeline=[np.array([],int) for _ in range(far_count)]
        if scenario!='no_reveal':timeline += [np.arange(j*12,(j+1)*12) for j in range(4)]
        event=[];reveal_count=0
        for idx in timeline:
            cue=prototypes[identity]+rng.normal(0,cue_sd,3) if cue_enabled else None
            y=truth[idx]+rng.normal(0,SIGMA,len(idx))
            packet=dict(uid=uid,idx=idx,y=y,cue=cue,identity=int(identity),truth=truth.copy(),unchanged=unchanged.copy(),changed=changed,episode=episode,reveal_index=reveal_count if len(idx) else -1)
            uid+=1
            event.append(packet)
            if len(idx):
                reveal_count+=1
                if scenario=='duplicates':event.extend([packet.copy() for _ in range(5)])
        packets.append(event)
    return packets,enroll,enrollv,prototypes,cue_sd,images


def ci(a):
    a=np.asarray(a);rng=np.random.default_rng(20261015)
    means=a[rng.integers(0,len(a),(5000,len(a)))].mean(axis=1)
    return float(a.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))


def writecsv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(rows,episode_rows,out):
    summary=[]
    for scenario in SCENARIOS:
        for method in METHODS:
            rr=[r for r in rows if r['scenario']==scenario and r['method']==method]
            ee=[r for r in episode_rows if r['scenario']==scenario and r['method']==method]
            for metric,phase in [('correct_branch','far'),('correct_branch','reveal'),('psnr','reveal'),('unchanged_mse','first_reveal'),('coverage95','reveal'),('active_bytes','all')]:
                filt=rr if phase=='all' else [r for r in rr if (r['reveal_index']<0 if phase=='far' else r['reveal_index']==0 if phase=='first_reveal' else r['reveal_index']>=0)]
                if not filt:continue
                vals=[np.mean([r[metric] for r in filt if r['seed']==s]) for s in sorted(set(r['seed'] for r in filt))]
                mean,lo,hi=ci(vals);summary.append(dict(scenario=scenario,method=method,metric=metric,phase=phase,mean=mean,ci_low=lo,ci_high=hi,n_seeds=len(vals)))
            for metric in ['final_correct','final_psnr','recovery_latency','first_correct_latency','registry_mse','untouched_identity_drift','archive_nodes']:
                vals=[np.mean([r[metric] for r in ee if r['seed']==s]) for s in sorted(set(r['seed'] for r in ee))]
                mean,lo,hi=ci(vals);summary.append(dict(scenario=scenario,method=method,metric=metric,phase='episode_final',mean=mean,ci_low=lo,ci_high=hi,n_seeds=len(vals)))
    writecsv(out/'summary.csv',summary)
    return summary


def capacity_experiment(out,seeds):
    rows=[]
    # All unresolved far packets have L=0: no identity-specific texture evidence.
    # Exact MHT may also share these identical distributions. The compact form
    # is an exact implementation, not a superior inference rule.
    for seed in range(seeds):
        packets,mu,v,proto,sd,images=worlds(seed,4,'no_cue')
        rng=np.random.default_rng(seed+95000)
        truth=(images-.5)@basis();trueid=int(rng.integers(K))
        first_cues=proto[trueid]+rng.normal(0,sd,(3,3))
        cue_log=-.5*np.sum((first_cues[:,None,:]-proto[None,:,:])**2,axis=(0,2))/sd**2
        latecue=proto[trueid]+rng.normal(0,sd,3)
        late_log=-.5*np.sum((latecue-proto)**2,axis=1)/sd**2
        y=truth[trueid]+rng.normal(0,SIGMA,D)
        near_ll=gaussian_ll(y,mu,v+SIGMA**2)
        prior_mu,prior_v,prior_lp=init_branches(mu,v)
        for unresolved in (0,16,64,256,1024):
            bankbytes=K*D*2*8
            # Allocate the actual numerical representations; .nbytes excludes
            # Python object overhead equally, so it is not an RSS benchmark.
            fullmu=np.tile(prior_mu,(unresolved,1,1));fullv=np.tile(prior_v,(unresolved,1,1))
            fulllp=np.tile(prior_lp,(unresolved,1));fulluids=np.zeros((unresolved,8,2),np.uint64)
            compact_cues=np.zeros((unresolved,K));compactuids=np.zeros((unresolved,8,2),np.uint64)
            budget_cues=np.zeros((min(unresolved,64),K));budgetuids=np.zeros((min(unresolved,64),8,2),np.uint64)
            conventional=bankbytes+sum(a.nbytes for a in (fullmu,fullv,fulllp,fulluids))
            compact=bankbytes+compact_cues.nbytes+compactuids.nbytes
            budget=bankbytes+budget_cues.nbytes+budgetuids.nbytes
            # With L=0 and no cue, pruning old unresolved metadata changes no
            # future texture posterior. Confirmed entries are always reserved.
            for name,cost in [('exact_mht_materialized',conventional),('shared_statistics',compact),('reserved_bank_budget64',budget)]:
                cue_retained=name!='reserved_bank_budget64' or unresolved<=64
                retained_log=cue_log if cue_retained else np.zeros(K)
                lateid=int(np.argmax(retained_log+late_log))
                nearid=int(np.argmax(retained_log+late_log+near_ll))
                restored=mu[nearid]+v[nearid]/(v[nearid]+SIGMA**2)*(y-mu[nearid])
                rows.append(dict(seed=seed,unresolved_episodes=unresolved,method=name,stored_bytes=cost,confirmed_identities_retained=K,return_theta_mse=float(np.mean((restored-truth[trueid])**2)),cue_only_late_correct=int(lateid==trueid),near_return_correct=int(nearid==trueid),oldest_episode_cue_retained=int(cue_retained)))
    writecsv(out/'capacity.csv',rows)


def archive_return_experiment(out,seeds):
    """A revealed subset selects an archived episode and restores hidden detail."""
    rows=[];maxgap=0.
    for seed in range(seeds):
        _,old,v,proto,sd,images=worlds(seed,4,'no_cue')
        truth=(images-.5)@basis();rng=np.random.default_rng(seed+150000)
        # Identity zero changes 12 coordinates and later returns to its old
        # appearance. Both versions are retained without using true HR states.
        current=old.copy();currentv=v.copy();newy=truth[0,:12]+rng.normal(0,.4,12)+rng.normal(0,SIGMA,12)
        current[0,:12]=newy
        currentv[0,:12]=SIGMA**2
        candidates=np.concatenate((current,old[:1]),axis=0)
        cv=np.concatenate((currentv,v[:1]),axis=0)
        labels=np.array([0,1,2,3,0]);prior=np.log([.125,.25,.25,.25,.125])
        mm=candidates.copy();vv=cv.copy();ll=prior.copy()
        g=np.zeros(D);b=np.zeros(D);c=np.zeros(D);ct=np.zeros(D);cue_log=np.zeros(5)
        latest=current[0].copy();latestv=currentv[0].copy();adaptive=latest.copy();reset=np.zeros(D);resetv=np.full(D,.4**2)
        for reveal in range(4):
            idx=np.arange(reveal*3,(reveal+1)*3)
            y=truth[0,idx]+rng.normal(0,SIGMA,len(idx))
            cue=proto[0]+rng.normal(0,sd,3)
            cue_ll=-.5*np.sum((cue-proto[labels])**2,axis=1)/sd**2
            ll+=cue_ll+gaussian_ll(y,mm[:,idx],vv[:,idx]+SIGMA**2)
            gain=vv[:,idx]/(vv[:,idx]+SIGMA**2);mm[:,idx]+=gain*(y-mm[:,idx]);vv[:,idx]*=1-gain
            ll-=logsumexp(ll)
            g[idx]+=1/SIGMA**2;b[idx]+=y/SIGMA**2;c[idx]+=y*y/SIGMA**2;ct[idx]+=1;cue_log+=cue_ll
            H=1/cv+g;cm=(candidates/cv+b)/H;cvpost=1/H
            cll=prior+cue_log-.5*np.sum(c+candidates*candidates/cv-(candidates/cv+b)**2/H+np.log(cv*H)+ct*np.log(2*np.pi*SIGMA**2),axis=1)
            cll-=logsumexp(cll)
            maxgap=max(maxgap,float(np.max(np.abs(cm-mm))),float(np.max(np.abs(np.exp(cll)-np.exp(ll)))))
            gain=latestv[idx]/(latestv[idx]+SIGMA**2)
            latest[idx]+=gain*(y-latest[idx]);latestv[idx]*=1-gain
            adaptive_gain=(currentv[0,idx]+CHANGE_SD**2)/(currentv[0,idx]+CHANGE_SD**2+SIGMA**2)
            adaptive[idx]+=adaptive_gain*(y-adaptive[idx])
            gain=resetv[idx]/(resetv[idx]+SIGMA**2)
            reset[idx]+=gain*(y-reset[idx]);resetv[idx]*=1-gain
            selected=int(np.argmax(ll));compressed_selected=int(np.argmax(cll))
            for method,z,correct_version in [('single_latest_map',latest,False),('single_adaptive_map',adaptive,False),('hard_reset_partial',reset,False),('exact_version_mht',mm[selected],selected==4),('fork_version_retrieve',cm[compressed_selected],compressed_selected==4)]:
                err=(z-truth[0])**2;mse=float(np.sum(err)/N**2)
                hidden=np.ones(D,bool);hidden[:(reveal+1)*3]=False
                rows.append(dict(seed=seed,reveal_index=reveal,method=method,correct_old_version=int(correct_version),psnr=float(-10*np.log10(max(mse,1e-16))),hidden_mse=float(np.mean(err[hidden])*D/N**2)))
    writecsv(out/'archive_return.csv',rows)
    return maxgap


def plots(rows,summary,out,example):
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,scenario in zip(axs,['no_cue','weak_cue','true_change']):
        for method in METHODS:
            if method=='exact_mht_dedup':continue
            ys=[]
            for rev in [-1,0,1,2,3]:
                sel=[r['correct_branch'] for r in rows if r['scenario']==scenario and r['method']==method and r['reveal_index']==rev]
                ys.append(np.mean(sel))
            ax.plot(range(5),ys,marker='.',label=method)
        ax.set_xticks(range(5),['far','near1','near2','near3','near4']);ax.set_ylim(-.03,1.03);ax.set_title(scenario);ax.set_ylabel('Correct identity branch');ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.15,1,1));fig.savefig(out/'branch_accuracy.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(12,6))
    images=example
    for i in range(K):
        axs[0,i].imshow(images[i].reshape(N,N),cmap='gray',vmin=0,vmax=1,interpolation='nearest');axs[0,i].set_title('Identity '+str(i))
        far=images[i].reshape(4,4,4,4).mean(axis=(1,3))
        axs[1,i].imshow(far,cmap='gray',vmin=0,vmax=1,interpolation='nearest');axs[1,i].set_title('Identical far LR')
        for a in axs[:,i]:a.set_xticks([]);a.set_yticks([])
    fig.tight_layout();fig.savefig(out/'aliased_identities.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for method in METHODS:
        if method=='exact_mht_dedup':continue
        ys=[]
        for rev in range(4):
            vals=[r['coverage95'] for r in rows if r['scenario']=='duplicates' and r['method']==method and r['reveal_index']==rev]
            ys.append(np.mean(vals))
        axs[0].plot(range(4),ys,label=method)
    axs[0].axhline(.95,color='black',ls='--',lw=1);axs[0].set_xlabel('Near reveal block');axs[0].set_ylabel('Empirical 95% interval coverage');axs[0].set_title('Repeated packet audit')
    cap=list(csv.DictReader((out/'capacity.csv').open()))
    for method in dict.fromkeys(r['method'] for r in cap):
        rr=[r for r in cap if r['method']==method and r['seed']=='0']
        axs[1].plot([int(r['unresolved_episodes']) for r in rr],[int(r['stored_bytes'])/1024 for r in rr],marker='.',label=method)
    axs[1].set_xlabel('Unresolved episodes');axs[1].set_ylabel('Persistent numeric storage (KiB)');axs[1].set_yscale('log');axs[1].set_title('Exact representation cost, 4 identities')
    axs[0].legend(fontsize=6);axs[1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'dedup_capacity.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=32);ap.add_argument('--episodes',type=int,default=8);ap.add_argument('--tag',default='main');args=ap.parse_args()
    out=OUT/args.tag;out.mkdir(parents=True,exist_ok=True)
    begin=time.time();rows=[];eps=[];max_equiv=0.;example=None
    for seed in range(args.seeds):
        for scenario in SCENARIOS:
            events,bank,v,prototypes,cue_sd,images=worlds(seed,args.episodes,scenario)
            if example is None:example=images
            engines={m:EpisodeEngine(m,bank,v,np.random.default_rng(seed+50000)) for m in METHODS}
            true_bank=(images-.5)@basis()
            for episode,packets in enumerate(events):
                true_bank[packets[0]['identity']]=packets[0]['truth']
                for e in engines.values():e.reset_episode()
                epmetrics={m:[] for m in METHODS}
                for frame,pkt in enumerate(packets):
                    for method,e in engines.items():
                        info=e.update(pkt,prototypes,cue_sd)
                        err=(e.theta_last-pkt['truth'])**2
                        mse=float(np.sum(err)/N**2)
                        keep=pkt['unchanged'];unchanged_mse=float(np.mean(err[keep])*D/N**2)
                        coverage=float(np.mean(np.abs(e.theta_last-pkt['truth'])<=1.96*np.sqrt(e.var_last)))
                        r=dict(seed=seed,scenario=scenario,episode=episode,frame=frame,packet_uid=pkt['uid'],method=method,reveal_index=pkt['reveal_index'],true_identity=pkt['identity'],selected_identity=e.selected_id,correct_branch=int(e.selected_id==pkt['identity']),psnr=float(-10*np.log10(max(mse,1e-16))),mse=mse,unchanged_mse=unchanged_mse,coverage95=coverage,confidence=info['confidence'],active_bytes=info['active_bytes'],duplicate=int(info['duplicate']),changed=int(pkt['changed']))
                        rows.append(r);epmetrics[method].append(r)
                    exact=engines['exact_mht_dedup'];fork=engines['fork_merge']
                    # Independent online Kalman branches versus one shared batch
                    # sufficient statistic. Matching must hold, including dedup.
                    gap=max(float(np.max(np.abs(exact.posterior_mu-fork.posterior_mu))),float(np.max(np.abs(exact.posterior_v-fork.posterior_v))),float(np.max(np.abs(np.exp(exact.posterior_lp)-np.exp(fork.posterior_lp)))))
                    max_equiv=max(max_equiv,gap)
                for method,e in engines.items():
                    e.finish();rr=epmetrics[method];near=[r for r in rr if r['reveal_index']>=0]
                    # Delay counted in unique near packets. A failed final
                    # recovery receives 4, making failure explicit in mean.
                    successes=[r['reveal_index'] for r in near if r['correct_branch']]
                    first_latency=min(successes) if successes else 4
                    sustained=[j for j in range(4) if near and all(r['correct_branch'] for r in near if r['reveal_index']>=j)]
                    latency=min(sustained) if sustained else 4
                    untouched=np.arange(K)!=pkt['identity']
                    eps.append(dict(seed=seed,scenario=scenario,episode=episode,method=method,final_correct=rr[-1]['correct_branch'],final_psnr=rr[-1]['psnr'],recovery_latency=latency,first_correct_latency=first_latency,registry_mse=float(np.mean((e.bank_mu-true_bank)**2)),untouched_identity_drift=float(np.mean((e.bank_mu[untouched]-e.start_bank_mu[untouched])**2)),archive_nodes=e.archive_nodes,merged=int(e.merged),true_changed=int(pkt['changed'])))
        print(json.dumps({'completed_seed':seed,'elapsed_seconds':round(time.time()-begin,2)}),flush=True)
    assert max_equiv<1e-8,max_equiv
    writecsv(out/'per_packet.csv',rows);writecsv(out/'per_episode.csv',eps)
    summary=summarize(rows,eps,out);capacity_experiment(out,args.seeds)
    archive_gap=archive_return_experiment(out,args.seeds);assert archive_gap<1e-8
    plots(rows,summary,out,example)
    meta=dict(args=vars(args),runtime_seconds=time.time()-begin,command=' '.join(sys.argv),python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),max_exact_MHT_compressed_difference=max_equiv,max_archive_return_MHT_difference=archive_gap,seeds=list(range(args.seeds)),dimensions=dict(identities=K,coefficients=D,hr_pixels=N*N),observation_protocol='4x4 box far LR exactly shared; 2x2 near LR Hadamard sufficient statistics with coefficient noise .03; 3 independent near enrollment observations',limitations=['Synthetic periodic box-filtered textures; no trained generator or real video claim','MHT/fork equivalence is intentional; exact sufficient-statistic compression is not new Bayesian inference','Capacity bytes are measured array .nbytes, not process RSS or GPU latency','Capacity pruning preserves confirmed entries but discarded weak cues can hurt future pre-reveal retrieval'])
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));(out/'source_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps({'done':str(out),'seconds':round(time.time()-begin,2),'max_MHT_gap':max_equiv}),flush=True)


if __name__=='__main__':main()
