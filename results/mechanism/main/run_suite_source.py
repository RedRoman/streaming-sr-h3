#!/usr/bin/env python3
"""Causal, LR-only sufficient-statistic super-resolution mechanism experiments.

This is a controlled inverse problem, not a trained video SR system. A persistent
normal-equation state is mathematically identical to all-history Gaussian MAP.
Only the quarantine/restart policy is an additional mechanism candidate.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
from scipy import ndimage, linalg
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "mechanism"
SCENARIOS = ["complementary", "repeated", "scale_reveal", "revisit", "wrong_prior", "intermittent_mismatch", "persistent_mismatch", "content_change"]
METHODS = ["current_only", "fixed_reference", "ema", "residual_injection", "short_window_map", "all_history_map", "persistent_evidence", "quarantine_restart"]


def operator(n, factor, phase, blur=0.60):
    """Periodic canonical shift + Gaussian PSF + integer decimation, exact matrix."""
    eye = np.eye(n*n).reshape(n, n, n*n)
    blurred = ndimage.gaussian_filter(eye, sigma=(blur, blur, 0), mode="wrap")
    return blurred[phase[0]::factor, phase[1]::factor].reshape(-1, n*n)


def upsample(y, n, factor, phase):
    # Same phase-aware Fourier/Tikhonov inversion is used to construct the prior;
    # no high-resolution image enters initialization or any update.
    side = n // factor
    z = ndimage.zoom(y.reshape(side, side), factor, order=3, mode="wrap")[:n, :n]
    return np.roll(z, phase, axis=(0, 1)).ravel()


def image_truth(seed, n, natural):
    rng = np.random.default_rng(seed)
    if natural:
        source = Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg"
        im = Image.open(source).convert("L").resize((160, 180), Image.Resampling.LANCZOS)
        a = np.asarray(im, float) / 255
        yy = int(rng.integers(0, a.shape[0]-n))
        xx = int(rng.integers(0, a.shape[1]-n))
        z = a[yy:yy+n, xx:xx+n]
        return np.clip(z, 0, 1).ravel(), {"source": str(source), "crop_xy": [xx, yy], "resize_wh": [160,180]}
    yy, xx = np.mgrid[0:n, 0:n] / n
    z = np.zeros((n,n))
    for k in range(10):
        fx, fy = rng.integers(1, n//2, size=2)
        z += rng.uniform(.1, 1) * np.cos(2*np.pi*(fx*xx+fy*yy)+rng.uniform(0,2*np.pi))
    z = .5 + .40*z/(np.max(np.abs(z))+1e-12)
    return z.ravel(), {"source": "analytic_random_fourier", "terms": 10}


def prior_from_lr(y, A, n, factor, phase, seed, wrong=False):
    base = upsample(y,n,factor,phase)
    rng=np.random.default_rng(seed+20000)
    hallucination=rng.normal(0,1,(n,n))
    hallucination-=ndimage.gaussian_filter(hallucination,1,mode="wrap")
    hallucination/=np.std(hallucination)
    # Plausible but ungrounded texture, generated without ground truth.
    p=np.clip(base+(.20 if wrong else .10)*hallucination.ravel(),0,1)
    # Initial reference agrees with first LR evidence. A fixed reference baseline
    # therefore is not artificially penalized for data inconsistency at t=0.
    p+=A.T@linalg.solve(A@A.T+1e-8*np.eye(A.shape[0]), y-A@p, assume_a="pos")
    return p


class Suite:
    def __init__(self,n,steps,noise,lam,window):
        self.n=n;self.steps=steps;self.noise=noise;self.lam=lam;self.window=window
        self.ops={}
        for f in (2,4):
            for y in range(f):
                for x in range(f):
                    k=(f,y,x)
                    A=operator(n,f,(y,x))
                    self.ops[k] = (A,A.T@A)
        self.factors={};self.current_factors={}

    def solve(self,G,b,p):
        return linalg.solve(G+self.lam*np.eye(self.n**2),b+self.lam*p,assume_a="pos",check_finite=False)

    def cached_solve(self, keys,b,p):
        key=tuple(sorted(keys))
        if key not in self.factors:
            G=sum((self.ops[k][1] for k in keys),np.zeros((self.n**2,self.n**2)))
            self.factors[key]=linalg.cho_factor(G+self.lam*np.eye(self.n**2),check_finite=False)
        return linalg.cho_solve(self.factors[key],b+self.lam*p,check_finite=False)

    def timeline(self,scenario):
        seq=[]
        for t in range(self.steps):
            if scenario=="repeated":k=(2,0,0)
            elif scenario=="scale_reveal":
                f=4 if t<self.steps//2 else 2
                j=t%(f*f);k=(f,j//f,j%f)
            else:
                j=t%4;k=(2,j//2,j%2)
            seq.append(k)
        return seq

    def run(self,seed,scenario):
        n=self.n;d=n*n;rng=np.random.default_rng(100000+seed)
        natural=bool(seed%2)
        x, provenance=image_truth(seed,n,natural)
        other,_=image_truth(seed+104729,n,natural)
        seq=self.timeline(scenario)
        G=np.zeros((d,d));b=np.zeros(d);state=None;p=None;ema=None;residual=None
        Gq=np.zeros((d,d));bq=np.zeros(d);pq=None;xq=None;quarantine=[]
        records=[];observations=[];observation_times=[];accepted_keys=[];allkeys=[]
        snapshots={};observed_gram=np.zeros((d,d));max_map_gap=0;max_stats_gap=0;last_current=None
        for t,key in enumerate(seq):
            A,AA=self.ops[key]
            is_revisit_gap=scenario=="revisit" and 4<=t<self.steps-4
            changed=scenario=="content_change" and t>=self.steps//2
            target=other if changed else x
            mismatch=(scenario=="intermittent_mismatch" and t in (6,11)) or (scenario=="persistent_mismatch" and t>=self.steps//2)
            # Incorrect object association: actual LR image comes from a second
            # identity but the user asks to render the original canonical object.
            observed=other if mismatch else target
            y=A@observed+rng.normal(0,self.noise,A.shape[0])
            if p is None:
                p=prior_from_lr(y,A,n,key[0],key[1:],seed,scenario=="wrong_prior")
                ema=p.copy();residual=p.copy();pq=p.copy();xq=p.copy()
            if is_revisit_gap:
                current=last_current.copy()
            else:
                current_p=upsample(y,n,key[0],key[1:])
                current=self.cached_solve([key],A.T@y,current_p)
                last_current=current.copy()
            gate_score=np.nan;action="gap";accepted=False
            if not is_revisit_gap:
                observations.append((key,y.copy()))
                observation_times.append(t)
                allkeys.append(key);G+=AA;b+=A.T@y;observed_gram+=AA
                ema=.8*ema+.2*current
                residual+=.35*self.cached_solve([key],A.T@(y-A@residual),np.zeros(d))
                state=self.solve(G,b,p)
                # Independently reconstruct batch statistics. This audit catches
                # accidental self-generated observations and history double count.
                batchG=sum((self.ops[kk][1] for kk,yy in observations),np.zeros((d,d)))
                batchb=sum((self.ops[kk][0].T@yy for kk,yy in observations),np.zeros(d))
                max_stats_gap=max(max_stats_gap,float(np.max(np.abs(batchG-G))),float(np.max(np.abs(batchb-b))))
                batch=self.cached_solve(allkeys,batchb,p)
                max_map_gap=max(max_map_gap,float(np.max(np.abs(batch-state))))
                # Posterior predictive residual with LR-only uncertainty. This is
                # a useful test, not an oracle association label.
                H=Gq+self.lam*np.eye(d)
                V=linalg.solve(H,A.T,assume_a="pos",check_finite=False)
                var=self.noise**2*(1+np.sum(A*V.T,axis=1))
                gate_score=float(np.mean((y-A@xq)**2/var))
                if gate_score<=4.0:
                    Gq+=AA;bq+=A.T@y;accepted_keys.append(key)
                    quarantine=[];action="accept";accepted=True
                else:
                    quarantine.append((key,y.copy()))
                    action="quarantine"
                    if len(quarantine)>=3:
                        # Consecutive contradiction is treated as a new epoch.
                        # This can recover content changes but cannot identify a
                        # persistently wrong identity from LR evidence alone.
                        Gq=sum((self.ops[kk][1] for kk,yy in quarantine),np.zeros((d,d)))
                        bq=sum((self.ops[kk][0].T@yy for kk,yy in quarantine),np.zeros(d))
                        kk,yy=quarantine[0]
                        pq=upsample(yy,n,kk[0],kk[1:])
                        accepted_keys=[kk for kk,yy in quarantine]
                        quarantine=[];action="restart";accepted=True
                if accepted:xq=self.solve(Gq,bq,pq)
            else:
                batch=state.copy()
            # A short temporal window expires across occlusion even if no new
            # observations arrive. Persistent state keeps canonical evidence.
            recent=[obs for obs,tt in zip(observations,observation_times) if tt>t-self.window]
            shortb=sum((self.ops[kk][0].T@yy for kk,yy in recent),np.zeros(d))
            short=self.cached_solve([kk for kk,yy in recent],shortb,p)
            outputs={"current_only":current,"fixed_reference":p,"ema":ema,"residual_injection":residual,"short_window_map":short,"all_history_map":batch,"persistent_evidence":state,"quarantine_restart":xq}
            for method,z in outputs.items():
                mse=float(np.mean((z-target)**2));psnr=-10*np.log10(max(mse,1e-16))
                zhp=z.reshape(n,n)-ndimage.gaussian_filter(z.reshape(n,n),1,mode="wrap")
                thp=target.reshape(n,n)-ndimage.gaussian_filter(target.reshape(n,n),1,mode="wrap")
                records.append(dict(seed=seed,data="natural" if natural else "analytic",scenario=scenario,frame=t,method=method,mse=mse,psnr=float(psnr),hf_mse=float(np.mean((zhp-thp)**2)),lr_residual=float(np.mean((A@z-y)**2)),gate_score=gate_score,gate_action=action,observation_count=len(observations),mismatch=int(mismatch),changed=int(changed),gap=int(is_revisit_gap)))
            if t in (0,self.steps//2-1,self.steps-1):
                snapshots[str(t)]={"target":target.reshape(n,n),"prior":p.reshape(n,n),**{k:v.reshape(n,n).copy() for k,v in outputs.items()}}
        eigen=linalg.eigvalsh(observed_gram,check_finite=False)
        rank=int(np.sum(eigen>max(eigen.max()*1e-9,1e-12)))
        firstA=self.ops[seq[0]][0]
        P=np.eye(d)-linalg.pinv(firstA)@firstA
        initial_null=float(np.linalg.norm(P@(p-x))**2/d)
        final_null=float(np.linalg.norm(P@(state-x))**2/d)
        audit=dict(seed=seed,scenario=scenario,data="natural" if natural else "analytic",rank=rank,dimensions=d,max_map_gap=max_map_gap,max_stats_gap=max_stats_gap,initial_first_observation_null_mse=initial_null,final_first_observation_null_mse=final_null,provenance=provenance)
        return records,audit,snapshots


def write_csv(path,rows):
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def boot_mean_ci(a,seed=20261015):
    a=np.asarray(a);rng=np.random.default_rng(seed)
    means=a[rng.integers(0,len(a),(5000,len(a)))].mean(axis=1)
    return float(a.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))


def summarize(rows,steps):
    summaries=[];contrasts=[]
    scenarios=list(dict.fromkeys(r["scenario"] for r in rows))
    for scenario in scenarios:
        for phase in ("final","post_event","all_frames"):
            def filt(r):return r["scenario"]==scenario and (phase=="all_frames" or (r["frame"]==steps-1 if phase=="final" else r["frame"]>=steps//2))
            for method in METHODS:
                chosen=[r for r in rows if filt(r) and r["method"]==method]
                seeds=sorted(set(r["seed"] for r in chosen))
                psnrs=[np.mean([r["psnr"] for r in chosen if r["seed"]==seed]) for seed in seeds]
                mean,lo,hi=boot_mean_ci(psnrs)
                summaries.append(dict(scenario=scenario,phase=phase,method=method,n_seeds=len(seeds),mean_psnr=mean,ci_low=lo,ci_high=hi,mean_mse=float(np.mean([r["mse"] for r in chosen]))))
            for method in METHODS:
                if method=="persistent_evidence":continue
                deltas=[]
                for seed in sorted(set(r["seed"] for r in rows)):
                    a=[r["psnr"] for r in rows if filt(r) and r["seed"]==seed and r["method"]=="persistent_evidence"]
                    b=[r["psnr"] for r in rows if filt(r) and r["seed"]==seed and r["method"]==method]
                    if a and b:deltas.append(np.mean(a)-np.mean(b))
                mean,lo,hi=boot_mean_ci(deltas)
                contrasts.append(dict(scenario=scenario,phase=phase,contrast="persistent_evidence-minus-"+method,mean_delta_db=mean,ci_low=lo,ci_high=hi,n_seeds=len(deltas)))
    return summaries,contrasts


def make_plots(rows,snapshots,out,steps):
    fig,axs=plt.subplots(2,4,figsize=(17,8),sharex=True)
    for ax,scenario in zip(axs.ravel(),SCENARIOS):
        for method in METHODS:
            if method=="all_history_map":continue
            vals=[]
            for t in range(steps):vals.append(np.mean([r["psnr"] for r in rows if r["scenario"]==scenario and r["method"]==method and r["frame"]==t]))
            ax.plot(vals,label=method,linewidth=1.5)
        ax.set_title(scenario);ax.set_xlabel("Causal frame index");ax.set_ylabel("PSNR (dB)");ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4,fontsize=9)
    fig.tight_layout(rect=(0,.10,1,1));fig.savefig(out/"learning_curves.png",dpi=160);plt.close(fig)
    methods=["target","fixed_reference","current_only","short_window_map","persistent_evidence","quarantine_restart"]
    fig,axs=plt.subplots(len(snapshots),len(methods),figsize=(12,2.1*len(snapshots)))
    for i,(label,ss) in enumerate(snapshots.items()):
        final=ss[str(steps-1)]
        for j,method in enumerate(methods):
            axs[i,j].imshow(final[method],vmin=0,vmax=1,cmap="gray",interpolation="nearest")
            axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
            if i==0:axs[i,j].set_title(method,fontsize=9)
            if j==0:axs[i,j].set_ylabel(label,fontsize=8)
    fig.tight_layout();fig.savefig(out/"qualitative_panels.png",dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--seeds",type=int,default=16);ap.add_argument("--n",type=int,default=20);ap.add_argument("--steps",type=int,default=16);ap.add_argument("--noise",type=float,default=.01);ap.add_argument("--lam",type=float,default=.0025);ap.add_argument("--window",type=int,default=4);ap.add_argument("--tag",default="main")
    args=ap.parse_args();assert args.n%4==0
    out=OUT/args.tag;out.mkdir(parents=True,exist_ok=True)
    begin=time.time();suite=Suite(args.n,args.steps,args.noise,args.lam,args.window)
    rows=[];audits=[];snapshot_examples={}
    for seed in range(args.seeds):
        for scenario in SCENARIOS:
            rec,audit,snap=suite.run(seed,scenario);rows.extend(rec);audits.append(audit)
            if seed in (0,1) and scenario in ("complementary","content_change","intermittent_mismatch"):
                snapshot_examples[("natural" if seed else "analytic")+" / "+scenario]=snap
        print(json.dumps({"completed_seed":seed,"seeds":args.seeds,"elapsed_seconds":round(time.time()-begin,2)}),flush=True)
    summaries,contrasts=summarize(rows,args.steps)
    write_csv(out/"per_frame.csv",rows);write_csv(out/"summary.csv",summaries);write_csv(out/"paired_contrasts.csv",contrasts)
    (out/"audits.json").write_text(json.dumps(audits,indent=2))
    assert max(a["max_map_gap"] for a in audits)<1e-8
    assert max(a["max_stats_gap"] for a in audits)<1e-10
    make_plots(rows,snapshot_examples,out,args.steps)
    meta={"args":vars(args),"command":" ".join(sys.argv),"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":__import__("scipy").__version__,"matplotlib":matplotlib.__version__,"pillow":__import__("PIL").__version__,"runtime_seconds":time.time()-begin,"seed_list":list(range(args.seeds)),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"data_provenance":"even seeds analytic Fourier textures; odd seeds deterministic patches of Matplotlib bundled Grace Hopper photograph", "strict_causality":"time-t methods access observations at <=t only; HR truth used only in metric computation and observation simulator", "limitations":["20x20 grayscale controlled patches, known periodic shift/blur/downsampling; not real video evaluation", "photograph patches share a single source image; bootstrap over seeds does not establish population generalization", "no network training, learned priors, optical flow estimation, or latency benchmark", "persistent evidence and all-history MAP intentionally identical", "quarantine rule thresholds prespecified; must be validated on separate scenes"]}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps({"done":str(out),"runtime_seconds":round(time.time()-begin,2),"max_MAP_gap":max(a["max_map_gap"] for a in audits)}),flush=True)


if __name__=="__main__":main()
