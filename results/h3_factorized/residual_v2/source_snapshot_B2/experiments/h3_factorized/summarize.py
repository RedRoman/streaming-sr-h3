"""Source-paired exploratory intervals and publication-style static plots."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);args=p.parse_args();out=Path(args.results)
    rows=list(csv.DictReader((out/'rgb_metrics.csv').open()));groups=defaultdict(list);by=defaultdict(dict)
    for r in rows:
        key=(r['method'],r['scenario']);groups[key].append(r)
        by[(r['method'],r['scenario'])][(int(r['seed']),r['source'])]=float(r['rgb_psnr'])
    summary=[]
    for (method,scenario),rr in groups.items():
        d=dict(method=method,scenario=scenario,n_episode_seed=len(rr))
        for metric in ['rgb_psnr','roi_psnr','outside_psnr']:d[metric+'_mean']=float(np.mean([float(x[metric]) for x in rr]))
        summary.append(d)
    with (out/'scenario_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    methods=sorted(set(r['method'] for r in rows));seeds=sorted(set(int(r['seed']) for r in rows if int(r['seed'])>=0));sources=sorted(set(r['source'] for r in rows));scenarios=list(dict.fromkeys(r['scenario'] for r in rows));rng=np.random.default_rng(82017);effects=[]
    for scenario in scenarios+['ALL_five_stages']:
        for control in [m for m in methods if m!='factorized_swap']:
            matrix=np.zeros((len(seeds),len(sources)))
            for i,seed in enumerate(seeds):
                for j,source in enumerate(sources):
                    ss=scenarios if scenario=='ALL_five_stages' else [scenario]
                    matrix[i,j]=np.mean([by[('factorized_swap',s)][(seed,source)]-by[(control,s)][((seed if not control.startswith(('H3_','input_')) else -1),source)] for s in ss])
            estimates=[]
            for _ in range(4000):
                ii=rng.integers(len(seeds),size=len(seeds));starts=rng.integers(len(sources),size=(len(sources)+1)//2);jj=((starts[:,None]+np.arange(2))%len(sources)).ravel()[:len(sources)];estimates.append(matrix[ii][:,jj].mean())
            lo,hi=np.percentile(estimates,[2.5,97.5]);effects.append(dict(scenario=scenario,comparison='factorized_swap minus '+control,mean_dB=float(matrix.mean()),CI95_low=float(lo),CI95_high=float(hi),per_seed_mean_dB={str(s):float(matrix[i].mean()) for i,s in enumerate(seeds)}))
    payload=dict(bootstrap='4000 paired nested seed + circular source blocks(length2) draws; adjacent donor reuse remains a limitation; three seeds are exploratory, not a population guarantee',seeds=seeds,sources=len(sources),effects=effects)
    (out/'paired_effects.json').write_text(json.dumps(payload,indent=2))
    plotted=['input_bicubic','H3_current_LR','factorized_swap','factorized_no_swap','noninvertible_swap','mixed_memory']+[m for m in ['fixed_B1_retained_detail','fixed_B1_direct_near','B2_direct_observed_detail'] if m in methods]
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,metric,title in zip(axes,['rgb_psnr','roi_psnr','outside_psnr'],['Whole image RGB','Central reveal/change ROI','Unchanged outside ROI']):
        for method in plotted:
            ys=[next(r[metric+'_mean'] for r in summary if r['method']==method and r['scenario']==s) for s in scenarios];ax.plot(range(5),ys,marker='o',label=method,linewidth=1.3)
        ax.set_xticks(range(5),['light A','reveal A','return A','change B','light B'],rotation=25);ax.set_ylabel('Mean per-episode PSNR(dB)');ax.set_title(title);ax.grid(alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,0.01),ncol=4,fontsize=8);fig.savefig(out/'actual_H3_RGB_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # Source-paired full/control effects at each event.
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for offset,control in zip([-.2,0,.2],['factorized_no_swap','noninvertible_swap','mixed_memory']):
        e=[next(e for e in effects if e['scenario']==s and e['comparison'].endswith(control)) for s in scenarios]
        y=np.array([x['mean_dB'] for x in e]);lo=np.array([x['CI95_low'] for x in e]);hi=np.array([x['CI95_high'] for x in e]);ax.errorbar(np.arange(5)+offset,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt='o',capsize=3,label='minus '+control)
    ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(5),['light A','reveal A','return A','change B','light B']);ax.set_ylabel('Factorized+swap difference(dB),95% interval');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(out/'paired_control_effects.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(summary=summary,effects=effects),indent=2))


if __name__=='__main__':main()
