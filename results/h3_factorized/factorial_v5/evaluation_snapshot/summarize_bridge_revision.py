"""Source-level fixed endpoint B5 statistics, no model/source selection."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ARMS=['anchor_skip0','noanchor_skip0','anchor_skip1','noanchor_skip1']
def write(path,rows):
    if rows:
        with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--B4',required=True);a=p.parse_args();root=Path(a.root);out=root/'summary_seed17';out.mkdir(exist_ok=True);rows=list(csv.DictReader((root/'val_seed17/rgb_metrics.csv').open()));b4=list(csv.DictReader((Path(a.B4)/'val_seed17/rgb_metrics.csv').open()));sources=sorted({r['source'] for r in rows});assert len(sources)==8;methods=sorted({r['method'] for r in rows});rng=np.random.default_rng(20261015);ii=rng.integers(0,8,(10000,8));aggregate=[];summary=[];pair=[];factors=[]
    conditions=['all','neutral_return','appearance_cue','true_content_change'];metrics=['union_psnr','old_unmodified_psnr','new_detail_psnr','changed_psnr','overlap_psnr','rgb_psnr'];values={}
    for family in ['raw','DC']:
        for condition in conditions:
            for metric in metrics:
                for method in methods:
                    v=[]
                    for source in sources:
                        x=[float(r[metric]) for r in rows if r['family']==family and r['method']==method and r['source']==source and (condition=='all' or r['condition']==condition) and r.get(metric) not in ['',None]]
                        if not x:break
                        mean=float(np.mean(x));v.append(mean);aggregate.append(dict(family=family,condition=condition,metric=metric,method=method,source=source,mean=mean))
                    if len(v)!=8:continue
                    v=np.array(v);values[family,condition,metric,method]=v;ci=np.quantile(v[ii].mean(1),[.025,.975]);summary.append(dict(family=family,condition=condition,metric=metric,method=method,n_sources=8,mean=float(v.mean()),CI_low=float(ci[0]),CI_high=float(ci[1])))
                for arm in ARMS:
                    key=(family,condition,metric,arm)
                    if key not in values:continue
                    basemethod='H3_preconditioned_decoded_atlas' if arm.endswith('1') else 'current_LR_VAE'
                    comparators=[basemethod,'LR_photo_revision_atlas',arm+'__NN_delta0',arm+'__NN_shuffled_history',arm+'__NN_frozen_photo',arm+'__NN_frozen_weights']
                    if arm.endswith('1'):comparators += [arm+'__bridge_removed',arm+'__bridge_shuffled']
                    for comparator in comparators:
                        if (family,condition,metric,comparator) not in values:continue
                        d=values[key]-values[family,condition,metric,comparator];ci=np.quantile(d[ii].mean(1),[.025,.975]);pair.append(dict(family=family,condition=condition,metric=metric,method=arm,comparator=comparator,source_mean_difference=float(d.mean()),CI_low=float(ci[0]),CI_high=float(ci[1]),source_wins=int((d>0).sum()),n_sources=8))
                    for comparator in ['closed_action','anchor_only','factor_cache_mlp','gru_replay']:
                        other=[]
                        for source in sources:
                            x=[float(r[metric]) for r in b4 if r['family']==family and r['method']==comparator and r['source']==source and (condition=='all' or r['condition']==condition) and r.get(metric) not in ['',None]]
                            if not x:break
                            other.append(np.mean(x))
                        if len(other)!=8:continue
                        d=values[key]-np.array(other);ci=np.quantile(d[ii].mean(1),[.025,.975]);pair.append(dict(family=family,condition=condition,metric=metric,method=arm,comparator='B4_'+comparator,source_mean_difference=float(d.mean()),CI_low=float(ci[0]),CI_high=float(ci[1]),source_wins=int((d>0).sum()),n_sources=8))
                if all((family,condition,metric,m) in values for m in ARMS):
                    a1s0,a0s0,a1s1,a0s1=[values[family,condition,metric,m] for m in ARMS];effects=dict(bridge_average_effect=(a1s1+a0s1-a1s0-a0s0)/2,current_native_conditioning_average_effect=(a1s1+a1s0-a0s1-a0s0)/2,interaction=a1s1-a1s0-a0s1+a0s0)
                    for name,v in effects.items():ci=np.quantile(v[ii].mean(1),[.025,.975]);factors.append(dict(family=family,condition=condition,metric=metric,effect=name,source_mean_difference=float(v.mean()),CI_low=float(ci[0]),CI_high=float(ci[1]),positive_sources=int((v>0).sum())))
    write(out/'source_aggregates.csv',aggregate);write(out/'method_summary.csv',summary);write(out/'paired_differences.csv',pair);write(out/'factorial_effects.csv',factors)
    selected=[m for m in methods if m in ARMS or m in ['current_LR_VAE','H3_preconditioned_decoded_atlas','LR_photo_revision_atlas','LR_photo_revision_inverse']];selected=sorted(selected,key=lambda m:values['DC','all','union_psnr',m].mean());fig,ax=plt.subplots(figsize=(9,5));means=[values['DC','all','union_psnr',m].mean() for m in selected];ax.barh(selected,means);ax.set_xlim(22,29);ax.set_xlabel('DC union PSNR (source-averaged dB, 8 development sources)');fig.tight_layout();fig.savefig(out/'endpoint_quality.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,suffix in zip(axs,['__NN_delta0','__NN_shuffled_history']):
        for i,m in enumerate(ARMS):
            d=values['DC','all','union_psnr',m]-values['DC','all','union_psnr',m+suffix];ax.scatter(np.full(8,i),d,s=25);ax.plot([i-.15,i+.15],[d.mean()]*2,color='black')
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(4),ARMS,rotation=30,ha='right');ax.set_ylabel('Actual minus NN intervention (dB)');ax.set_title(suffix.replace('__NN_','')+'; physical base fixed')
    fig.tight_layout();fig.savefig(out/'neural_history_dependence.png',dpi=160);plt.close(fig)
    # Query15 is a separate no-current-observation metric, no DC.
    nr=list(csv.DictReader((root/'val_seed17/no_observation_query.csv').open()));nsummary=[]
    for method in sorted({r['method'] for r in nr}):
        v=np.array([float(next(r for r in nr if r['source']==s and r['method']==method)['union_PSNR']) for s in sources]);ci=np.quantile(v[ii].mean(1),[.025,.975]);nsummary.append(dict(method=method,mean=float(v.mean()),CI_low=float(ci[0]),CI_high=float(ci[1])))
    (out/'statistics.json').write_text(json.dumps(dict(sources=sources,source_count=8,trained_seeds=[17],bootstrap=10000,source_then_query='Average queries within source, then paired source bootstrap. Seeds/queries are not independent videos.',primary='DC all-query union; fixedlast672',main_results=[r for r in summary if r['family']=='DC' and r['condition']=='all' and r['metric']=='union_psnr' and r['method'] in selected],paired_primary=[r for r in pair if r['family']=='DC' and r['condition']=='all' and r['metric']=='union_psnr'],factorial_primary=[r for r in factors if r['family']=='DC' and r['condition']=='all' and r['metric']=='union_psnr'],no_observation_query15=nsummary),indent=2))
    print(json.dumps(dict(primary=json.loads((out/'statistics.json').read_text())['main_results'],paired=json.loads((out/'statistics.json').read_text())['paired_primary'],factorial=json.loads((out/'statistics.json').read_text())['factorial_primary']),indent=2))


if __name__=='__main__':main()
