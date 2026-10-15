"""Source-paired B3R statistics; one seed and eight development sources."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_csv(path,rows):
    if rows:
        with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--evaluation',default='val_seed17');p.add_argument('--runtime',default='runtime_seed17');p.add_argument('--summary',default='summary_seed17');args=p.parse_args();root=Path(args.root);evaluation=root/args.evaluation;out=root/args.summary;out.mkdir(exist_ok=True);rows=list(csv.DictReader((evaluation/'rgb_metrics.csv').open()));methods=sorted({r['method'] for r in rows});sources=sorted({r['source'] for r in rows});assert len(sources)==8
    regions=['rgb','union','old_unmodified','new_detail','changed','overlap','transient_foreground'];conditions=['all','appearance_cue','neutral_return','true_content_change'];aggregates=[];paired=[];source_rows=[];rng=np.random.default_rng(20261015);indices=rng.integers(0,len(sources),size=(10000,len(sources)))
    for family in ['raw','DC']:
        for condition in conditions:
            for metric in [r+'_psnr' for r in regions]:
                values={}
                for method in methods:
                    source_values=[]
                    for source in sources:
                        numbers=[float(r[metric]) for r in rows if r['family']==family and r['source']==source and r['method']==method and (condition=='all' or r['condition']==condition) and r[metric]!='']
                        if numbers:source_values.append(float(np.mean(numbers)));source_rows.append(dict(family=family,condition=condition,metric=metric,source=source,method=method,value=float(np.mean(numbers))))
                    if len(source_values)!=8:continue
                    v=np.array(source_values);values[method]=v;ci=np.quantile(v[indices].mean(1),[.025,.975]);aggregates.append(dict(family=family,condition=condition,metric=metric,method=method,n_sources=8,mean=float(v.mean()),CI_low=float(ci[0]),CI_high=float(ci[1])))
                if 'closed_action' not in values:continue
                for method,v in values.items():
                    if method=='closed_action':continue
                    d=values['closed_action']-v;ci=np.quantile(d[indices].mean(1),[.025,.975]);paired.append(dict(family=family,condition=condition,metric=metric,reference='closed_action',control=method,n_sources=8,mean_difference=float(d.mean()),CI_low=float(ci[0]),CI_high=float(ci[1]),source_wins=int((d>0).sum()),source_ties=int((d==0).sum()),per_source_difference=json.dumps(dict(zip(sources,d.tolist())))))
    write_csv(out/'source_aggregates.csv',source_rows);write_csv(out/'method_summary.csv',aggregates);write_csv(out/'paired_differences.csv',paired)
    steps=sorted({int(r['query_step']) for r in rows});selected=['current_LR_VAE','closed_action','closed_action_no_equiv','factor_cache_mlp','nonlinear_group','gru_online','gru_replay','LR_photo_revision_atlas','LR_photo_revision_inverse','H3_ED_revision_inverse'];fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,metric in zip(axes,['union_psnr','changed_psnr']):
        for method in selected:
            vals=[]
            for step in steps:
                numbers=[float(r[metric]) for r in rows if r['family']=='DC' and r['method']==method and int(r['query_step'])==step and r[metric]!=''];vals.append(np.mean(numbers) if numbers else np.nan)
            ax.plot(steps,vals,marker='o',ms=3,label=method)
        ax.set(xlabel='Causal query step',ylabel='Source mean PSNR (dB)',title=metric.replace('_',' '));ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,bbox_to_anchor=(1.03,1),loc='upper left');fig.tight_layout();fig.savefig(out/'quality_by_query.png',dpi=160);plt.close(fig)
    costs=list(csv.DictReader((root/args.runtime/'state_cost_and_revision_runtime.csv').open())) if (root/args.runtime/'state_cost_and_revision_runtime.csv').exists() else []
    if costs:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for method in ['closed_action','factor_cache_mlp','nonlinear_group','gru_online','gru_replay']:
            rr=[r for r in costs if r['kind']==method];nn=[int(r['N_packets']) for r in rr];axes[0].loglog(nn,[float(r['actual_whole_state_revision_seconds'])*1000 for r in rr],marker='o',label=method);axes[1].plot(nn,[(float(r['persistent_compressed_runtime_tensor_and_ID_bytes'])+float(r['controller_reference_descriptor_group_bytes'])+float(r['query_current_LR_anchor_decoded_scratch_bytes']))/1024**2 for r in rr],marker='o',label=method)
        axes[0].set(xlabel='Packet count (repeated-feature compute stress)',ylabel='Whole state revision, ms');axes[1].set(xlabel='Packet count',ylabel='State + controller + current query buffers, MiB');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'revision_cost.png',dpi=160);plt.close(fig)
    key=lambda condition,control,metric='union_psnr':next(r for r in paired if r['family']=='DC' and r['condition']==condition and r['metric']==metric and r['control']==control)
    metadata=json.loads((evaluation/'complete.json').read_text());checkpoint=metadata.get('checkpoint','best');lines=['# B3R fixed seed17 development pilot','', '32 train / 8 val natural source photos; synthetic known camera, nuisance-group tags, photometric changes, transient foreground and true local replacement. These BSDS sources were used earlier in the project. No old16test or sealed DIV2K evaluation. One seed; source, not query/packet/epoch, is the bootstrap unit.','', 'All six controls were trained for600 latent+200 actual-native-H3-D steps. The original protocol specified val-selected best checkpoints; parent subsequently required a separate fixed-endpoint comparison, reinitializing all six RGB runs from last600 and evaluating last200. Original selection-pilot and extra physical compute are retained separately. Affine actions, sum statistics, factor caching and replay are standard ingredients; this remains a candidate mechanism experiment.',f'\nThis report evaluates **{checkpoint}.pt**, from `{args.evaluation}`. Exact checkpoint step/hash/initial path: evaluation `complete.json`.','', '|Condition/region|Control|full minus control dB|95% source bootstrap CI|Wins/8|','|---|---|---:|---|---:|']
    for condition in ['all','neutral_return','appearance_cue','true_content_change']:
        for control in ['current_LR_VAE','closed_action_no_equiv','factor_cache_mlp','nonlinear_group','gru_online','gru_replay','LR_photo_revision_atlas','LR_photo_revision_inverse','H3_ED_revision_inverse']:
            r=key(condition,control);lines.append(f"|{condition}/union|{control}|{r['mean_difference']:+.4f}|[{r['CI_low']:+.4f}, {r['CI_high']:+.4f}]|{r['source_wins']}|")
    lines+=['','Fixed-checkpoint mechanism interventions (no retraining):','', '|Condition|Intervention|full minus intervention union dB|CI|Wins/8|','|---|---|---:|---|---:|']
    for condition in ['all','appearance_cue','neutral_return','true_content_change']:
        for control in ['closed_action_frozen_photo','closed_action_frozen_weights','closed_action_frozen_both','diagnostic_closed_no_measured_detail','diagnostic_closed_shuffled_history']:
            r=key(condition,control);lines.append(f"|{condition}|{control}|{r['mean_difference']:+.4f}|[{r['CI_low']:+.4f}, {r['CI_high']:+.4f}]|{r['source_wins']}|")
    lines+=['','Raw/DC, per-source and regional data are retained in `method_summary.csv`, `source_aggregates.csv` and `paired_differences.csv`. The `changed` region is evaluated only after actual replacement queries13/17; a pre-confirmation branch claim is not made.','', 'The independent streaming implementation stores compressed h rather than native delta/proxy codes. It is audited against the vectorized rollout; all raw factor h/weight/ID/descriptor and group buffers are included. Appearance-action-only cost may be constant in N for fixed group count, while the actual whole-state update still scans packet IDs/weights, and LR reliability/controller work can scale with N. No O(1) whole pipeline claim.','', 'Shared current LR49152B + native anchor172032B + decoded RGB786432B =1,007,616B of current-query payload, reference LR49152B additionally persistent, and shared camera524288B. Frozen H3 weights are shared and separately reported. Compressed factor h is28672B per packet plus weights1024B,8x8RGB descriptor768B and assumed24B ID/group/geometry. The raw-LR inverse archive is49152B per packet plus controller/metadata; this is a stronger storage comparison than contrasting4 channels with a dense RGB atlas. Python allocator overhead, activations and CUDA peak are separate.','', 'The classical inverse models relative spatial gain+bias before the exact observation operator; observed RGB clipping makes this affine model approximate. Native decoded-atlas photometric normalization is an empirical control, not exact latent equivariance. All methods use the same causal confidence with explicitly different resolution resampling and the same64step current-LR DC. Generated outputs never enter the measurement ledger.','', 'Interpretation must be filled from the fixed results, including whether full beats the same-compression factor-cache/replay-GRU and preserves newly acquired detail beyond current-anchor priors. Algebraic closure alone does not establish originality.']
    (out/'B3R_seed17_report.md').write_text('\n'.join(lines)+'\n');(out/'statistics.json').write_text(json.dumps(dict(n_sources=8,n_train_sources=32,n_trained_seeds=1,seed=17,bootstrap=10000,bootstrap_unit='source; queries averaged before sampling',source_ids=sources,all_methods=methods,paired=paired),indent=2));print(out/'B3R_seed17_report.md')


if __name__=='__main__':main()
