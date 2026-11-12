"""One TRAIN-only objective intervention; no head/teacher training or DEV read.

Same eleven raw features, five source-group folds, fold-training standardization,
ridge .001 and 200 fixed GD steps. Only the objective changes to class-balanced
Huber regression of the unchanged normalized utility. Predictions are not calibrated probabilities.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

LEGACY_SHA='b8e087522775c43f6f2dea5f4a047ffcbed2046417a7097d9ca1a2c9ab2e7c59'
REPORT_SHA='0508eaa406ce8053e9fbb0fb4f477e5aeaab4dd83d6bd52e8a5cb8c4ec5bfd87'
SPLIT_SHA='2b1b1d627b8b8cfd2169385139466026e3d0db938ad43200c35c1504f1d82ae4'
CONTRACT_SHA='09fa7efa488c2cc8760d1edd264befe5ce1031b73774e529a4a57b4f3d46a32e'
SCHEMA='rphi.input_preservation_balanced_huber_probe_execution.v1'
PROTOCOL={
 'single_intervention':'Only replace equal tile weights by fold-training class-balanced positive/negative tile weights; normalized targets, beta1 Huber loss, model/features/split/std/ridge/steps/threshold unchanged.',
 'features':'Same frozen FP64 arithmetic mean of two raw-feature slots;11 tile-local channels plus intercept; no source/profile input.',
 'folds':'Exact accepted19TRAIN source_group split, seed20261112, five folds. No DEV files.',
 'labels':'Keep the original FP32 normalized benefit converted to FP64 without target change. Positive/negative signs define weights only. The accepted cache contains no zero labels; any zero encountered aborts rather than dropping/relabeling rows.',
 'weights':'Each positive training tile weight1/(2*n_positive); each negative1/(2*n_negative); total weight1. Counts from fold-training only; missing class aborts.',
 'objective':'sum_i weight_i*SmoothL1_beta1(score_i-original_normalized_y_i)+.5*.001*sum(slopes**2); intercept unregularized.',
 'solver':'Zero coefficients; exactly200 full-batch GD steps with1/L, L=lambda_max(Z.T@diag(weights)@Z+ridge_diagonal). No line search/early stop.',
 'threshold':0,'tie':'preserve raw','ridge':.001,'iterations':200,'std_zero_threshold':1e-12,
 'baseline':'Replay accepted Huber coefficients/constants on the exact same held-out standardized features; require both complete score array SHA match before any new fit. Zero extra unweighted Huber fits. Weighted constant uses the same60-bisection interval search and closest-to-zero choice.',
 'metrics':'Per-source sign balanced accuracy, negative-label falseK1, positive-label recall, coverage, exact-tie half-credit AUROC; source macro primary with valid denominators, source/profile and tile micro secondary. No pooled cross-fold AUC used for a scientific conclusion.',
 'decision':'Descriptive diagnostic only: report old ranking vs fixed-zero action, and new balanced sign tradeoff. No deployment/threshold/teacher/head-training admission. After this single experiment pause preservation-head iterations; no automatic followup regardless of outcome.',
 'interpretation':'AUC>0.5 with all-positive Huber scores is evidence of ranking hidden by its positive offset. Better balanced sign decisions after this weights-only intervention support a role for class weighting in the failed decisions, not proof of raw MAE improvement. Failure is limited to this tile-local linear family and fixed200 steps.'}


def legacy():
    import hashlib
    p=Path(__file__).with_name('legacy_probe.py')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=LEGACY_SHA:raise ValueError('frozen reader changed')
    spec=importlib.util.spec_from_file_location('frozen_huber_probe',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def huber(z):
    import numpy as np
    a=np.abs(z);return np.where(a<=1,.5*z*z,a-.5)


def balanced_problem(z,y):
    import numpy as np
    if z.ndim!=2 or y.ndim!=1 or len(z)!=len(y) or not np.isfinite(z).all() or not np.isfinite(y).all():raise ValueError('nonfinite/invalid design')
    if (y==0).any():raise ValueError('accepted fixed cache has no zero labels; no silent row removal')
    pos=int((y>0).sum());neg=len(y)-pos
    if not pos or not neg:raise ValueError('both training sign classes required; no resampling')
    weight=np.where(y>0,1/(2*pos),1/(2*neg))
    return z,y,weight,{'positive_tiles':pos,'negative_tiles':neg,'zero_tiles':0,
      'positive_tile_weight':1/(2*pos),'negative_tile_weight':1/(2*neg),
      'positive_total_weight':float(weight[y>0].sum()),'negative_total_weight':float(weight[y<0].sum())}


def weighted_constant(y):
    """Same60-step endpoint search; signs affect weights, never the target."""
    import numpy as np
    if y.ndim!=1 or not np.isfinite(y).all() or np.abs(y).max()>4 or (y==0).any():raise ValueError('invalid fixed target')
    groups=[np.sort(y[y>0]),np.sort(y[y<0])]
    if any(len(v)==0 for v in groups):raise ValueError('both sign classes required')
    prefixes=[np.r_[0.,np.cumsum(v,dtype=np.float64)] for v in groups]
    def gradient(c):
        means=[]
        for v,prefix in zip(groups,prefixes):
            left=int(np.searchsorted(v,c-1,'left'));right=int(np.searchsorted(v,c+1,'right'))
            means.append((left-(len(v)-right)+(right-left)*c-(prefix[right]-prefix[left]))/len(v))
        return .5*(means[0]+means[1])
    ends=[]
    for upper in (False,True):
        lo,hi=-4.,4.
        for _ in range(60):
            mid=(lo+hi)/2;g=gradient(mid)
            if g<0 or (upper and g==0):lo=mid
            else:hi=mid
        ends.append((lo+hi)/2)
    lo,hi=ends
    if lo>hi+1e-12:raise ValueError('invalid weighted minimizer interval')
    c=0. if lo<=0<=hi else (lo if abs(lo)<=abs(hi) else hi)
    return {'minimizer_interval':ends,'selected_nearest_zero':c,'gradient_at_selected':gradient(c),
      'weighted_training_Huber_loss':float(.5*(huber(c-groups[0]).mean()+huber(c-groups[1]).mean())),
      'unweighted_training_Huber_loss':float(huber(c-y).mean()),'root_iterations_per_endpoint':60}


def fit_balanced_huber(z,y,completed_update=None):
    import numpy as np
    x,target,weight,counts=balanced_problem(z,y);d=x.shape[1]
    regularizer=np.full(d,.001);regularizer[-1]=0.;w=np.zeros(d)
    curvature=x.T@(x*weight[:,None])+np.diag(regularizer)
    L=float(np.linalg.eigvalsh(curvature)[-1])
    if not math.isfinite(L) or L<=0:raise ValueError('invalid Lipschitz bound')
    initial=float(np.dot(weight,huber(-target)))
    for _ in range(200):
        gradient=x.T@(weight*np.clip(x@w-target,-1,1))+regularizer*w
        w-=gradient/L
        if not np.isfinite(w).all():raise ValueError('nonfinite fixed update')
        if completed_update is not None:completed_update()
    scores=x@w;objective=float(np.dot(weight,huber(scores-target))+.5*np.dot(regularizer*w,w))
    if not math.isfinite(objective) or objective>initial+1e-10:raise ValueError('convex objective increased/nonfinite')
    return w,{'training_classes':counts,'iterations':200,'Lipschitz_upper_bound':L,'step_size':1/L,
      'initial_weighted_objective':initial,'final_weighted_objective':objective,
      'final_gradient_norm':float(np.linalg.norm(x.T@(weight*np.clip(scores-target,-1,1))+regularizer*w)),
      'final_unweighted_training_Huber_loss':float(huber(scores-target).mean()),
      'coefficients':w.tolist(),'convergence_claimed':False,'prediction_is_physical_MAE_or_probability':False}


def auc(y,s):
    """Mann-Whitney tie-half-credit AUC, independent of any action threshold."""
    import numpy as np
    keep=y!=0;yy=y[keep]>0;ss=s[keep];pos=int(yy.sum());neg=len(yy)-pos
    if not pos or not neg:return None
    order=np.argsort(ss,kind='stable');values=ss[order];labels=yy[order].astype(np.int64)
    start=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1];size=np.diff(np.r_[start,len(values)])
    positive=np.add.reduceat(labels,start);negative=size-positive;prior_negative=np.cumsum(negative)-negative
    return float(np.sum(positive*(prior_negative+.5*negative))/(pos*neg))


def classify(y,s):
    import numpy as np
    if y.shape!=s.shape or not np.isfinite(s).all():raise ValueError('invalid classification score')
    pos=y>0;neg=y<0;act=s>0
    tpr=float(act[pos].mean()) if pos.any() else None;fpr=float(act[neg].mean()) if neg.any() else None
    return {'tiles':len(y),'positive_tiles':int(pos.sum()),'negative_tiles':int(neg.sum()),'zero_tiles':int((y==0).sum()),
      'balanced_accuracy':(tpr+1-fpr)/2 if tpr is not None and fpr is not None else None,
      'negative_false_K1':fpr,'positive_K1_recall':tpr,'K1_tile_fraction':float(act.mean()),
      'zero_label_K1_fraction':float(act[y==0].mean()) if (y==0).any() else None,'AUROC':auc(y,s),'unweighted_Huber_loss':float(huber(s-y).mean()),
      'score_quantiles_0_25_50_75_100':np.quantile(s,[0,.25,.5,.75,1]).tolist()}


def replay_old(P,x,y,split,old):
    import numpy as np
    tilefold=np.repeat(np.asarray(split['row_folds']),36);pred=np.empty(len(y));constant=np.empty(len(y));stats=[]
    for f in range(5):
        train=tilefold!=f;test=~train;st=P.standardizer(x[train]);prior=old['result']['folds'][f]
        P.need(prior['fold']==f and prior['training_tiles']==int(train.sum()) and prior['held_out_tiles']==int(test.sum()),'old fold identity differs')
        P.need(np.array_equal(st[0],prior['training_mean']) and np.array_equal(st[1],prior['training_scale'])
            and np.flatnonzero(st[2]).tolist()==prior['zero_variance_channel_indices'],'fold standardization changed')
        w=np.asarray(prior['probe_fit']['coefficients'],np.float64);P.need(w.shape==(12,),'old coefficient shape')
        pred[test]=P.transform(x[test],st)@w;constant[test]=prior['constant']['selected_nearest_zero'];stats.append(st)
    P.need(P.array_sha(pred)==old['result']['held_out_score_sha256']['linear']
      and P.array_sha(constant)==old['result']['held_out_score_sha256']['constant'],'old score exact replay failed; no new fit')
    return tilefold,stats,{'old_Huber_linear':pred,'old_Huber_constant':constant}


def summaries(y,scores,rows):
    import numpy as np
    sources=np.repeat(np.asarray([r['source_id'] for r in rows]),36);profiles=np.repeat(np.asarray([r['profile_id'] for r in rows]),36)
    per={s:{arm:classify(y[sources==s],v[sources==s]) for arm,v in scores.items()} for s in sorted(set(sources))}
    by_profile={p:{s:{arm:classify(y[(sources==s)&(profiles==p)],v[(sources==s)&(profiles==p)]) for arm,v in scores.items()}
        for s in sorted(set(sources)) if ((sources==s)&(profiles==p)).any()} for p in sorted(set(profiles))}
    keys=('balanced_accuracy','negative_false_K1','positive_K1_recall','K1_tile_fraction','AUROC','unweighted_Huber_loss')
    def macro(table):
        result={}
        for arm in scores:
            result[arm]={}
            for k in keys:
                values=[v[arm][k] for v in table.values() if v[arm][k] is not None]
                result[arm][k]={'mean':float(np.mean(values)) if values else None,'valid_sources':len(values)}
        return result
    paired=[]
    for sid,v in per.items():
        def difference(k):
            a=v['balanced_Huber'][k];b=v['old_Huber_linear'][k]
            return None if a is None or b is None else a-b
        paired.append({'source_id':sid,'balanced_accuracy_balanced_minus_Huber':difference('balanced_accuracy'),
          'AUROC_balanced_minus_Huber':difference('AUROC'),'negative_false_K1_balanced_minus_Huber':difference('negative_false_K1'),
          'positive_recall_balanced_minus_Huber':difference('positive_K1_recall'),'unweighted_Huber_loss_balanced_minus_Huber':difference('unweighted_Huber_loss')})
    return {'source_macro_primary':macro(per),'per_source':per,'profile_source_macro':{p:macro(v) for p,v in by_profile.items()},
      'per_profile_source':by_profile,'paired_sources':paired,'tile_micro_secondary':{arm:classify(y,v) for arm,v in scores.items()},
      'descriptive_source_counts':{'old_Huber_AUROC_above_chance':sum(v['old_Huber_linear']['AUROC'] is not None and v['old_Huber_linear']['AUROC']>.5 for v in per.values()),
        'weighted_balanced_accuracy_above_chance':sum(v['balanced_Huber']['balanced_accuracy'] is not None and v['balanced_Huber']['balanced_accuracy']>.5 for v in per.values()),
        'weighted_balanced_accuracy_better_than_Huber':sum(p['balanced_accuracy_balanced_minus_Huber'] is not None and p['balanced_accuracy_balanced_minus_Huber']>0 for p in paired)},
      'source_count':len(per),'physical_MAE_gain_or_harm':None,'automatic_admission':False}


def inspect(P,c):
    P.need(c['schema']==SCHEMA and c['protocol']==PROTOCOL and c['resources']==P.RESOURCES and c['source_sha256']==P.file_sha(__file__),'new protocol/source differs')
    P.need(c['legacy_source_sha256']==LEGACY_SHA and c['root']==c['expected_root'],'legacy/root identity differs')
    fs=P.Files(c['root'])
    for key,sha in [('old_execution',CONTRACT_SHA),('old_report',REPORT_SHA),('old_split',SPLIT_SHA)]:P.need(c[key]['sha256']==sha,'accepted prior changed')
    original=P.read(fs.bound(c['old_execution']));old=P.read(fs.bound(c['old_report']));split=P.read(fs.bound(c['old_split']))
    P.need(original['root']==c['root'] and old['complete'] is True and old['status']=='TRAIN_internal_probe_complete_no_quality_or_training_admission'
        and old['source_sha256']==LEGACY_SHA and old['execution_contract_sha256']==CONTRACT_SHA,'old probe not completed')
    actualfs,manifest,rows,computed=P.metadata(original)
    P.need(computed==split and original['runtime']==c['runtime'],'exact split/runtime changed')
    return actualfs,manifest,rows,split,old


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',required=True);p.add_argument('--execution-contract-sha256',required=True)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args();P=legacy()
    P.need(P.file_sha(a.execution_contract)==a.execution_contract_sha256,'execution SHA differs');c=P.read(a.execution_contract)
    P.need(c['out']==str(a.out.resolve()) and not a.out.exists(),'fresh exact output required');a.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();r={'schema':'rphi.input_preservation_balanced_huber_probe.v1','complete':False,'status':'preparing',
      'source_sha256':P.file_sha(__file__),'execution_contract_sha256':a.execution_contract_sha256,'protocol':PROTOCOL,'resources':P.RESOURCES,
      'new_weighted_Huber_fit_attempts':0,'new_weighted_Huber_fits':0,'new_gradient_updates_completed':0,'extra_unweighted_Huber_fits':0,'research_head_updates':0,'restorer_calls':0,'DEV_paths_opened':0,
      'original_development_failure_unchanged':True,'deployment_or_training_admitted':False,
      'limitations':['Only the small probe is group-held-out; the fixed teacher previously trained on all19 TRAIN sources.',
       'Class-balanced weights change positive/negative costs but keep the original clipped normalized utility target. Predictions are not physical MAE or calibrated probabilities.',
       'This tile-local linear family omits original3x3/nonlinear context. A negative result does not prove all raw features uninformative.',
       'Source statistics describe correlated TRAIN development data, not significance or new independent-test quality.',
       'Old Huber raw scores support per-source AUROC even if all are positive. Pooled cross-fold AUC is secondary only.',
       'Fixed200 steps do not certify convergence. Numerical near-zero scores retain literal threshold0 with no epsilon recentering; constant actions still have balanced accuracy.5.',
       'No threshold, step, ridge, feature or fold search; no automatic repeat.']}
    try:
        if a.execute:P.set_limits()
        fs,manifest,rows,split,old=inspect(P,c);sp=a.out/'split.json';P.write_new(sp,split);sh=P.file_sha(sp)
        P.need(sh==SPLIT_SHA,'persisted split differs');r['split']={'path':str(sp),'sha256':sh,'written_before_arrays':True}
        if a.execute:
            import numpy as np
            versions={'python':'.'.join(map(str,sys.version_info[:3])),'numpy':np.__version__,'BLAS_threads':2,'Torch_imported':False}
            P.need(versions==c['runtime'],'actual runtime differs');r['runtime']=versions
            x,y,proof=P.read_training_arrays(fs,manifest,rows,sp,sh);r['array_load']=proof
            P.need(proof==old['array_load'],'actual input features/labels/load scope differs')
            folds,stats,scores=replay_old(P,x,y,split,old);r['old_score_replay_exact']=True
            predicted=np.empty(len(y));constant=np.empty(len(y));fits=[]
            def completed_update():r['new_gradient_updates_completed']+=1
            for f in range(5):
                train=folds!=f;test=~train;r['new_weighted_Huber_fit_attempts']+=1
                cw=weighted_constant(y[train]);w,fit=fit_balanced_huber(P.transform(x[train],stats[f]),y[train],completed_update)
                predicted[test]=P.transform(x[test],stats[f])@w;constant[test]=cw['selected_nearest_zero'];fits.append({'fold':f,'fit':fit,'weighted_constant':cw})
                r['new_weighted_Huber_fits']+=1
            scores['balanced_Huber']=predicted;scores['weighted_Huber_constant']=constant
            r['fits']=fits;r['result']=summaries(y,scores,rows);r['held_out_score_sha256']={k:P.array_sha(v) for k,v in scores.items()}
        inspect(P,c);P.need(P.file_sha(a.execution_contract)==a.execution_contract_sha256 and P.file_sha(sp)==SPLIT_SHA,'contract/split changed')
        P.need(not any(v.split('.')[0] in {'torch','torch_npu','h3_comfy'} for v in sys.modules),'model framework imported')
        r.update(complete=True,status='TRAIN_only_objective_diagnostic_complete_no_admission' if a.execute else 'metadata_exact_split_only_no_array_load')
    except BaseException as e:r.update(status='failed_fixed_budget_no_retry',error={'type':type(e).__name__,'message':str(e)})
    finally:
        signal.alarm(0);r['elapsed_seconds']=time.monotonic()-start;r['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        P.write_new(a.out/'report.json',r)
    print(json.dumps({k:r.get(k) for k in ('complete','status','new_weighted_Huber_fits','new_gradient_updates_completed','error')}));return 0 if r['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
