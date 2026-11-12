"""One TRAIN-only objective intervention; no head/teacher training or DEV read.

Same eleven raw features, five source-group folds, fold-training standardization,
ridge .001 and 200 fixed GD steps. Only the objective changes to class-balanced
logistic sign prediction. Logits are not utility or calibrated probabilities.
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
SCHEMA='rphi.input_preservation_balanced_logistic_probe_execution.v1'
PROTOCOL={
 'single_intervention':'Replace clipped-utility Huber objective by fold-training class-balanced binary logistic sign objective; model/features/split/std/ridge/steps/threshold unchanged.',
 'features':'Same frozen FP64 arithmetic mean of two raw-feature slots;11 tile-local channels plus intercept; no source/profile input.',
 'folds':'Exact accepted19TRAIN source_group split, seed20261112, five folds. No DEV files.',
 'labels':'positive=original normalized benefit>0; negative=<0. Exact zeros excluded only from fitting/classification denominators and counted; all rows remain in standardization/replay.',
 'weights':'Each positive training tile weight1/(2*n_positive); each negative1/(2*n_negative); total weight1. Counts from fold-training only; missing class aborts.',
 'objective':'sum_i weight_i*(logaddexp(0,score_i)-binary_label_i*score_i)+.5*.001*sum(slopes**2); intercept unregularized.',
 'solver':'Zero coefficients; exactly200 full-batch GD steps with1/L, L=lambda_max(.25*Z.T@diag(weights)@Z+ridge_diagonal). No line search/early stop.',
 'threshold':0,'tie':'preserve raw','ridge':.001,'iterations':200,'std_zero_threshold':1e-12,
 'baseline':'Replay accepted Huber coefficients/constants on the exact same held-out standardized features; require both complete score array SHA match before any new fit. Zero extra Huber fits.',
 'metrics':'Per-source sign balanced accuracy, negative-label falseK1, positive-label recall, coverage, exact-tie half-credit AUROC; source macro primary with valid denominators, source/profile and tile micro secondary. No pooled cross-fold AUC used for a scientific conclusion.',
 'decision':'Descriptive diagnostic only: report old ranking vs fixed-zero action, and new balanced sign tradeoff. No deployment/threshold/teacher/head-training admission. After this single experiment pause preservation-head iterations; no automatic followup regardless of outcome.',
 'interpretation':'AUC>0.5 with all-positive Huber scores is evidence of ranking hidden by its positive offset. Better balanced sign decisions after this intervention are consistent with objective/prior effects, not proof of raw MAE improvement. Failure is limited to this tile-local linear family and fixed200 steps.'}


def legacy():
    import hashlib
    p=Path(__file__).with_name('legacy_probe.py')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=LEGACY_SHA:raise ValueError('frozen reader changed')
    spec=importlib.util.spec_from_file_location('frozen_huber_probe',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sigmoid(z):
    import numpy as np
    t=np.exp(-np.abs(z));return np.where(z>=0,1/(1+t),t/(1+t))


def balanced_problem(z,y):
    import numpy as np
    if z.ndim!=2 or y.ndim!=1 or len(z)!=len(y) or not np.isfinite(z).all() or not np.isfinite(y).all():raise ValueError('nonfinite/invalid design')
    keep=y!=0;x=z[keep];target=(y[keep]>0).astype(np.float64)
    pos=int(target.sum());neg=len(target)-pos
    if not pos or not neg:raise ValueError('both training sign classes required; no resampling')
    weight=np.where(target==1,1/(2*pos),1/(2*neg))
    return x,target,weight,{'positive_tiles':pos,'negative_tiles':neg,'excluded_exact_zero_tiles':int((~keep).sum()),
      'positive_total_weight':float(weight[target==1].sum()),'negative_total_weight':float(weight[target==0].sum())}


def fit_logistic(z,y):
    import numpy as np
    x,target,weight,counts=balanced_problem(z,y);d=x.shape[1]
    regularizer=np.full(d,.001);regularizer[-1]=0.;w=np.zeros(d)
    curvature=.25*(x.T@(x*weight[:,None]))+np.diag(regularizer)
    L=float(np.linalg.eigvalsh(curvature)[-1])
    if not math.isfinite(L) or L<=0:raise ValueError('invalid Lipschitz bound')
    initial=math.log(2)
    for _ in range(200):
        gradient=x.T@(weight*(sigmoid(x@w)-target))+regularizer*w
        w-=gradient/L
        if not np.isfinite(w).all():raise ValueError('nonfinite fixed update')
    scores=x@w;objective=float(np.dot(weight,np.logaddexp(0,scores)-target*scores)+.5*np.dot(regularizer*w,w))
    if not math.isfinite(objective) or objective>initial+1e-10:raise ValueError('convex objective increased/nonfinite')
    return w,{'training_classes':counts,'iterations':200,'Lipschitz_upper_bound':L,'step_size':1/L,
      'initial_objective':initial,'final_objective':objective,'final_gradient_norm':float(np.linalg.norm(x.T@(weight*(sigmoid(scores)-target))+regularizer*w)),
      'coefficients':w.tolist(),'convergence_claimed':False,'logits_are_utility_or_calibrated_probability':False}


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
      'zero_label_K1_fraction':float(act[y==0].mean()) if (y==0).any() else None,'AUROC':auc(y,s),
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
    keys=('balanced_accuracy','negative_false_K1','positive_K1_recall','K1_tile_fraction','AUROC')
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
            a=v['balanced_logistic'][k];b=v['old_Huber_linear'][k]
            return None if a is None or b is None else a-b
        paired.append({'source_id':sid,'balanced_accuracy_logistic_minus_Huber':difference('balanced_accuracy'),
          'AUROC_logistic_minus_Huber':difference('AUROC'),'negative_false_K1_logistic_minus_Huber':difference('negative_false_K1'),
          'positive_recall_logistic_minus_Huber':difference('positive_K1_recall')})
    return {'source_macro_primary':macro(per),'per_source':per,'profile_source_macro':{p:macro(v) for p,v in by_profile.items()},
      'per_profile_source':by_profile,'paired_sources':paired,'tile_micro_secondary':{arm:classify(y,v) for arm,v in scores.items()},
      'descriptive_source_counts':{'old_Huber_AUROC_above_chance':sum(v['old_Huber_linear']['AUROC'] is not None and v['old_Huber_linear']['AUROC']>.5 for v in per.values()),
        'logistic_balanced_accuracy_above_chance':sum(v['balanced_logistic']['balanced_accuracy'] is not None and v['balanced_logistic']['balanced_accuracy']>.5 for v in per.values()),
        'logistic_balanced_accuracy_better_than_Huber':sum(p['balanced_accuracy_logistic_minus_Huber'] is not None and p['balanced_accuracy_logistic_minus_Huber']>0 for p in paired)},
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
    start=time.monotonic();r={'schema':'rphi.input_preservation_balanced_logistic_probe.v1','complete':False,'status':'preparing',
      'source_sha256':P.file_sha(__file__),'execution_contract_sha256':a.execution_contract_sha256,'protocol':PROTOCOL,'resources':P.RESOURCES,
      'new_logistic_fits':0,'new_gradient_updates':0,'extra_Huber_fits':0,'research_head_updates':0,'restorer_calls':0,'DEV_paths_opened':0,
      'original_development_failure_unchanged':True,'deployment_or_training_admitted':False,
      'limitations':['Only the small probe is group-held-out; the fixed teacher previously trained on all19 TRAIN sources.',
       'Class-balanced sign loss deliberately changes costs and discards utility magnitude. Logits are not utility or calibrated deployment probabilities.',
       'This tile-local linear family omits original3x3/nonlinear context. A negative result does not prove all raw features uninformative.',
       'Source statistics describe correlated TRAIN development data, not significance or new independent-test quality.',
       'Old Huber raw scores support per-source AUROC even if all are positive. Pooled cross-fold AUC is secondary only.',
       'Fixed200 steps do not certify convergence. No threshold, step, ridge, feature or fold search; no automatic repeat.']}
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
            predicted=np.empty(len(y));fits=[]
            for f in range(5):
                train=folds!=f;test=~train;w,fit=fit_logistic(P.transform(x[train],stats[f]),y[train])
                predicted[test]=P.transform(x[test],stats[f])@w;fits.append({'fold':f,'fit':fit})
                r['new_logistic_fits']+=1;r['new_gradient_updates']+=200
            scores['balanced_logistic']=predicted;scores['balanced_constant_zero']=np.zeros(len(y))
            r['fits']=fits;r['result']=summaries(y,scores,rows);r['held_out_score_sha256']={k:P.array_sha(v) for k,v in scores.items()}
        inspect(P,c);P.need(P.file_sha(a.execution_contract)==a.execution_contract_sha256 and P.file_sha(sp)==SPLIT_SHA,'contract/split changed')
        P.need(not any(v.split('.')[0] in {'torch','torch_npu','h3_comfy'} for v in sys.modules),'model framework imported')
        r.update(complete=True,status='TRAIN_only_objective_diagnostic_complete_no_admission' if a.execute else 'metadata_exact_split_only_no_array_load')
    except BaseException as e:r.update(status='failed_fixed_budget_no_retry',error={'type':type(e).__name__,'message':str(e)})
    finally:
        signal.alarm(0);r['elapsed_seconds']=time.monotonic()-start;r['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        P.write_new(a.out/'report.json',r)
    print(json.dumps({k:r.get(k) for k in ('complete','status','new_logistic_fits','new_gradient_updates','error')}));return 0 if r['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
