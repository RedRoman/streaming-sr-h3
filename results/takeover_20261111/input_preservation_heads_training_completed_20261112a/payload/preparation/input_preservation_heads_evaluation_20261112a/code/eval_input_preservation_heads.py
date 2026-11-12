"""One fixed CPU development screen of raw/K1/final matched/shuffled heads.

No restorer construction/replay, threshold search, checkpoint selection or
source fallback. Source/provenance metadata is excluded from head inputs.
LPIPS is a separately counted frozen CPU metric model, never a head input.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import audit_input_preservation_cache as A
import train_history_utility as T
import train_input_preservation_heads as H

SCHEMA='rphi.input_preservation_heads_evaluation_execution.v1'
REPORT_SCHEMA='rphi.input_preservation_heads_evaluation.v1'
SCREEN_SHA='144120baa552ae8371a73a03dcd3c48da9204e86a9f7e87186f3697ee0178577'
PROPOSAL_SHA='d1dd2afd1b9daddea8340fe1edff7c71ed6215fbf6f3ef038adaa4b01aa262de'
ALEX_SHA='7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'
LINEAR_SHA='df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'
TRAINER_SHA='2df11a8563940aea729e2b3e4252f38515c0a66889caa2635debef72239e54cd'
FROZEN_HELPERS={'train_input_preservation_heads.py':TRAINER_SHA,
 'rphi_input_preservation.py':'7a52656a3119dcaad515dee0ec591bccbb21ee7756ba150b7f59827ad3b8100d',
 'rphi_history_utility.py':'22131519f54403c08cb0992ea52acfd6b8de73783c4619353f25a3554a6d0367',
 'audit_input_preservation_cache.py':'ee9cbcd1dcb1a1de8cee7b09f19f1437e2d665e33added8bf42f606581d55aa1',
 'eval_rphi.py':'6f2526a0f503813b1e7364710433e40a87fd8cc442fbf29f56dad34914c64809'}
MODES=('raw','K1','matched','shuffled')
HEADS=('matched','shuffled')
ROWS,BATCH,TILE=2880,16,16
MAX_OUTPUT_BYTES=256<<20
METRIC_DEFINITION={
 'primary':'FP32 clipped[0,1] actual composed RGB; CPU FP64 MAE/MSE reductions',
 'psnr':'-10log10(mean frame clipped MSE) per source/profile; exact-zero is null+exact_match. Native macro averages finite source/profile PSNR; report finite/exact counts.',
 'LPIPS':'Alex v0.1, local pinned AlexNet and learned linear weights, CPU2threads/batch16; clipped FP32 RGB mapped to[-1,1]; no download/fallback.',
 'seam':'Let e=clip(pred)-target in FP64. Mean abs first difference of e across all horizontal/vertical internal16-pixel tile edges (15->16,...79->80). Pool both orientations by element count. Nonboundary edges, output-gradient, and pred-raw correction-gradient are separate diagnostics. The gate uses the absolute residual boundary mean, compared to raw and K1 under identical support.',
 'coverage':'correction mask tile mean; changed-pixel fraction is any FP32 RGB channel bit changed versus raw. Both unclipped and clipped output reported.',
 'temporal':'All95 adjacent pairs per source/profile. Mask xor fraction plus unwarped clipped-output/error change MAE. Phase-label changes separately reported; no flow-aligned or temporal-quality claim.',
 'selection':'threshold score>0; exact/signed-zero tie preserves raw; nearest16, no smoothing.',
 'arrays':'Save actual FP32 scores, uint8 masks, FP64 LPIPS distances; all four raw/clipped RGB SHA. RGB is exactly reconstructible from authenticated original raw/K1+mask. No repeated RGB payload.',
}


def need(value,message):
    if not value:raise ValueError(message)
def asha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def encoded(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def psnr(mse):return None if mse==0 else float(-10*np.log10(mse))


def rgb(value,shape=None,unit=False):
    need(isinstance(value,np.ndarray) and value.dtype==np.float32 and value.ndim==3 and value.shape==(3,96,96)
         and np.isfinite(value).all(),'finite FP32 3x96x96 RGB required')
    if unit:need(((value>=0)&(value<=1)).all(),'unit-range RGB required')
    return value


def edge_means(value):
    """First differences on actual internal tile edges; no learned support."""
    x=np.asarray(value,np.float64)
    need(x.shape==(3,96,96) and np.isfinite(x).all(),'edge RGB geometry/finite')
    dx=np.abs(np.diff(x,axis=2));dy=np.abs(np.diff(x,axis=1))
    boundary=(np.arange(95)+1)%16==0
    return {'boundary':float((dx[:,:,boundary].sum()+dy[:,boundary,:].sum())/(dx[:,:,boundary].size+dy[:,boundary,:].size)),
            'nonboundary':float((dx[:,:,~boundary].sum()+dy[:,~boundary,:].sum())/(dx[:,:,~boundary].size+dy[:,~boundary,:].size))}


def row_metrics(prediction,current,target,mask,lpips,clipped=None):
    rgb(prediction);rgb(current,unit=True);rgb(target,unit=True)
    need(mask.shape==(1,6,6) and mask.dtype==np.bool_ and math.isfinite(float(lpips)),'mask/LPIPS invalid')
    if clipped is None:clipped=np.ascontiguousarray(np.clip(prediction,0,1))
    else:
        rgb(clipped,unit=True)
        need(np.array_equal(clipped,np.clip(prediction,0,1)),'actual clipped output differs numerically')
    p=prediction.astype(np.float64);c=clipped.astype(np.float64);y=target.astype(np.float64)
    unclipped=p-y;error=c-y;seam=edge_means(error);outedge=edge_means(c);correction=edge_means(c-current.astype(np.float64))
    mse=float(np.mean(error**2));changed=lambda a:float(np.any(a.view(np.uint32)!=current.view(np.uint32),axis=0).mean())
    return {'clipped_mae':float(np.abs(error).mean()),'clipped_mse':mse,'clipped_psnr':psnr(mse),'clipped_exact_match':mse==0,
      'unclamped_mae':float(np.abs(unclipped).mean()),'unclamped_mse':float(np.mean(unclipped**2)),
      'lpips':float(lpips),'corrected_tile_fraction':float(mask.mean()),'changed_pixel_fraction_unclamped':changed(np.ascontiguousarray(prediction)),
      'changed_pixel_fraction_clipped':changed(clipped),'clipped_component_fraction':float(((prediction<0)|(prediction>1)).mean()),
      'clipped_pixel_fraction':float(np.any((prediction<0)|(prediction>1),axis=0).mean()),
      'clip_mean_absolute_change':float(np.abs(p-c).mean()),'residual_boundary_jump':seam['boundary'],
      'residual_nonboundary_jump':seam['nonboundary'],'output_boundary_jump':outedge['boundary'],
      'correction_boundary_jump':correction['boundary'],'unclamped_rgb_sha256':asha(prediction),'clipped_rgb_sha256':asha(clipped)}


def compose_numpy(current,k1,score):
    """Independent byte-exact composition check, not a replacement network."""
    need(score.dtype==np.float32 and score.shape==(len(current),1,6,6) and np.isfinite(score).all(),'score invalid')
    mask=score>0;pixels=np.repeat(np.repeat(mask,16,axis=2),16,axis=3)
    return np.ascontiguousarray(np.where(pixels,k1,current)),mask


def pair_metrics(previous,current):
    need(current['frame_index']==previous['frame_index']+1 and
         all(current[k]==previous[k] for k in ('source_id','profile_id','window_key','crop_tlhw')),'nonadjacent/mixed temporal pair')
    result={'source_id':current['source_id'],'profile_id':current['profile_id'],'from_frame':previous['frame_index'],
      'to_frame':current['frame_index'],'from_phase':previous['severity_phase'],'to_phase':current['severity_phase'],
      'declared_phase_transition':previous['severity_phase']!=current['severity_phase'],'arms':{}}
    for arm in MODES:
        p=previous['arrays'][arm].astype(np.float64);q=current['arrays'][arm].astype(np.float64)
        result['arms'][arm]={'mask_flip_fraction':float(np.not_equal(previous['masks'][arm],current['masks'][arm]).mean()),
          'unwarped_output_change_mae':float(np.abs(q-p).mean()),
          'unwarped_error_change_mae':float(np.abs((q-current['target'].astype(np.float64))-(p-previous['target'].astype(np.float64))).mean())}
    return result


MEAN_KEYS=('clipped_mae','clipped_mse','unclamped_mae','unclamped_mse','lpips','corrected_tile_fraction',
 'changed_pixel_fraction_unclamped','changed_pixel_fraction_clipped','clipped_component_fraction','clipped_pixel_fraction',
 'clip_mean_absolute_change','residual_boundary_jump','residual_nonboundary_jump','output_boundary_jump','correction_boundary_jump')


def aggregate_rows(rows):
    groups=defaultdict(list)
    for row in rows:
        for phase in {row['severity_phase'],'all_frames'}:groups[row['source_id'],row['profile_id'],phase].append(row)
    details=[]
    for (source,profile,phase),values in sorted(groups.items()):
        arms={}
        for arm in MODES:
            m={k:float(np.mean([r['arms'][arm][k] for r in values])) for k in MEAN_KEYS}
            m.update(psnr_from_mean_mse=psnr(m['clipped_mse']),exact_match=m['clipped_mse']==0);arms[arm]=m
        details.append({'source_id':source,'profile_id':profile,'phase':phase,'frames':len(values),'arms':arms})
    sources=[];profiles=sorted({r['profile_id'] for r in rows if r['profile_id'] not in ('identity','near_clean')})
    need(len(profiles)==3,'exactly three native profiles required')
    ids=sorted({r['source_id'] for r in rows});need(len(ids)==6,'exactly six dev sources required')
    def macro(values):
        arms={}
        for arm in MODES:
            m={k:float(np.mean([v['arms'][arm][k] for v in values])) for k in MEAN_KEYS}
            finite=[v['arms'][arm]['psnr_from_mean_mse'] for v in values if v['arms'][arm]['psnr_from_mean_mse'] is not None]
            m.update(psnr_macro_finite=float(np.mean(finite)) if finite else None,psnr_finite_count=len(finite),
                     psnr_exact_count=sum(v['arms'][arm]['exact_match'] for v in values));arms[arm]=m
        return arms
    for sid in ids:
        native=[r for r in details if r['source_id']==sid and r['phase']=='all_frames' and r['profile_id'] in profiles]
        need(len(native)==3 and all(r['frames']==96 for r in native),'native source/profile coverage differs')
        sources.append({'source_id':sid,'arms':macro(native)})
    all_native=[r for r in details if r['phase']=='all_frames' and r['profile_id'] in profiles]
    # Exactly 18 equally sized source/profile groups, equivalent to mean3 then mean6.
    native=macro(all_native);controls={}
    for control in ('identity','near_clean'):
        selected=[r for r in details if r['profile_id']==control and r['phase']=='all_frames']
        need(len(selected)==6 and all(r['frames']==96 for r in selected),'control coverage differs')
        controls[control]={'arms':macro(selected),'per_source':selected}
    by_profile={p:macro([r for r in all_native if r['profile_id']==p]) for p in profiles}
    return {'per_source_profile_phase':details,'native_per_source':sources,'native_macro':native,'native_per_profile':by_profile,
            'controls':controls,'native_profile_ids':profiles,'source_ids':ids}


def aggregate_pairs(pairs):
    groups=defaultdict(list)
    for r in pairs:groups[r['source_id'],r['profile_id']].append(r)
    details=[]
    for (source,profile),values in sorted(groups.items()):
        need(len(values)==95 and [(v['from_frame'],v['to_frame']) for v in values]==[(t-1,t) for t in range(1,96)],'stride1 pair coverage differs')
        conditions={'all':values,'declared_transition':[v for v in values if v['declared_phase_transition']],
                    'same_phase':[v for v in values if not v['declared_phase_transition']]}
        for key,selected in conditions.items():
            details.append({'source_id':source,'profile_id':profile,'support':key,'pairs':len(selected),
              'arms':{a:{k:float(np.mean([v['arms'][a][k] for v in selected])) if selected else None
                         for k in ('mask_flip_fraction','unwarped_output_change_mae','unwarped_error_change_mae')} for a in MODES}})
    need(len(groups)==30 and len(pairs)==2850,'full temporal coverage differs')
    return {'per_source_profile':details,'pairs':len(pairs),'declared_phase_transition_pairs':sum(r['declared_phase_transition'] for r in pairs),
            'interpretation':'Decision changes and unwarped RGB changes only; no flow-aware flicker or temporal-quality gate.'}


def screen_decision(summary,complete_metrics=True):
    n=summary['native_macro'];m=n['matched'];k=n['K1'];s=n['shuffled'];raw=n['raw'];margin=.01/255;harm=.05/255
    src=summary['native_per_source'];g1=[v['arms']['K1']['clipped_mae']-v['arms']['matched']['clipped_mae'] for v in src]
    gs=[v['arms']['shuffled']['clipped_mae']-v['arms']['matched']['clipped_mae'] for v in src]
    den=raw['clipped_mae']-k['clipped_mae'];retained=(raw['clipped_mae']-m['clipped_mae'])/den if den>0 else None
    identity=summary['controls']['identity'];near=summary['controls']['near_clean'];im=identity['arms']['matched'];nc=near['arms']
    gates={
      'native_gain_vs_K1':k['clipped_mae']-m['clipped_mae']>=margin,
      'native_gain_vs_shuffle':s['clipped_mae']-m['clipped_mae']>=margin,
      'four_sources_vs_K1':sum(v>0 for v in g1)>=4,'four_sources_vs_shuffle':sum(v>0 for v in gs)>=4,
      'native_each_profile_not_worse_than_K1':all(v['matched']['clipped_mae']<=v['K1']['clipped_mae'] for v in summary['native_per_profile'].values()),
      'native_worst_source_harm':max(-v for v in g1)<=harm,
      'raw_restoration_retained':den>0 and raw['clipped_mae']>m['clipped_mae'] and retained>=.95,
      'unclamped_gain_not_reversed':k['unclamped_mae']>=m['unclamped_mae'] and s['unclamped_mae']>=m['unclamped_mae'],
      'native_matched_coverage_nondegenerate':0<m['corrected_tile_fraction']<1,
      'native_shuffled_coverage_nondegenerate':0<s['corrected_tile_fraction']<1,
      'identity_mean_MAE':im['clipped_mae']<=.05/255,
      'identity_every_source_MAE':all(v['arms']['matched']['clipped_mae']<=.10/255 for v in identity['per_source']),
      'identity_better_than_K1':im['clipped_mae']<identity['arms']['K1']['clipped_mae'],
      'identity_correction_cap':im['corrected_tile_fraction']<=.05,
      'nearclean_mean_harm':nc['matched']['clipped_mae']-nc['raw']['clipped_mae']<=.05/255,
      'nearclean_every_source_harm':all(v['arms']['matched']['clipped_mae']-v['arms']['raw']['clipped_mae']<=.10/255 for v in near['per_source']),
      'nearclean_not_worse_than_K1':nc['matched']['clipped_mae']<=nc['K1']['clipped_mae'],
      'native_LPIPS_noninferior':m['lpips']<=k['lpips']+.001 and m['lpips']<=s['lpips']+.001,
      'control_LPIPS_noninferior':all(summary['controls'][c]['arms']['matched']['lpips']<=summary['controls'][c]['arms']['raw']['lpips']+.001 for c in ('identity','near_clean'))}
    mp,kp=m['psnr_macro_finite'],k['psnr_macro_finite']
    gates['native_PSNR_noninferior']=(mp is None) if kp is None else (True if mp is None else mp>=kp-.05)
    seam=m['residual_boundary_jump']>raw['residual_boundary_jump'] and m['residual_boundary_jump']>k['residual_boundary_jump']
    status='failed_dev_screen' if not all(gates.values()) else ('pending_review' if not complete_metrics or seam else 'eligible_for_next_bounded_fullframe_validation')
    return {'status':status,'numeric_gates':gates,'required_metric_coverage_complete':bool(complete_metrics),
      'seam_requires_review':bool(seam),'retained_gain':retained,
      'paired_source_gains':[{'source_id':v['source_id'],'K1_minus_matched':a,'shuffled_minus_matched':b} for v,a,b in zip(src,g1,gs)],
      'quality_proven':False,'independent_test':False,'automatic_training_or_fullframe_admission':False}


def validate_checkpoint_pair(checkpoints):
    common=None;binding=None;initial=None
    for arm in HEADS:
        x=checkpoints[arm];c=x['training_contract']
        need(c['schema']==H.SCHEMA and c['steps']==1000 and c['batch_size']==16 and c['seed']==20261111
             and c['optimizer']==H.OPTIMIZER and c['loss']==H.LOSS and c['model_config']==H.CONFIG
             and x['optimizer_specification']==H.OPTIMIZER and x['binding']==H.binding(c),
             'checkpoint training recipe/binding differs')
        rt=c['runtime'];need(rt['device']=='cpu' and rt['threads']==2 and rt['dtype']=='float32'
             and rt['autocast'] is False and rt['grad_scaler'] is False and rt['expected_versions']['torch_npu'] is None,
             'checkpoint training precision/backend differs')
        sources={Path(p).name:h for p,h in c['implementation_sha256'].items()}
        need(sources.get('train_input_preservation_heads.py')==TRAINER_SHA and sources.get('rphi_input_preservation.py')==A.INPUT_MODULE_SHA,
             'checkpoint trained by a different implementation')
        need(x['schema']=='rphi.input_preservation_head_checkpoint.v1' and x['artifact_role']=='research_final_head'
             and x['arm']==c['arm']==arm and x['completed_step']==1000 and x['row_cursor']==16000
             and c['mode']=='research','only final1000 research head accepted')
        need(x['execution_payload_sha256']==T.execution_sha(c),'checkpoint execution identity differs')
        value={k:v for k,v in c.items() if k not in ('arm','admission')}
        if common is None:common,binding,initial=value,x['binding'],x['initial_state_sha256']
        need(value==common and x['binding']==binding and x['initial_state_sha256']==initial and bool(initial),'head arms are not a matched pair')
    return binding


def verify_readiness(c,files):
    ready=T.read(files['metric_readiness'])
    need(ready['schema']=='rphi.input_preservation_metric_readiness.v1' and ready['complete'] is True and ready['frozen'] is True
         and ready['proposal_sha256']==PROPOSAL_SHA and ready['screen_sha256']==SCREEN_SHA,'metric readiness differs')
    need(T.bound_path(ready['evaluator'],c['root'])==Path(__file__).resolve(),'different frozen evaluator')
    need(ready['runtime']['device']=='cpu' and ready['runtime']['threads']==2 and ready['runtime']['batch_size']==16,'CPU metric backend differs')
    lp=ready['LPIPS'];need(lp['available'] is True and lp['network']=='alex' and lp['version']=='0.1'
      and lp['input']=='clipped_RGB_float32_mapped_minus1_plus1' and lp['downloads'] is False,'LPIPS protocol differs')
    weights={x['role']:x for x in lp['weights']};need(set(weights)=={'alexnet','linear'},'two LPIPS weight files required')
    need(weights['alexnet']['sha256']==ALEX_SHA and weights['linear']['sha256']==LINEAR_SHA,'LPIPS weights differ')
    for v in weights.values():T.bound_path({k:v[k] for k in ('path','sha256')},c['root'])
    for p,h in ready['implementation_sha256'].items():need(T.file_sha(p)==h,'metric source changed')
    need(ready['implementation_sha256'].get(str(Path(__file__).resolve()))==T.file_sha(__file__),'evaluator source not frozen')
    return ready,weights


def inspection(path,execute=False,out=None):
    c=T.read(path);root=c['root']
    need(c['schema']==SCHEMA and Path(root).is_absolute() and c['threshold']==0 and c['batch_size']==16
         and c['metric_definition']==METRIC_DEFINITION,'fixed evaluation contract differs')
    need(c['resources']=={'wall_seconds':1800,'CPU_cores':2,'RSS_limit_bytes':4<<30,'max_output_bytes':MAX_OUTPUT_BYTES},'finite CPU evaluation budget differs')
    names={'screen','proposal','metric_readiness','matched_checkpoint','shuffled_checkpoint','train_cache_manifest','train_cache_audit','dev_cache_manifest','dev_cache_audit'}
    need(set(c['files'])==names and c['files']['screen']['sha256']==SCREEN_SHA and c['files']['proposal']['sha256']==PROPOSAL_SHA,'evaluation input bindings differ')
    paths={k:T.bound_path(v,root) for k,v in c['files'].items()};docs={k:T.read(p) for k,p in paths.items() if not k.endswith('checkpoint')}
    for role,rows,arrays in [('train',16000,7000),('dev',2880,1440)]:
        a=docs[role+'_cache_audit'];m=docs[role+'_cache_manifest']
        need(a['schema']=='rphi.input_preservation_cache_actual_audit.v1' and a['complete'] is True
             and a['status']=='actual_cache_data_features_and_labels_verified_no_training_admission'
             and a['role']==role and a['actual_rows']==rows and a['actual_cache_arrays']==arrays
             and a['producer_manifest']==c['files'][role+'_cache_manifest'] and m['complete'] is True and m['role']==role,
             'actual independent cache acceptance missing')
    ready,weights=verify_readiness(c,paths)
    for p,h in c['implementation_sha256'].items():need(T.file_sha(p)==h,'evaluation implementation pin changed')
    for name,h in FROZEN_HELPERS.items():need(T.file_sha(Path(__file__).parent/name)==h,'frozen numerical helper changed: '+name)
    for name in ('eval_input_preservation_heads.py','audit_input_preservation_cache.py','train_history_utility.py','train_input_preservation_heads.py','rphi_input_preservation.py','rphi_history_utility.py','eval_rphi.py'):
        p=Path(__file__).parent/name;need(c['implementation_sha256'].get(str(p.resolve()))==T.file_sha(p),'loaded implementation unpinned: '+name)
    if execute:
        admission=T.read(T.bound_path(c['admission'],root))
        need(admission['schema']=='rphi.input_preservation_heads_evaluation_admission.v1' and admission['approved'] is True
             and admission['execution_sha256']==T.execution_sha(c) and admission['entry_sha256']==T.file_sha(__file__)
             and admission['out']==str(Path(out).resolve()),'exact separate evaluation admission required')
    return c,paths,docs,ready,weights


def load_heads(paths,c,report):
    import torch
    from rphi_input_preservation import InputPreservationHead
    ck={arm:torch.load(paths[arm+'_checkpoint'],map_location='cpu',weights_only=True) for arm in HEADS}
    binding=validate_checkpoint_pair(ck);models={};baseline=InputPreservationHead(seed=20261111)
    initial=T.state_sha(baseline.state_dict());config=baseline.model_config
    for key in ('proposal','screen','metric_readiness','train_cache_manifest','train_cache_audit','dev_cache_manifest','dev_cache_audit'):
        need(binding['files'][key]==c['files'][key],'checkpoint/cache/readiness binding differs: '+key)
    for arm in HEADS:
        x=ck[arm];model=InputPreservationHead(seed=20261111)
        need(x['model_config']==config and x['initial_state_sha256']==initial,'head initialization/config differs')
        state=x['state_dict'];expected=model.state_dict()
        need(set(state)==set(expected) and all(v.dtype==torch.float32 and tuple(v.shape)==tuple(expected[k].shape)
             and bool(torch.isfinite(v).all()) for k,v in state.items()) and sum(v.numel() for v in state.values())==1745,'head state tensors differ')
        need(T.state_sha(state)==x['final_state_sha256'],'head final state SHA differs')
        model.load_state_dict(state,strict=True);model.eval().requires_grad_(False);models[arm]=model
    report['checkpoints']={arm:{'file':c['files'][arm+'_checkpoint'],'initial_state_sha256':initial,
                         'final_state_sha256':ck[arm]['final_state_sha256'],'completed_step':1000} for arm in HEADS}
    return models


def load_lpips(ready,weights,root):
    import torch
    import lpips
    import lpips.pretrained_networks as pn
    import torchvision.models as tv
    import eval_rphi
    sources=[inspect.getsourcefile(lpips.LPIPS),inspect.getsourcefile(pn.alexnet),inspect.getsourcefile(tv.alexnet),
             inspect.getsourcefile(inspect.unwrap(tv.alexnet)),inspect.getsourcefile(eval_rphi.load_local_lpips)]
    for p in sources:need(p is not None and ready['implementation_sha256'].get(str(Path(p).resolve()))==T.file_sha(p),'actual LPIPS/loader source unpinned')
    linear=Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
    need(linear.resolve()==T.bound_path({k:weights['linear'][k] for k in ('path','sha256')},root).resolve(),'actual LPIPS linear path differs')
    metric,info=eval_rphi.load_local_lpips(T.bound_path({k:weights['alexnet'][k] for k in ('path','sha256')},root))
    need(metric is not None and info['available'] is True and info['downloads'] is False and metric.version=='0.1','local fixed LPIPS unavailable; no fallback')
    metric.eval().requires_grad_(False);need(all(p.device.type=='cpu' and p.grad is None for p in metric.parameters()),'LPIPS not frozen CPU')
    return metric,info


def lpips_batch(metric,clipped_prediction,target):
    import torch
    need(clipped_prediction.dtype==target.dtype==np.float32 and clipped_prediction.shape==target.shape==(16,3,96,96)
         and np.isfinite(clipped_prediction).all() and np.isfinite(target).all()
         and ((clipped_prediction>=0)&(clipped_prediction<=1)).all() and ((target>=0)&(target<=1)).all(),
         'LPIPS must receive actual clipped FP32 RGB and unit target')
    with torch.inference_mode():
        p=torch.from_numpy(np.ascontiguousarray(clipped_prediction))*2-1;y=torch.from_numpy(np.ascontiguousarray(target))*2-1
        value=metric(p,y).detach().cpu().reshape(-1).numpy().astype(np.float64)
    need(value.shape==(16,) and np.isfinite(value).all(),'complete finite LPIPS required')
    return value


def dev_batches(manifest):
    """Authenticated complete96 histories; never fall back to source NPZ."""
    pc=manifest['execution_contract'];root=pc['root'];h=T.read(T.bound_path(pc['files']['headroom_report'],root))
    original=A.read_lines(T.bound_path(pc['files']['headroom_records'],root));cache_rows=A.read_lines(manifest['records']['path']);entries=A.read_lines(manifest['batches']['path'])
    for key in ('records','batches'):
        v=manifest[key];need(T.file_sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes'],'dev row/index file changed')
    need(len(original)==len(cache_rows)==2880 and len(entries)==180,'complete dev rows required')
    base=[{k:v for k,v in r.items() if k not in A.EXTRA} for r in original]
    need(T.value_sha(base)==h['evaluation_rows_sha256'],'old dev row table changed')
    oldroot=Path(h['records']['path']).parent;newroot=Path(manifest['records']['path']).parent
    oldshape={n:((16,3,96,96),np.dtype('float32')) for n in A.OLD_NAMES}
    for start in range(0,2880,96):
        retained=[]
        for j in range(start//16,start//16+6):
            e=h['actual_rgb_caches'][j];a=A.load_npz(e,oldroot,oldshape)
            need(e['row_indices']==list(range(j*16,(j+1)*16)) and e['row_sha256']==[r['row_sha256'] for r in base[j*16:(j+1)*16]],'old cache rows changed')
            retained.append({k:a[k] for k in ('current','target','current_only')})
        arrays={k:np.concatenate([v[k] for v in retained]) for k in retained[0]};obs=[]
        for t in range(96):
            r=original[start+t];first=original[start]
            need(r['split']=='dev' and r['frame_index']==t and all(r[k]==first[k] for k in ('source_id','profile_id','source_sha256','source_group_id','window_key','crop_tlhw')),'mixed or incomplete dev history')
            ix=[max(t-2,0),max(t-1,0),t];x=np.ascontiguousarray(arrays['current'][ix])
            need(r['history_indices']==ix and r['crop_tlhw']==[192,368,96,96] and asha(x)==r['observed_crop_sha256_float32']
              and asha(arrays['target'][t])==r['target_crop_sha256_float32'] and asha(arrays['current_only'][t])==r['metrics']['current_only']['prediction_float32_sha256'],'exact dev K3/target/K1 hash differs')
            obs.append(x)
        obs=np.stack(obs)
        for offset in range(0,96,16):
            j=(start+offset)//16;e=entries[j];rr=cache_rows[j*16:(j+1)*16];a=A.load_npz(e,newroot,A.expected_shapes('dev'));A.verify_record_records(e,rr,a,'dev')
            x=obs[offset:offset+16];availability=np.asarray([r['availability'] for r in original[j*16:(j+1)*16]],np.float32)
            need(asha(a['availability'])==asha(availability) and asha(A.raw_features_cpu(x,availability))==asha(a['raw_features']),'recomputed raw-only features differ')
            for i,r in enumerate(rr):
                need(all(r.get(k)==v for k,v in base[j*16+i].items()) and r['observed_crop_sha256_float32']==original[j*16+i]['observed_crop_sha256_float32']
                  and r['input_RGB_reference']=={'cache':h['actual_rgb_caches'][j],'batch_row':i},'dev cache provenance differs')
            yield j,rr,a['raw_features'],arrays['current'][offset:offset+16],arrays['current_only'][offset:offset+16],arrays['target'][offset:offset+16]


class Output:
    def __init__(self,path):self.path=Path(path).resolve();self.path.mkdir(parents=True,exist_ok=False);self.bytes=0;self.sizes={}
    def put(self,name,data,replace=False):
        p=self.path/name;p.parent.mkdir(exist_ok=True);old=self.sizes.get(name,0)
        need((replace and name in self.sizes) or not p.exists(),'output overwrite forbidden')
        need(self.bytes-old+len(data)+2*(1<<20)<=MAX_OUTPUT_BYTES,'evaluation output budget exceeded')
        temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(p)
        self.sizes[name]=len(data);self.bytes+=len(data)-old
        return {'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    def append(self,name,rows):
        b=b''.join(encoded(r) for r in rows);need(self.bytes+len(b)+2*(1<<20)<=MAX_OUTPUT_BYTES,'evaluation metadata budget exceeded')
        p=self.path/name;need(not p.exists() or name in self.sizes,'unowned metadata file')
        with p.open('ab') as f:f.write(b)
        self.sizes[name]=self.sizes.get(name,0)+len(b);self.bytes+=len(b)
    def arrays(self,index,values,rows):
        stream=io.BytesIO();np.savez(stream,**values);e=self.put(f'arrays/batch_{index:04d}.npz',stream.getvalue())
        e.update(batch_index=index,row_sha256=[r['row_sha256'] for r in rows],arrays={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':asha(v)} for k,v in values.items()})
        self.append('batches.jsonl',[e]);return e


def evaluate(c,paths,docs,ready,weights,sink,result):
    need(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','disable backend autoload before Torch import')
    import torch
    from rphi_input_preservation import compose_input_preservation
    need('torch_npu' not in sys.modules,'CPU only; NPU backend must not be imported');torch.set_num_threads(2)
    actual={'python':'.'.join(map(str,sys.version_info[:3])),'numpy':np.__version__,'torch':str(torch.__version__),
            'torchvision':importlib.metadata.version('torchvision'),'lpips':importlib.metadata.version('lpips'),'torch_npu':None}
    need(actual==ready['runtime']['expected_versions'],'exact metric/head CPU runtime differs');result['runtime']=actual
    models=load_heads(paths,c,result);metric,info=load_lpips(ready,weights,c['root']);result['lpips']=info
    before={a:T.state_sha(m.state_dict()) for a,m in models.items()};versions={a:{n:p._version for n,p in m.named_parameters()} for a,m in models.items()}
    metric_before=T.state_sha(metric.state_dict());records=[];pairs=[];previous=None
    counts=result['counts']
    observed={'matched':0,'shuffled':0,'LPIPS':0};hooks=[]
    def observer(name,maximum):
        def called(_module,_args):
            observed[name]+=1;need(observed[name]<=maximum,'actual '+name+' forward budget exceeded')
        return called
    for arm,model in models.items():hooks.append(model.gate.net.register_forward_pre_hook(observer(arm,180)))
    hooks.append(metric.register_forward_pre_hook(observer('LPIPS',720)))
    try:
        for index,rows,features,current,k1,target in dev_batches(docs['dev_cache_manifest']):
            predictions={'raw':current,'K1':k1};clipped_predictions={a:torch.from_numpy(v).clamp(0,1).numpy().copy() for a,v in predictions.items()}
            masks={'raw':np.zeros((16,1,6,6),bool),'K1':np.ones((16,1,6,6),bool)};saved={}
            with torch.inference_mode():
                for arm in HEADS:
                    counts['head_forward_attempts']+=1;counts['head_row_attempts']+=16
                    output=models[arm].forward_features(torch.from_numpy(features));score=output['benefit_score'].numpy().copy()
                    composed=compose_input_preservation(torch.from_numpy(current),torch.from_numpy(k1),output['benefit_score'])
                    pred=composed['rgb'].numpy().copy();mask=composed['correction_mask'].numpy().copy();expected,em=compose_numpy(current,k1,score)
                    need(asha(expected)==asha(pred) and asha(em)==asha(mask),'composition byte parity differs')
                    predictions[arm]=pred;clipped_predictions[arm]=composed['rgb_clipped'].numpy().copy()
                    masks[arm]=mask;saved[arm+'_score']=score;saved[arm+'_mask']=mask.astype(np.uint8)
                    counts['head_forwards_completed']+=1;counts['head_rows_completed']+=16
                distances=[]
                for arm in MODES:
                    counts['LPIPS_forward_attempts']+=1;counts['LPIPS_row_attempts']+=16
                    d=lpips_batch(metric,clipped_predictions[arm],target)
                    distances.append(d);counts['LPIPS_forwards_completed']+=1;counts['LPIPS_rows_completed']+=16
            saved['lpips']=np.stack(distances,axis=1);artifact=sink.arrays(index,saved,rows);batch_records=[];batch_pairs=[]
            for i,row in enumerate(rows):
                m={arm:row_metrics(predictions[arm][i],current[i],target[i],masks[arm][i],distances[j][i],clipped_predictions[arm][i]) for j,arm in enumerate(MODES)}
                record={k:row[k] for k in ('row_index','row_sha256','source_id','source_sha256','source_group_id','profile_id','severity_phase','window_key','crop_tlhw','frame_index','observed_crop_sha256_float32','target_crop_sha256_float32')}
                record.update(arms=m,cache_record_sha256=row['cache_record_sha256'],input_RGB_reference=row['input_RGB_reference'],actual_array_file=artifact['sha256'],batch_row=i,
                  array_row_sha256={k:asha(v[i]) for k,v in saved.items()})
                batch_records.append(record);records.append(record)
                actualrow={**record,'arrays':{a:clipped_predictions[a][i] for a in MODES},'masks':{a:masks[a][i] for a in MODES},'target':target[i]}
                if row['frame_index']>0:
                    need(previous is not None,'missing prior temporal row');pair=pair_metrics(previous,actualrow);pairs.append(pair);batch_pairs.append(pair)
                previous=actualrow
            sink.append('records.jsonl',batch_records);sink.append('pairs.jsonl',batch_pairs)
            result['completed_rows']+=16;sink.put('partial.json',encoded({k:result[k] for k in ('schema','complete','completed_rows','counts')}),replace='partial.json' in sink.sizes)
        need(result['completed_rows']==2880 and counts['head_forwards_completed']==360 and counts['head_rows_completed']==5760
          and counts['LPIPS_forwards_completed']==720 and counts['LPIPS_rows_completed']==11520,'complete four-arm evaluation budget required')
        need(observed=={'matched':180,'shuffled':180,'LPIPS':720},'actual model invocation coverage differs')
        result['summary']=aggregate_rows(records);result['temporal']=aggregate_pairs(pairs);result['screen']=screen_decision(result['summary'],True)
    finally:
        for hook in hooks:hook.remove()
        result['observed_model_invocations']=observed
        result['frozen_checks']={a:{'state_dict_exact':T.state_sha(m.state_dict())==before[a],
          'versions_exact':versions[a]=={n:p._version for n,p in m.named_parameters()},'eval_frozen_no_grads':not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters())} for a,m in models.items()}
        result['LPIPS_frozen_exact']=T.state_sha(metric.state_dict())==metric_before
    need(all(all(v.values()) for v in result['frozen_checks'].values()) and result['LPIPS_frozen_exact'] and 'torch_npu' not in sys.modules,'frozen CPU model/runtime changed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    c,paths,docs,ready,weights=inspection(a.execution_contract,a.execute,a.out)
    for role in ('train','dev'):A.fresh_audit_location(a.out,paths[role+'_cache_manifest'].parent)
    sink=Output(a.out);started=time.monotonic();result={'schema':REPORT_SCHEMA,'complete':False,'completed_rows':0,
      'execution_contract':c,'execution_contract_file_sha256':T.file_sha(a.execution_contract),'metric_definition':METRIC_DEFINITION,
      'counts':{k:0 for k in ('head_forward_attempts','head_row_attempts','head_forwards_completed','head_rows_completed','LPIPS_forward_attempts','LPIPS_row_attempts','LPIPS_forwards_completed','LPIPS_rows_completed','restorer_forwards','backwards','optimizer_updates')},
      'original_history_headroom_remains_failed':True,'head_or_restorer_training_admitted':False,'quality_proven':False}
    try:
        if a.execute:evaluate(c,paths,docs,ready,weights,sink,result);result['status']='complete_fixed_development_screen'
        else:result['status']='metadata_inspected_no_models_or_arrays'
        inspection(a.execution_contract,a.execute,a.out);result['complete']=True
    except BaseException as error:
        result.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        result.setdefault('screen',{'status':'pending_review','required_metric_coverage_complete':False,'quality_proven':False})
    result['elapsed_seconds']=time.monotonic()-started
    for name in ('records.jsonl','pairs.jsonl','batches.jsonl'):
        if name in sink.sizes:result[name.split('.')[0]]={'path':str(sink.path/name),'sha256':T.file_sha(sink.path/name),'bytes':sink.sizes[name]}
    result['output_bytes_before_report']=sink.bytes;sink.put('report.json',encoded(result))
    print(json.dumps({k:result[k] for k in ('complete','status','completed_rows','counts')}));return 0 if result['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
