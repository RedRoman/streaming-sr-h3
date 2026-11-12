"""Bounded target-only continuity preparation; no degradation or model calls.

Four immutable targets are reused; 36 missing targets follow the exact frozen
center96 decoder. Each source gets a fresh process within one owned CPU group.
Scores rank review candidates only; this tool never approves continuity.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SCHEMA='openh3.history_utility_target_screen.v1'
REUSE={'2BdYaGuVAKo','4EKWTiWjPTQ','6UZVovkR3IU','6rJrrfRO3cA'}
BUDGET={'wall_seconds':1800,'grace_seconds':30,'memory_mib':768,'CPU_threads':1,
        'simultaneous_compute_workers':1,'output_disk_bytes':6*(1<<30),'new_target_decodes':36,'reused_targets':4,
        'degradation_codec_passes':0,'model_calls':0,'NPU_calls':0}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def write(path,r):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def bound(path):return {'path':str(path),'sha256':sha(path)}


def validate(c):
    require(c['schema']==SCHEMA and c['budget']==BUDGET,'scope/budget changed')
    require(c['scientific_continuity_approved'] is False and c['top_candidate_pairs_per_source']==6,'no automatic cut approval')
    require(c['full_resolution_pairs']=={'6rJrrfRO3cA':[72],'2BdYaGuVAKo':[94]},'full-size diagnostic scope changed')
    for name,row in c['files'].items():require(sha(row['path'])==row['sha256'],'pin changed: '+name)
    require(Path(c['files']['entry']['path']).resolve()==Path(__file__).resolve(),'entry not pinned')
    inventory=read(c['files']['source_inventory']['path'])
    require(inventory['existing']==inventory['SHA_matching']==inventory['geometry_duration_supported']==40
            and inventory['existing_role_counts']=={'train':32,'dev':8},'source inventory incomplete')
    require(all(inventory[k] for k in ('source_id_disjoint','source_SHA_disjoint','source_group_disjoint')),'source role overlap')
    rows={r['source_id']:r for r in inventory['records']}
    require(c['source_order']==sorted(rows) and len(c['source_order'])==40,'cohort/order changed')
    require(set(c['reused_targets'])==REUSE,'wrong reuse inventory')
    original=read(c['files']['numerical_window_contract']['path'])
    require(c['files']['numerical_window_contract']['sha256']=='13b6e03f8b5387a101997e1545214a03b43d281454e1d1848a589a57fd9c47ad','numerical window namespace changed')
    require(original['files']['entry_source']==c['files']['original_entry'],'old entry differs')
    for sid,r in rows.items():
        p=Path(r['path']);require(p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['source_sha256'],'actual source changed: '+sid)
    for sid,e in c['reused_targets'].items():
        require(sha(e['receipt']['path'])==e['receipt']['sha256'],'reuse receipt changed');r=read(e['receipt']['path'])
        require(r['source']['source_id']==sid and r['source']['source_sha256']==rows[sid]['source_sha256']
            and r['source']['split']==rows[sid]['split'],'reuse source binding differs')
        require(e['target']==r['target'] and sha(e['target']['path'])==e['target']['file_sha256'],'reuse target file changed')
    return original,rows


def load_original(c):
    p=Path(c['files']['original_entry']['path']);sys.path.insert(0,str(p.parent))
    spec=importlib.util.spec_from_file_location('_target_screen_frozen_entry',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def inspect(c):
    original,rows=validate(c);m=load_original(c)
    registry=m.D.load_source_registry(read(original['files']['plan']['path']),original['files']['train_manifest']['path'],original['files']['dev_manifest']['path'])
    require(set(rows)==set(registry) and all(rows[s]['source_sha256']==registry[s]['source_sha256'] and rows[s]['split']==registry[s]['split'] for s in rows),'full frozen registry differs')
    return {'schema':SCHEMA,'status':'inspected_not_materialized','source_file_SHA_verified':40,
        'all_source_file_SHA_verified':40,
        'reused_target_file_SHA_verified':4,'reused_target_pixels_verified':False,'budget':BUDGET,
        'model_calls':0,'NPU_calls':0,'import_guard':m.check_no_models(),'scientific_continuity_approved':False}


def diagnostics(frames,timestamps,out,sid,full_pairs):
    import numpy as np
    from PIL import Image,ImageDraw
    require(len(frames)==96 and len(timestamps)==96,'96 frames required')
    thumbs=[Image.fromarray(f).resize((104,60),Image.Resampling.BOX) for f in frames]
    scores=[]
    for i in range(1,96):
        a=np.asarray(thumbs[i-1],dtype=np.float32);b=np.asarray(thumbs[i],dtype=np.float32)
        scores.append({'new_frame_index':i,'score_mean_abs_RGB_0to1':float(np.abs(b-a).mean(dtype=np.float64)/255.0)})
    ranked=sorted(scores,key=lambda r:(-r['score_mean_abs_RGB_0to1'],r['new_frame_index']))[:6]
    sheet=Image.new('RGB',(832,1000),'white');draw=ImageDraw.Draw(sheet)
    draw.text((4,4),sid+' TARGET ONLY; cut/continuity NOT approved',fill='black')
    for i,img in enumerate(thumbs):
        x=(i%8)*104;y=24+(i//8)*80;sheet.paste(img,(x,y))
        draw.text((x+1,y+61),'%02d +%.3fs'%(i,timestamps[i]['pts_time_seconds']),fill='black')
    p=out/'target96.png';sheet.save(p);artifacts={'all_frame_sheet':bound(p)}
    candidates=Image.new('RGB',(416,6*150+24),'white');d=ImageDraw.Draw(candidates)
    d.text((4,4),'Ranked RGB changes; motion can score high',fill='black')
    for rank,r in enumerate(ranked):
        i=r['new_frame_index'];y=24+rank*150
        for column,index in enumerate((i-1,i)):
            candidates.paste(Image.fromarray(frames[index]).resize((208,120),Image.Resampling.BOX),(column*208,y))
            d.text((column*208+2,y+121),'frame%d  score%.5f'%(index,r['score_mean_abs_RGB_0to1']),fill='black')
    p=out/'candidate_pairs.png';candidates.save(p);artifacts['ranked_candidate_pairs']=bound(p)
    artifacts['full_resolution_pairs']=[]
    for i in full_pairs:
        require(1<=i<96,'invalid original-resolution pair')
        pair=Image.new('RGB',(frames.shape[2]*2,frames.shape[1]+24),'white');d=ImageDraw.Draw(pair)
        for j,index in enumerate((i-1,i)):
            pair.paste(Image.fromarray(frames[index]),(j*frames.shape[2],24));d.text((j*frames.shape[2]+2,2),sid+' frame'+str(index),fill='black')
        p=out/('full_pair_%02d_%02d.png'%(i-1,i));pair.save(p);artifacts['full_resolution_pairs'].append(bound(p))
    return {'score_definition':'mean absolute RGB change after PIL BOX104x60, normalized by255; descriptive only',
        'all95_pair_scores':scores,'top6':ranked,'artifacts':artifacts,'scientific_continuity_approved':False}


def worker(c,out,sid):
    import numpy as np
    original,rows=validate(c);require(sid in rows,'unlisted source');m=load_original(c)
    old=m.load_numeric(original['files']['numeric_source']['path']);budget=old.Budget(1800,768)
    registry=m.D.load_source_registry(read(original['files']['plan']['path']),original['files']['train_manifest']['path'],original['files']['dev_manifest']['path'])
    require(set(rows)==set(registry) and all(rows[s]['source_sha256']==registry[s]['source_sha256'] and rows[s]['split']==registry[s]['split'] for s in rows),'worker source registry differs')
    dest=out/'sources'/sid;dest.mkdir(parents=True,exist_ok=False)
    report={'schema':'openh3.history_utility_target_only_receipt.v1','complete':False,'source':registry[sid],
        'execution_contract_sha256':m.D.value_sha(c),'numerical_window_contract_file_sha256':c['files']['numerical_window_contract']['sha256'],
        'window_namespace_scope':'Frozen original13b6 numerical target/noise namespace only; current40source execution is separately authorized by this target-screen contract.',
        'materialized_observations':0,'degradation_codec_passes':0,'model_calls':0,'NPU_calls':0,'scientific_continuity_approved':False,'pid':os.getpid()}
    write(dest/'partial.json',report)
    try:
        if sid in c['reused_targets']:
            entry=c['reused_targets'][sid];prior=read(entry['receipt']['path']);target=prior['target']
            with np.load(target['path'],allow_pickle=False) as z:frames=z['frames']
            spec=prior['target_specification'];decode=prior['decode'];report.update(reused=True,reused_receipt=entry['receipt'],new_target_decodes=0)
            require(m.D.value_sha(prior['execution_contract'])==spec['execution_contract_sha256']==m.D.value_sha(original),'reused numerical namespace differs')
        else:
            source=Path(rows[sid]['path']);probe=old.source_probe(source,original['ffprobe']);spec=old.target_specification(registry[sid],probe,original)
            with tempfile.TemporaryDirectory(prefix='scratch_',dir=dest) as tmp:
                frames,decode=m.decode_target_owned(old,source,spec,Path(tmp),original['ffmpeg'],budget)
                shutil.copyfile(Path(tmp)/'target_decode.log',dest/'target_decode.log')
                target=old.save_npz(dest/'target.npz',frames,{'schema':'openh3.history_utility_target_cache.v1','role':'target',
                    **registry[sid],'target_specification':spec,'window_key':old.value_sha(spec),
                    'numeric_source_sha256':m.FROZEN_NUMERIC_SHA,'screen_execution_contract_sha256':m.D.value_sha(c)})
                del frames
            with np.load(target['path'],allow_pickle=False) as z:frames=z['frames']
            report.update(reused=False,new_target_decodes=1)
        require(frames.dtype==np.uint8 and frames.shape==m.D.SHAPE and old.pixel_sha(frames)==target['pixel_sha256'],'target geometry/pixels differ')
        report.update(target=target,target_specification=spec,window_key=old.value_sha(spec),decode=decode)
        report['diagnostics']=diagnostics(frames,decode['timestamps']['target'],dest,sid,c['full_resolution_pairs'].get(sid,[]))
        del frames;budget.check();require(sha(rows[sid]['path'])==rows[sid]['source_sha256'],'source changed after decode')
        report.update(complete=True,status='target_only_complete_not_observation_dataset',elapsed_seconds=time.monotonic()-budget.started,
            peak_parent_rss_bytes=old.peak_rss(),conservative_parent_plus_child_peak_bytes=budget.conservative_peak_sum,
            import_guard=m.check_no_models(),cache_reuse_note='Exact uint8 pixels can feed later observation materialization; new target-only metadata is not claimed to be an already-complete old v1 training NPZ receipt.')
        write(dest/'complete.json',report);(dest/'partial.json').unlink();return report
    except BaseException as e:
        report.update(complete=False,error=repr(e));write(dest/'failure.json',report);raise


def execute(path,c,out):
    require(out.is_absolute(),'absolute new namespace required')
    if out.exists():raise FileExistsError('output exists; no overwrite')
    validate(c);require(shutil.disk_usage(out.parent).free>=BUDGET['output_disk_bytes'],'need6GiB free disk')
    out.mkdir(parents=True,exist_ok=False)
    r={'schema':SCHEMA,'complete':False,'status':'running','sources':[],'new_target_decodes':0,'reused_targets':0,
        'degradation_codec_passes':0,'model_calls':0,'NPU_calls':0,'scientific_continuity_approved':False};started=time.monotonic();write(out/'partial.json',r)
    prior=signal.getsignal(signal.SIGTERM)
    def terminate(signum,frame):raise TimeoutError('owned supervisor requested termination')
    signal.signal(signal.SIGTERM,terminate)
    try:
        for sid in c['source_order']:
            args=[sys.executable,str(Path(__file__).resolve()),'--execution-contract',str(path),'--out',str(out),'--source-id',sid]
            with (out/(sid+'.log')).open('xb') as log:done=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
            require(done.returncode==0,'fresh target worker failed: '+sid);p=out/'sources'/sid/'complete.json';x=read(p);require(x['complete'],'worker incomplete')
            r['sources'].append({'source_id':sid,'receipt':bound(p),'pid':x['pid'],'reused':x['reused']});r['new_target_decodes']+=x['new_target_decodes'];r['reused_targets']+=int(x['reused'])
            used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());require(used<=BUDGET['output_disk_bytes'],'output disk budget exceeded')
            r['output_bytes']=used;write(out/'partial.json',r)
        require(len(r['sources'])==40 and r['new_target_decodes']==36 and r['reused_targets']==4,'actual count differs')
        require(len({x['pid'] for x in r['sources']})==40,'fresh PID reused');validate(c)
        r.update(complete=True,status='target_screen_complete_pending_root_visual_review',elapsed_seconds=time.monotonic()-started)
        write(out/'complete.json',r);(out/'partial.json').unlink();return r
    except BaseException as e:r.update(complete=False,status='failed',error=repr(e));write(out/'failure.json',r);raise
    finally:signal.signal(signal.SIGTERM,prior)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--source-id');a=p.parse_args();c=read(a.execution_contract)
    if a.source_id:worker(c,a.out,a.source_id)
    elif a.execute:execute(a.execution_contract,c,a.out)
    else:
        if a.out.exists():raise FileExistsError('inspection exists')
        r=inspect(c);a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open('x') as f:json.dump(r,f,indent=2)


if __name__=='__main__':main()
