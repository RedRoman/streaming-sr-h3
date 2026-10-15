"""Actual remote PID/progress snapshot for the foreground nativeSRpilot."""
import datetime,json,subprocess
from pathlib import Path
remote=r'''
import json,subprocess,time
from pathlib import Path
root=Path('/home/saiadmin/streaming_sr_20261015/results/h3_setup/native_sr_pilot400')
p=root/'LIVE_STATE.json';state=json.loads(p.read_text()) if p.exists() else None
proc=None
if state and state.get('pid'):
 r=subprocess.run(['ps','-p',str(state['pid']),'-o','pid=,etime=,stat=,args='],capture_output=True,text=True)
 if r.returncode==0 and 'train_native_sr_pilot.py' in r.stdout:proc=r.stdout.strip()
progress={}
for mode in ['lora_only','guide_only','joint']:
 log=root/mode/'train.jsonl';rows=log.read_text().splitlines() if log.exists() else []
 progress[mode]={'logged_updates':len(rows),'last_update':json.loads(rows[-1]) if rows else None,'result':json.loads((root/mode/'result.json').read_text()) if (root/mode/'result.json').exists() else None}
validation=root/'validation.jsonl'
print(json.dumps({'epoch':time.time(),'state':state,'matching_live_process':proc,'progress':progress,'validation':list(map(json.loads,validation.read_text().splitlines())) if validation.exists() else [],'gpu6':subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()}))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','sai-gpu161','/home/saiadmin/miniconda3/bin/python -'],input=remote,capture_output=True,text=True,check=True)
snapshot=json.loads(r.stdout);out=Path('results/h3_setup/native_sr_pilot400');out.mkdir(exist_ok=True);(out/'live_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
p=Path('results/h3_setup/compute_live_jobs.json');d=json.loads(p.read_text());d['checked_at']=datetime.datetime.fromtimestamp(snapshot['epoch']).astimezone().isoformat();d['native_sr_pilot400_actual_live_snapshot']=snapshot;p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'checked_at':d['checked_at'],'live_confirmed':snapshot['matching_live_process'] is not None,'state':snapshot['state'],'gpu6':snapshot['gpu6'],'progress':{k:v['logged_updates'] for k,v in snapshot['progress'].items()}}))
