"""Actual foreground postprocess PID, phase and metric snapshot."""
import datetime,json,subprocess
from pathlib import Path

remote=r'''
import json,subprocess,time
from pathlib import Path
root=Path('/home/saiadmin/streaming_sr_20261015/results/h3_setup/native_sr_pilot400_native_io_v2')
p=root/'postprocess_live.json';state=json.loads(p.read_text()) if p.exists() else None
proc=None
if state:
 r=subprocess.run(['ps','-p',str(state['pid']),'-o','pid=,etime=,stat=,args='],capture_output=True,text=True)
 if r.returncode==0 and 'finish_native_sr_pilot_v2.py' in r.stdout:proc=r.stdout.strip()
launch=json.loads((root/'postprocess_launch.json').read_text())
log=Path('/home/saiadmin/streaming_sr_20261015')/launch['stdout_stderr_log']
events=[]
for line in log.read_text().splitlines()[-20:]:
 try:
  d=json.loads(line)
  if d.get('event'):events.append({k:d[k] for k in ['event','case','mode','method','mean_frame_psnr_db','mean_lpips_alex_v01'] if k in d})
 except Exception:pass
print(json.dumps({'epoch':time.time(),'state':state,'matching_live_process':proc,'launch':launch,'latest_events':events,
 'gpu6':subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()}))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','sai-gpu161','/home/saiadmin/miniconda3/bin/python -'],input=remote,capture_output=True,text=True,check=True)
s=json.loads(r.stdout);root=Path('results/h3_setup/native_sr_pilot400_native_io_v2');(root/'postprocess_snapshot.json').write_text(json.dumps(s,indent=2)+'\n')
p=Path('results/h3_setup/compute_live_jobs.json');d=json.loads(p.read_text());d['checked_at']=datetime.datetime.fromtimestamp(s['epoch']).astimezone().isoformat();d['native_sr_pilot400_native_io_v2_postprocess_snapshot']=s
if s['matching_live_process']:
 d['active']={'native_sr_pilot400_native_io_v2_postprocess':{'exec_session_id':88349,'pid':s['state']['pid'],'host':'H100','phase':s['state']['phase'],'device':s['state'].get('device'),'log':s['launch']['stdout_stderr_log']}}
else:d['active']={}
p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'checked_at':d['checked_at'],'live_confirmed':bool(s['matching_live_process']),'phase':s['state']['phase'],'status':s['state']['status'],'gpu6':s['gpu6'],'latest_events':s['latest_events'][-3:]}))
