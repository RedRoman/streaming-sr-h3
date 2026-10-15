"""Traceable foreground CPU audit; no detached worker."""
import datetime,json,os,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[2];os.chdir(r);o=Path('results/h3_setup/sampling_dc_davis8_temporal');(o/'logs').mkdir(parents=True,exist_ok=True);log=o/'logs'/('temporal_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log');cmd=[sys.executable,'-u','experiments/h3_backbone/evaluate_inner_dc_temporal.py'];d={'pid':os.getpid(),'epoch':time.time(),'command':cmd,'stdout_stderr_log':str(log),'foreground_only':True,'not_detached':True,'GPU_used':False};(o/'launch.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
with log.open('ab',buffering=0) as s:os.dup2(s.fileno(),1);os.dup2(s.fileno(),2)
os.execv(sys.executable,cmd)
