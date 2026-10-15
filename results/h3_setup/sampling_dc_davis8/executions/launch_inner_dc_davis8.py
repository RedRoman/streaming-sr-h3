"""Traceable foreground runner; all stdout/stderr in durable projectlog."""
import datetime,json,os,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[2];os.chdir(r);phase=sys.argv[1];assert phase in ['sample','evaluate'];out=Path('results/h3_setup/sampling_dc_davis8');(out/'logs').mkdir(parents=True,exist_ok=True)
log=out/'logs'/(phase+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
command=[sys.executable,'-u','experiments/h3_backbone/'+('sample_inner_dc_davis8.py' if phase=='sample' else 'evaluate_inner_dc_davis8.py')]
if phase=='evaluate':command+=['--run',str(out),'--val',str(out/'input_manifest.json'),'--device','cpu']
record={'pid':os.getpid(),'epoch':time.time(),'phase':phase,'command':command,'stdout_stderr_log':str(log),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'foreground_only':True,'not_detached':True};(out/(phase+'_launch.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
with log.open('ab',buffering=0) as stream:os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
os.execv(sys.executable,command)
