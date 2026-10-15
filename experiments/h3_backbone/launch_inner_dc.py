"""Tracked foreground exec for sampling or its separate postprocess."""
import datetime,json,os,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root);phase=sys.argv[1];assert phase in ['sample','postprocess']
out=Path('results/h3_setup/sampling_dc_pilot');(out/'logs').mkdir(parents=True,exist_ok=True)
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S');log=out/'logs'/(phase+'_'+stamp+'.log')
command=[sys.executable,'-u','experiments/h3_backbone/'+('sample_inner_dc.py' if phase=='sample' else 'finish_inner_dc.py')]
record={'pid':os.getpid(),'epoch':time.time(),'phase':phase,'command':command,'stdout_stderr_log':str(log),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'foreground_only':True,'not_detached':True}
(out/(phase+'_launch.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'event':'foreground_launch',**record}),flush=True)
with log.open('ab',buffering=0) as stream:os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
os.execv(sys.executable,command)
