"""Foreground postprocess exec with a durable project log, never detached."""
import datetime
import json
import os
from pathlib import Path
import sys
import time

root=Path(__file__).resolve().parents[2];os.chdir(root)
out=Path('results/h3_setup/native_sr_pilot400_native_io_v2')
assert json.loads((out/'LIVE_STATE.json').read_text())['status']=='complete'
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
log=out/'logs'/('postprocess_'+stamp+'.log')
command=[sys.executable,'-u','experiments/h3_backbone/finish_native_sr_pilot_v2.py']
record={'pid':os.getpid(),'epoch':time.time(),'command':command,'stdout_stderr_log':str(log),
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'foreground_only':True,'not_detached':True}
(out/'postprocess_launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'event':'foreground_postprocess_launch',**record}),flush=True)
with log.open('ab',buffering=0) as stream:
    os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
os.execv(sys.executable,command)
