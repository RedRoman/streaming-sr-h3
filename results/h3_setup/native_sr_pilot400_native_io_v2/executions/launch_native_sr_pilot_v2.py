"""Foreground, tracked exec only; stdout/stderr survive transient transport loss."""
import datetime
import json
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[2]
os.chdir(root)
out=Path('results/h3_setup/native_sr_pilot400_native_io_v2')
out.mkdir(parents=True,exist_ok=True)
(out/'logs').mkdir(exist_ok=True)
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
log=out/'logs'/('train_'+stamp+'.log')
command=[sys.executable,'-u','experiments/h3_backbone/train_native_sr_pilot_native_io_v2.py',
         '--comfy','external/ComfyUI_H3',
         '--model','external/h3_assets/minimax_h3_fl2va_pruned_bf16.safetensors',
         '--train','results/h3_setup/native_sr_train_pairs/pairs_manifest.json',
         '--val','results/h3_setup/native_sr_val_pairs/pairs_manifest.json',
         '--context','external/h3_assets/h3_qwen_context_neutral_restore.pt',
         '--out',str(out),'--steps','400','--seed','703','--lr','1e-5']
record={'pid':os.getpid(),'epoch':__import__('time').time(),'command':command,
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'stdout_stderr_log':str(log),
        'foreground_only':True,'not_detached':True}
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'event':'foreground_launch',**record}),flush=True)
with log.open('ab',buffering=0) as stream:
    os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
os.execv(sys.executable,command)
