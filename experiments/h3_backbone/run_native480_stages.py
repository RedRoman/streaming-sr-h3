"""Foreground admitted sample/decode of an already cached native480 guide."""
import argparse
import json
from pathlib import Path
import sys
import time
from run_guide_diagnostic import admit,run_monitored


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert (a.out/'guide.safetensors').exists()
    status={'status':'running','completed':[]};path=a.out/'native480_live.json'
    try:
        for phase in ['sample','decode']:
            result=a.out/f'{phase}_result.json'
            if result.exists() and json.loads(result.read_text()).get('status')=='complete':continue
            admit()
            cmd=[sys.executable,'experiments/h3_backbone/staged_bf16_guide.py',phase,'--comfy','/root/autodl-tmp/h3-w4a8-20261011/ComfyUI','--output',str(a.out),'--lr-directory','data/real_video/davis_bmx_trees_native480_lr22','--frames','22','--width','832','--height','480']
            if phase=='sample':cmd+=['--model','/root/autodl-tmp/h3-qf-vdn-20261015/models/minimax_h3_fl2va_pruned_bf16.safetensors','--context','external/h3_assets/h3_qwen_context_neutral_restore.pt','--steps','49','--official-grid']
            else:cmd+=['--vae','/root/autodl-tmp/h3-quality-repair-20261013/models/vae/minimax_h3_video_vae_fp16.safetensors']
            status.update(phase=phase,command=cmd,epoch=time.time());path.write_text(json.dumps(status,indent=2)+'\n')
            run_monitored(cmd);status['completed'].append(phase)
        status['status']='complete'
    except Exception as exc:status.update(status='stopped_or_failed',exception=repr(exc));raise
    finally:path.write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)


if __name__=='__main__':main()
