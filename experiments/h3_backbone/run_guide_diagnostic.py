"""Foreground serial native guide diagnostic; re-admit each GPU process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def admit():
    queues={}
    for port in [8188,8189,8190,8191]:
        data=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/queue',timeout=5))
        queues[str(port)]=[len(data['queue_running']),len(data['queue_pending'])]
    for attempt in range(3):
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        free,util=map(int,gpu.split(','))
        if util==0 or free<19000:break
        time.sleep(1)
    print(json.dumps({'event':'admission','queues':queues,'free_MiB':free,'util':util}),flush=True)
    if any(v!=[0,0] for v in queues.values()) or free<19000 or util!=0:
        raise RuntimeError('Current shared GPU is busy; no process launched')


def gpu_processes():
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return {int(row.split(',')[0]):int(row.split(',')[1]) for row in rows if row.strip()}


def run_monitored(cmd):
    resident=set(gpu_processes())
    child=subprocess.Popen(cmd)
    while child.poll() is None:
        time.sleep(1)
        foreign={pid:used for pid,used in gpu_processes().items() if pid not in resident and pid!=child.pid and used>1536}
        if foreign:
            print(json.dumps({'event':'yield_owned_process_to_new_shared_job','own_pid':child.pid,'new_gpu_jobs':foreign}),flush=True)
            child.terminate();child.wait(timeout=20)
            raise RuntimeError('Only our foreground child was stopped to release shared GPU for a newly arrived job')
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode,cmd)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--context',default='external/h3_assets/h3_qwen_context_neutral_restore.pt')
    args=p.parse_args()
    status_path=args.root/'diagnostic_live.json'
    status={'status':'running','completed':[]}
    try:
        for case in args.cases:
            out=args.root/case;manifest=json.loads((out/'guide_manifest.json').read_text())
            frames=manifest['target_frames']
            for phase in ['sample','decode']:
                result=out/f'{phase}_result.json'
                if result.exists() and json.loads(result.read_text()).get('status')=='complete':
                    continue
                admit()
                cmd=[sys.executable,'experiments/h3_backbone/staged_bf16_guide.py',phase,'--comfy','/root/autodl-tmp/h3-w4a8-20261011/ComfyUI','--output',str(out),'--frames',str(frames)]
                if phase=='sample':
                    cmd+=['--model','/root/autodl-tmp/h3-qf-vdn-20261015/models/minimax_h3_fl2va_pruned_bf16.safetensors','--context',args.context,'--steps','49','--official-grid']
                else:
                    cmd+=['--vae','/root/autodl-tmp/h3-quality-repair-20261013/models/vae/minimax_h3_video_vae_fp16.safetensors','--decode-guide']
                status.update(case=case,phase=phase,command=cmd,epoch=time.time())
                status_path.write_text(json.dumps(status,indent=2)+'\n')
                print(json.dumps({'event':'stage','case':case,'phase':phase}),flush=True)
                run_monitored(cmd)
            status['completed'].append(case)
        status['status']='complete'
    except Exception as exc:
        status.update(status='stopped_or_failed',exception=repr(exc))
        raise
    finally:
        status_path.write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)


if __name__=='__main__':main()
