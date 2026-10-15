"""Foreground postprocess after the fixed400 pilot exits: decode/audit/evaluate."""
import json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/h3_setup/native_sr_pilot400'

def admit():
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','');assert gpu.isdigit()
    for _ in range(4):
        free,util=map(int,subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        if util==0 and free>30000:return
        time.sleep(2)
    raise RuntimeError(f'GPU{gpu} no longer idle: free={free}MiB util={util}; no child launched')

def main():
    os.chdir(ROOT);assert json.loads((RUN/'LIVE_STATE.json').read_text())['status']=='complete'
    status={'status':'running','pid':os.getpid(),'host':'H100','gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'completed':[]};p=RUN/'postprocess_live.json'
    python=sys.executable
    jobs=[('decode',[python,'experiments/h3_backbone/decode_native_sr_pilot.py','--comfy','external/ComfyUI_H3','--vae','external/h3_assets/minimax_h3_video_vae_fp16.safetensors','--run',str(RUN),'--val','results/h3_setup/native_sr_val_pairs/pairs_manifest.json']),
          ('protocol_audit',[python,'experiments/h3_backbone/audit_native_sr_pilot.py','--run',str(RUN)]),
          ('fixed_validation_metrics',[python,'experiments/h3_backbone/evaluate_native_sr_pilot.py','--run',str(RUN),'--val','results/h3_setup/native_sr_val_pairs/pairs_manifest.json','--device','cuda'])]
    try:
        for phase,command in jobs:
            if phase!='protocol_audit':admit()
            status.update(phase=phase,command=command,epoch=time.time());p.write_text(json.dumps(status,indent=2)+'\n')
            env=dict(os.environ)
            if phase=='fixed_validation_metrics':env.update(PYTHONPATH=str(ROOT/'external/eval_deps'),TORCH_HOME=str(ROOT/'external/eval_torch_cache'))
            subprocess.run(command,env=env,check=True)
            status['completed'].append(phase)
        status['status']='complete'
    except Exception as exc:status.update(status='failed',exception=repr(exc));raise
    finally:
        status['epoch']=time.time();p.write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)

if __name__=='__main__':main()
