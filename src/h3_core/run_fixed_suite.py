"""Foreground-only fixed-budget suite after a complete, verified cache manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--span',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    expected=56;last=-1
    while True:
        try:manifest=json.loads((Path(args.cache)/'manifest.json').read_text())
        except (OSError,json.JSONDecodeError):manifest={'items':[]}
        count=len(manifest['items'])
        if count!=last:print(json.dumps({'event':'await_cache','sources_complete':count,'expected':expected}),flush=True);last=count
        if count==expected:
            if not manifest['config'].get('strict_fp32'):raise ValueError('Fixed suite requires the audited strict FP32 teacher cache')
            if any(not Path(item['path']).exists() for item in manifest['items']):raise ValueError('Manifest cache file absent')
            break
        if count>expected:raise ValueError('Unexpected source count')
        time.sleep(10)
    jobs=[('learned',17),('learned',29),('learned',43),('fixed_pca',17)]
    state={'config':vars(args),'budget':{'basis_steps':600,'latent_steps_per_head':400,'true_rgb_steps_per_head':100},'jobs':[],'scope':'A fixed VAE-memory research suite; not a completed streaming SR/H3-DiT method'}
    snapshot=out/'protocol_source';snapshot.mkdir(exist_ok=True);state['source_sha256']={}
    for name in ['train_nonlinear.py','nonlinear_memory.py','memory_adapter.py','h3_interface.py']:
        content=(Path('src/h3_core')/name).read_bytes();(snapshot/name).write_bytes(content);state['source_sha256'][name]=hashlib.sha256(content).hexdigest()
    (out/'suite_state.json').write_text(json.dumps(state,indent=2))
    for basis,seed in jobs:
        destination=out/f'{basis}_seed{seed}'
        command=[sys.executable,'-u','src/h3_core/train_nonlinear.py','--cache',args.cache,'--span',args.span,'--vae',args.vae,'--comfy',args.comfy,'--out',str(destination),'--seed',str(seed),'--basis-mode',basis,'--basis-steps','600','--latent-steps','400','--rgb-steps','100']
        print(json.dumps({'event':'run','basis':basis,'seed':seed,'command':command}),flush=True)
        destination.mkdir(parents=True,exist_ok=True)
        # Foreground child and a durable log, no detached/background process.
        with (destination/'run.log').open('w') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        state['jobs'].append({'basis':basis,'seed':seed,'exit_code':code,'output':str(destination)})
        (out/'suite_state.json').write_text(json.dumps(state,indent=2))
        if code:raise SystemExit(code)
    print(json.dumps({'event':'suite_complete','jobs':len(jobs)}),flush=True)


if __name__=='__main__':main()
