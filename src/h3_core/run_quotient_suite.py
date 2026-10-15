"""Foreground fixed projected-RGB suite; validation only, no test selection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43]);args=p.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True);snapshot=out/'protocol_source';snapshot.mkdir(exist_ok=True)
    state={'config':vars(args),'budget':{'shared_prior_actualRGB_steps':400,'actualRGB_steps_per_four_heads':400},'scope':'VAL-only structural pilot, no rawHRlatentloss, no new heldouttest claim','source_hashes':{},'jobs':[]}
    for name in ['train_quotient.py','nonlinear_memory.py','memory_adapter.py','h3_interface.py','observation_quotient.py','cache_quotient.py']:
        value=(Path('src/h3_core')/name).read_bytes();(snapshot/name).write_bytes(value);state['source_hashes'][name]=hashlib.sha256(value).hexdigest()
    for seed in args.seeds:
        dest=out/f'seed{seed}';dest.mkdir(exist_ok=True)
        command=[sys.executable,'-u','src/h3_core/train_quotient.py','--cache',args.cache,'--vae',args.vae,'--comfy',args.comfy,'--out',str(dest),'--seed',str(seed),'--prior-steps','400','--steps','400','--eval-splits','val']
        state['active_command']=command;(out/'suite_state.json').write_text(json.dumps(state,indent=2));print(json.dumps({'event':'start_seed','seed':seed,'command':command}),flush=True)
        with (dest/'run.log').open('w') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        state['jobs'].append({'seed':seed,'exit_code':code,'out':str(dest)});state['active_command']=None;(out/'suite_state.json').write_text(json.dumps(state,indent=2))
        if code:raise SystemExit(code)
    print(json.dumps({'event':'suite_complete','seeds':args.seeds}),flush=True)


if __name__=='__main__':main()
