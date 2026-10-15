"""Foreground, resumable execution of untouched official H3 proposal stages.

Each subprocess remains attached to this foreground runner. Its output is sent
to a project-local file to avoid making CUDA execution depend on SSH stdout.
No training, new observations, or HR files are read by this inference runner.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def complete(path):
    try:return json.loads(path.read_text()).get('status')=='complete'
    except (OSError,json.JSONDecodeError):return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--comfy',required=True);p.add_argument('--model',required=True);p.add_argument('--vae',required=True);p.add_argument('--context',required=True);p.add_argument('--memory-fraction',type=float,default=.9);p.add_argument('--reserve-vram',type=float,default=8.);args=p.parse_args()
    out=Path(args.out);manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['distinct_observations']==1 and manifest['latest_observed_index']==0
    assert manifest['native_repeated_context_frames']==22
    runner=Path('experiments/h3_backbone/staged_bf16_guide.py')
    state={'status':'running','scope':'TrueBF16H3 fixedinitialproposal; not learnedmemory efficacy','config':vars(args),'seeds':{}};began=time.monotonic()
    def save():
        state['seconds']=time.monotonic()-began;(out/'run_state.json').write_text(json.dumps(state,indent=2))
    for seed in manifest['seeds']:
        folder=out/f'seed{seed}';entry={};state['seeds'][str(seed)]=entry
        common=[sys.executable,str(runner),'PHASE','--comfy',args.comfy,'--output',str(folder),'--frames','22','--width','256','--height','256','--memory-fraction',str(args.memory_fraction),'--reserve-vram',str(args.reserve_vram)]
        for phase in ['sample','decode']:
            record=folder/f'{phase}_result.json'
            if complete(record):entry[phase]={'status':'already_complete'};continue
            cmd=common.copy();cmd[2]=phase
            if phase=='sample':cmd+=['--model',args.model,'--context',args.context,'--steps','49','--official-grid','--seed',str(seed)]
            else:cmd+=['--vae',args.vae,'--decode-guide']
            entry[phase]={'status':'running','command':cmd,'log':str(folder/f'{phase}.log')};save()
            with (folder/f'{phase}.log').open('w') as log:
                done=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
            entry[phase]['exit_code']=done.returncode
            if done.returncode or not complete(record):entry[phase]['status']='failed';state['status']='failed';save();raise RuntimeError(f'{phase} seed{seed}: see {folder/phase}.log')
            entry[phase]['status']='complete';save();print(json.dumps({'event':'proposal_phase_complete','seed':seed,'phase':phase,'seconds':time.monotonic()-began}),flush=True)
        sample=json.loads((folder/'sample_result.json').read_text());decode=json.loads((folder/'decode_result.json').read_text())
        assert len(sample['forward_events'])==49 and decode['output_shape']==[22,256,256,3]
        entry['verified_full50block_calls']=len(sample['forward_events']);entry['status']='complete';save()
    cmd=[sys.executable,'src/h3_core/dit_proposal.py','collect','--out',str(out)]
    with (out/'collect_nohr.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    state['status']='complete';save();print(json.dumps({'event':'proposal_run_complete','seeds':manifest['seeds'],'seconds':state['seconds']}),flush=True)


if __name__=='__main__':main()
