"""Foreground, checkpointed execution of the already-frozen matched DC pilot."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path('results/h3_core/operator_function_dc_v1')
raw = Path('results/h3_core/operator_function_v1')
common = ['--comfy', 'external/ComfyUI_H3', '--vae',
          'external/h3_assets/minimax_h3_video_vae_fp16.safetensors']
stages = [
    ('DCcarrier_rebase', ['src/h3_core/operator_function_dc.py', 'rebase', '--root', str(root), '--raw', str(raw), *common]),
    ('DCfamily_assemble', ['src/h3_core/operator_function_data.py', 'assemble', '--root', str(root), '--old', str(raw), *common]),
    ('DCformal900', ['src/h3_core/train_operator_function.py', '--root', str(root), '--out', str(root/'fixed_seed17'), '--steps', '300', '--seed', '17', *common]),
    ('RAWfixed_compression_diagnostic', ['src/h3_core/diagnose_operator_compression.py', '--root', str(raw), *common]),
    ('DCfixed_compression_diagnostic', ['src/h3_core/diagnose_operator_compression.py', '--root', str(root), *common]),
    ('sourcewise_matched_summary', ['src/h3_core/summarize_operator_function.py', '--roots', str(raw), str(root), '--comparison', 'results/h3_core/operator_function_matched_comparison']),
]
state = dict(status='running', pid=os.getpid(), GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
             began=time.time(), stages=[])
path = root/'pipeline_state.json'
for name, arguments in stages:
    state['current_stage'] = name
    state['command'] = [sys.executable, *arguments]
    path.write_text(json.dumps(state, indent=2))
    began = time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        completed = subprocess.run([sys.executable, *arguments], stdout=log, stderr=subprocess.STDOUT)
    state['stages'].append(dict(stage=name, exit_code=completed.returncode, seconds=time.monotonic()-began))
    path.write_text(json.dumps(state, indent=2))
    if completed.returncode:
        state['status'] = 'failed'
        path.write_text(json.dumps(state, indent=2))
        sys.exit(completed.returncode)
state.update(status='complete', current_stage=None, seconds=time.time()-state['began'])
path.write_text(json.dumps(state, indent=2))
