"""Add a control name to existing v2 decoder/evaluator; keep their math intact."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
operation=sys.argv.pop(1);assert operation in ['decode','evaluate']
source=ROOT/'experiments/h3_backbone'/('decode_native_sr_pilot.py' if operation=='decode' else 'evaluate_native_sr_pilot.py')
original=argparse.ArgumentParser.add_argument
def allow_control(self,*args,**kwargs):
 if args and args[0]=='--modes' and 'choices' in kwargs:kwargs['choices']=[*kwargs['choices'],'sampling_dc']
 return original(self,*args,**kwargs)
argparse.ArgumentParser.add_argument=allow_control
spec=importlib.util.spec_from_file_location('original_fixed_v2_'+operation,source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
try:module.main()
finally:argparse.ArgumentParser.add_argument=original
if operation=='evaluate':
 run=Path(sys.argv[sys.argv.index('--run')+1]);path=run/'evaluation/aggregate.json';d=json.loads(path.read_text())
 assert set(d['modes_evaluated'])=={'unadapted','sampling_dc'} and len(d['rows'])==32
 d.update(status='complete',scope='Two fixed held-outval sources x2contexts; frozen H3 sampling-inner DC baseline, no training/hyperparameterselection; rawandmatchedDC25',
          metric_math_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),control_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 path.write_text(json.dumps(d,indent=2)+'\n')
 for file in (run/'evaluation').glob('*/metrics.json'):
  item=json.loads(file.read_text());item['scope']=d['scope'];file.write_text(json.dumps(item,indent=2)+'\n')
