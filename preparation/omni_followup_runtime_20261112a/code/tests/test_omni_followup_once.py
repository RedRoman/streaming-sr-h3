import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'run_omni_followup_once.py'
spec = importlib.util.spec_from_file_location('p1_once', SOURCE)
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


class OnceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'code').mkdir(); (self.root/'runs').mkdir()
        for name in ('code/takeover_queue.py', 'code/run_takeover_command.sh', 'env.sh',
                     'supervision.py', 'execution.json', 'check_h3_adamw_saved_continuation.py'):
            (self.root/name).write_text('{}')
        (self.root/'execution.json').write_text(json.dumps({'root':str(self.root)}))
        paths = [SOURCE, *(self.root/x for x in ('code/takeover_queue.py', 'code/run_takeover_command.sh',
                 'env.sh', 'supervision.py', 'execution.json', 'check_h3_adamw_saved_continuation.py'))]
        self.plan = {'schema': R.SCHEMA, 'root': str(self.root), 'kind': 'saved_adamw_continuation',
            'resource': {'wall_seconds': 3600, 'CPU_threads': 4, 'rss_bytes': 128 << 30, 'grace_seconds': 30},
            'physical_card': 2, 'pins': {str(p): R.sha(p) for p in paths},
            'supervision_source': str(self.root/'supervision.py'),
            'execution_contract': str(self.root/'execution.json'), 'run': str(self.root/'runs/test'),
            'out': str(self.root/'runs/test/output'), 'report_file': 'report.json',
            'report_requires': {'complete': True}, 'admission': str(self.root/'admission.json')}
        self.plan['argv'] = ['/usr/bin/python3', str(self.root/'check_h3_adamw_saved_continuation.py'), '--execution-contract',
                             self.plan['execution_contract'], '--out', self.plan['out'], '--execute']
        self.accept = self.root/'accepted.json'; self.accept.write_text(json.dumps({'accepted':True}))
        self.plan['pins'][str(self.accept)] = R.sha(self.accept)
        self.plan['required_json_values'] = [{'path':str(self.accept),'keys':['accepted'],'value':True}]
        self.plan['pretrained_manifest']=str(self.root/'pretrained.json')
        Path(self.plan['pretrained_manifest']).write_text('{}')
        self.plan['pins'][self.plan['pretrained_manifest']]=R.sha(self.plan['pretrained_manifest'])
        self.plan['argv'][2:2]=['--pretrained-manifest',self.plan['pretrained_manifest']]
        self.path = self.root/'plan.json'; self.save()

    def save(self):
        self.path.write_text(json.dumps(self.plan))
        (self.root/'admission.json').write_text(json.dumps({'schema': 'omni.followup_once_admission.v1',
            'approved': True, 'plan_sha256': R.sha(self.path), 'runner_sha256': R.sha(SOURCE),
            'kind': self.plan['kind'], 'physical_card': self.plan['physical_card']}))

    def test_exact_valid_plan(self):
        self.assertEqual(R.validate(self.path), self.plan)

    def test_tamper_and_unbounded_scope_rejected(self):
        (self.root/'check_h3_adamw_saved_continuation.py').write_text('changed')
        with self.assertRaises(ValueError): R.validate(self.path)
        (self.root/'check_h3_adamw_saved_continuation.py').write_text('{}')
        self.plan['resource']['wall_seconds'] = 3601; self.save()
        with self.assertRaises(ValueError): R.validate(self.path)

    def test_admission_binds_card(self):
        a = R.read(self.root/'admission.json'); a['physical_card'] = 1
        (self.root/'admission.json').write_text(json.dumps(a))
        with self.assertRaises(ValueError): R.validate(self.path)

    def test_missing_or_false_prerequisite_blocks(self):
        accepted = {'accepted':False}
        self.accept.write_text(json.dumps(accepted))
        self.plan['pins'][str(self.accept)] = R.sha(self.accept); self.save()
        with self.assertRaises(ValueError): R.validate(self.path)

    def test_extra_argument_and_dev_role_substitution_rejected(self):
        self.plan['argv'].append('--extra-model-call'); self.save()
        with self.assertRaises(ValueError): R.validate(self.path)
        self.plan['argv'].pop()
        self.plan['kind']='input_preservation_dev_cache'
        self.plan['physical_card']=None
        self.plan['resource']={'wall_seconds':900,'CPU_threads':2,'rss_bytes':4<<30,'grace_seconds':30}
        entry=self.root/'export_input_preservation_cache.py';entry.write_text('{}')
        self.plan['pins'][str(entry)]=R.sha(entry)
        self.plan['argv']=['/usr/bin/python3',str(entry),'--role','train','--execution-contract',self.plan['execution_contract'],'--out',self.plan['out'],'--execute'];self.save()
        with self.assertRaises(ValueError):R.validate(self.path)
        self.plan['argv'][3]='dev';self.save()
        self.assertEqual(R.validate(self.path)['physical_card'],None)

    def test_cpu_job_never_loads_device_inspector(self):
        from contextlib import nullcontext
        self.plan['kind']='saved_adamw_cpu_audit';self.plan['physical_card']=None
        self.plan['resource']={'wall_seconds':900,'CPU_threads':2,'rss_bytes':4<<30,'grace_seconds':30}
        entry=self.root/'audit_h3_saved_adamw_partial.py';entry.write_text('{}')
        self.plan['pins'][str(entry)]=R.sha(entry);self.plan['argv'][1]=str(entry);del self.plan['argv'][2:4];self.save()
        calls=[]
        def supervise(command,root,env,log,status,status_path,**kw):
            calls.append((command,env,kw));status['state']='worker_completed'
            out=Path(self.plan['out']);out.mkdir();(out/'report.json').write_text('{"complete":true}')
        sup=SimpleNamespace(signal_guard=nullcontext,write=lambda p,v:Path(p).write_text(json.dumps(v)),supervise=supervise)
        with patch.object(R,'load_module',return_value=sup) as load,patch.object(R,'resources',return_value={}), \
             patch.object(R.os,'sched_getaffinity',return_value={0,1,2,3},create=True), \
             patch.object(R.os,'sched_setaffinity',create=True),patch.object(R.os,'nice'):
            status=R.run(self.path)
        self.assertEqual(load.call_count,1);self.assertTrue(status['complete']);self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1]['ASCEND_RT_VISIBLE_DEVICES'],'')
        self.assertEqual(calls[0][1]['TORCH_DEVICE_BACKEND_AUTOLOAD'],'0')
        self.assertEqual(calls[0][2]['wall'],900)

    def test_busy_device_never_launches(self):
        from contextlib import nullcontext
        calls = []
        def write(p, v): Path(p).write_text(json.dumps(v))
        s = SimpleNamespace(signal_guard=nullcontext, write=write, supervise=lambda *a, **k: calls.append(a))
        q = SimpleNamespace(inspect=lambda: {2: {'pids': [123]}}, available=lambda *a: False)
        with patch.object(R, 'load_module', side_effect=[s, q]), \
             patch.object(R, 'resources', return_value={}), \
             patch.object(R.os, 'sched_getaffinity', return_value={0,1,2,3}, create=True), \
             patch.object(R.os, 'sched_setaffinity', create=True):
            status = R.run(self.path)
        self.assertEqual(status['state'], 'not_launched'); self.assertFalse(status['complete'])
        self.assertFalse(calls)
        with self.assertRaises(FileExistsError): R.run(self.path)

    def test_second_check_occupied_never_launches(self):
        from contextlib import nullcontext
        calls=[]
        sup=SimpleNamespace(signal_guard=nullcontext,write=lambda p,v:Path(p).write_text(json.dumps(v)),supervise=lambda *a,**k:calls.append(a))
        q=SimpleNamespace(inspect=lambda:{2:{}},available=__import__('unittest.mock',fromlist=['Mock']).Mock(side_effect=[True,False]))
        with patch.object(R,'load_module',side_effect=[sup,q]),patch.object(R,'resources',return_value={}), \
             patch.object(R,'LOCK_DIR',self.root/'locks'),patch.object(R.time,'sleep') as sleep, \
             patch.object(R.os,'sched_getaffinity',return_value={0,1,2,3},create=True), \
             patch.object(R.os,'sched_setaffinity',create=True):
            status=R.run(self.path)
        sleep.assert_called_once_with(15)
        self.assertEqual(q.available.call_count,2);self.assertEqual(status['state'],'not_launched');self.assertFalse(calls)

    def test_shared_card_lock_busy_never_launches(self):
        from contextlib import nullcontext
        calls=[]
        sup=SimpleNamespace(signal_guard=nullcontext,write=lambda p,v:Path(p).write_text(json.dumps(v)),supervise=lambda *a,**k:calls.append(a))
        q=SimpleNamespace(inspect=lambda:{2:{}},available=lambda *a:True)
        with patch.object(R,'load_module',side_effect=[sup,q]),patch.object(R,'resources',return_value={}), \
             patch.object(R,'LOCK_DIR',self.root/'locks'),patch.object(R.fcntl,'flock',side_effect=[None,BlockingIOError('busy')]), \
             patch.object(R.os,'sched_getaffinity',return_value={0,1,2,3},create=True), \
             patch.object(R.os,'sched_setaffinity',create=True):
            status=R.run(self.path)
        self.assertEqual(status['state'],'not_launched');self.assertFalse(calls)


if __name__ == '__main__': unittest.main()
