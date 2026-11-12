import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'run_history_utility_once.py'
spec = importlib.util.spec_from_file_location('p1_once', SOURCE)
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


class OnceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'code').mkdir(); (self.root/'runs').mkdir()
        for name in ('code/takeover_queue.py', 'code/run_takeover_command.sh', 'env.sh',
                     'supervision.py', 'execution.json', 'producer.py'):
            (self.root/name).write_text('{}')
        paths = [SOURCE, *(self.root/x for x in ('code/takeover_queue.py', 'code/run_takeover_command.sh',
                 'env.sh', 'supervision.py', 'execution.json', 'producer.py'))]
        self.plan = {'schema': R.SCHEMA, 'root': str(self.root), 'kind': 'backend_smoke',
            'resource': {'wall_seconds': 900, 'CPU_threads': 4, 'rss_bytes': 32 << 30, 'grace_seconds': 30},
            'physical_card': 2, 'pins': {str(p): R.sha(p) for p in paths},
            'supervision_source': str(self.root/'supervision.py'),
            'execution_contract': str(self.root/'execution.json'), 'run': str(self.root/'runs/test'),
            'out': str(self.root/'runs/test/output'), 'report_file': 'report.json',
            'report_requires': {'complete': True}, 'admission': str(self.root/'admission.json')}
        self.plan['argv'] = ['/usr/bin/python3', str(self.root/'producer.py'), '--execution-contract',
                             self.plan['execution_contract'], '--out', self.plan['out'], '--execute']
        self.path = self.root/'plan.json'; self.save()

    def save(self):
        self.path.write_text(json.dumps(self.plan))
        (self.root/'admission.json').write_text(json.dumps({'schema': 'rphi.history_utility_once_admission.v1',
            'approved': True, 'plan_sha256': R.sha(self.path), 'runner_sha256': R.sha(SOURCE),
            'kind': self.plan['kind'], 'physical_card': self.plan['physical_card']}))

    def test_exact_valid_plan(self):
        self.assertEqual(R.validate(self.path), self.plan)

    def test_tamper_and_unbounded_scope_rejected(self):
        (self.root/'producer.py').write_text('changed')
        with self.assertRaises(ValueError): R.validate(self.path)
        (self.root/'producer.py').write_text('{}')
        self.plan['resource']['wall_seconds'] = 901; self.save()
        with self.assertRaises(ValueError): R.validate(self.path)

    def test_admission_binds_card(self):
        a = R.read(self.root/'admission.json'); a['physical_card'] = 1
        (self.root/'admission.json').write_text(json.dumps(a))
        with self.assertRaises(ValueError): R.validate(self.path)

    def test_teacher_requires_actual_backend_acceptance(self):
        self.plan['kind'] = 'teacher1000'; self.plan['accepted_backend_smoke'] = {
            'path': str(self.root/'execution.json'), 'sha256': R.sha(self.root/'execution.json')}
        self.save()
        with self.assertRaises((ValueError, KeyError)): R.validate(self.path)

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


if __name__ == '__main__': unittest.main()
