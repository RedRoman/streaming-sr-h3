"""Supervisor unit tests with fake workers only; no model/data access."""
import importlib.util,json
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('guard',Path(__file__).with_name('run_once.py'));guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

class GuardTests(unittest.TestCase):
    def test_active_budget_is_shared_and_never_two_300_second_runs(self):
        self.assertEqual(guard.phase_budget(False),(90.,0.))
        inspect={'complete':True,'phase':'inspect','returncode':0,'active_worker_seconds':17.25}
        self.assertEqual(guard.phase_budget(True,inspect),(282.75,17.25))
        for wrong in (-1,300,301,float('nan')):
            with self.assertRaises(ValueError):guard.phase_budget(True,{**inspect,'active_worker_seconds':wrong})
        with self.assertRaises(ValueError):guard.phase_budget(True,{**inspect,'complete':False})

    def test_stop_owned_uses_only_worker_group_and_declared_grace(self):
        class Process:
            pid=111;returncode=None
            def poll(self):return self.returncode
            def wait(self,timeout):self.returncode=-9
        process=Process()
        with patch.object(guard.os,'killpg') as kill,patch.object(guard,'group_rows',return_value=[{'pid':111,'rss_bytes':1}]),patch.object(guard.time,'monotonic',side_effect=[0.,11.]):
            guard.stop_owned(process,10)
        self.assertEqual(kill.call_args_list[0].args,(111,guard.signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args,(111,guard.signal.SIGKILL))
        self.assertEqual(process.returncode,-9)

    def fake_run(self,td,usage,clock):
        out=Path(td);root=out/'project';root.mkdir();runs=root/'runs';runs.mkdir()
        config={'root':str(root),'out':str(runs/'actual'),'inspection_out':str(runs/'inspect'),'supervisor_out':str(runs/'supervisor'),
            'schema':'rphi.history_utility_one_step_cpu_supervisor.v1','pins':{},'python':'python','entry':'entry','execution_contract':'contract'}
        class Process:
            pid=123;returncode=None
            def poll(self):return self.returncode
        process=Process();old_read=Path.read_text
        def read_text(path,*a,**k):return 'MemAvailable: 9000000 kB\n' if str(path)=='/proc/meminfo' else old_read(path,*a,**k)
        def stop(p,grace):self.assertIs(p,process);self.assertEqual(grace,10);p.returncode=-15
        with patch.object(Path,'read_text',read_text),patch.object(guard.os,'sched_getaffinity',return_value={0,1,2},create=True),patch.object(guard.os,'sched_setaffinity',create=True),patch.object(guard.subprocess,'Popen',return_value=process),patch.object(guard,'group_rows',return_value=[{'pid':123,'rss_bytes':usage}]),patch.object(guard,'stop_owned',side_effect=stop) as stopped,patch.object(guard.time,'monotonic',side_effect=clock):
            result=guard.run(config,'freeze',False)
        self.assertEqual(stopped.call_count,1);self.assertFalse(result['complete']);self.assertEqual(result['phase'],'inspect')
        self.assertTrue((runs/'supervisor/inspect/status.json').is_file())
        return result

    def test_RSS_guard_preserves_failure_in_inspection(self):
        with tempfile.TemporaryDirectory() as td:result=self.fake_run(td,(4<<30)+1,[0.,.2])
        self.assertEqual(result['error']['type'],'MemoryError')
        self.assertGreater(result['peak_owned_group_RSS_bytes'],4<<30)

    def test_wall_guard_preserves_failure_in_inspection(self):
        with tempfile.TemporaryDirectory() as td:result=self.fake_run(td,1,[0.,91.,91.1])
        self.assertEqual(result['error']['type'],'TimeoutError');self.assertEqual(result['returncode'],-15)

    def test_existing_output_never_spawns(self):
        with tempfile.TemporaryDirectory() as td,patch.object(guard.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(ValueError,'already exists'):guard.run({'supervisor_out':td,'inspection_out':td},'x',False)
            popen.assert_not_called()

if __name__=='__main__':unittest.main()
