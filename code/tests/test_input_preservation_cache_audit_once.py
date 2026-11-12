"""Local stdlib runner boundaries; no real cache, Torch, subprocess or device."""
from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'run_input_preservation_cache_audit_once.py'
spec = importlib.util.spec_from_file_location('cache_audit_once', SOURCE)
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


class AuditRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); (self.root/'runs').mkdir()
        # Pin the real frozen auditor/supervisor in production. Fixtures use
        # harmless source bytes with explicit constants replaced only here.
        self.entry = self.root/R.ENTRY_NAME; self.entry.write_text('# fixture only\n')
        self.supervisor = self.root/'owned_process.py'; self.supervisor.write_text('# fixture supervisor only\n')
        self.patches = [patch.object(R,'ENTRY_SHA',R.sha(self.entry)),
                        patch.object(R,'SUPERVISOR_SHA',R.sha(self.supervisor))]
        for p in self.patches: p.start(); self.addCleanup(p.stop)
        self.path = self.root/'plan.json'; self.contract_path = self.root/'execution.json'
        self.producer_path = self.root/'producer.json'; self.producer_contract = self.root/'producer_execution.json'
        self.numerical = self.root/'numerical_admission.json'; self.launch = self.root/'launch_admission.json'
        self.plan = {'schema':R.SCHEMA,'root':str(self.root),'role':'train','physical_card':None,
            'resource':copy.deepcopy(R.LIMITS['train']),'supervision_source':str(self.supervisor),
            'entry':str(self.entry),'entry_sha256':R.sha(self.entry),'python':sys.executable,
            'execution_contract':str(self.contract_path),'run':str(self.root/'runs/fixture'),
            'out':str(self.root/'runs/fixture/output'),'report_file':'audit.json',
            'report_requires':R.report_requirements('train'),'admission':str(self.launch)}
        self.producer = {'schema':'rphi.input_preservation_cache.v1','role':'train','complete':True,
            'status':'cache_complete_pending_independent_audit_no_training_admission',
            'completed_rows':16000,'head_training_admitted':False,'original_history_headroom_remains_failed':True,
            'usable_for_training':True,'execution_contract':{'schema':'fixture.producer','role':'train'}}
        self.save()

    def write(self,path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False))

    def save(self):
        self.write(self.producer_contract,self.producer['execution_contract'])
        self.producer['execution_contract_file_sha256']=R.sha(self.producer_contract); self.write(self.producer_path,self.producer)
        role=self.plan['role'];c={'schema':'rphi.input_preservation_cache_audit_execution.v1','root':str(self.root),'role':role,
            'producer_manifest':{'path':str(self.producer_path),'sha256':R.sha(self.producer_path)},
            'producer_execution_contract':{'path':str(self.producer_contract),'sha256':R.sha(self.producer_contract)},
            'implementation_sha256':{str(self.entry):R.sha(self.entry)},'resources':{'wall_seconds':900,'CPU_cores':2,
                'RSS_limit_bytes':R.LIMITS[role]['rss_bytes'],'max_small_output_bytes':32<<20},
            'admission':{'path':str(self.numerical),'sha256':None},'description':'仅CPU'}
        self.write(self.numerical,{'schema':'rphi.input_preservation_cache_audit_admission.v1','approved':True,'role':role,
            'execution_sha256':R.canonical_sha({k:v for k,v in c.items() if k!='admission'}),
            'entry_sha256':self.plan['entry_sha256'],'out':self.plan['out']})
        c['admission']['sha256']=R.sha(self.numerical);self.write(self.contract_path,c)
        self.plan['pins']={str(p):R.sha(p) for p in (SOURCE,self.entry,self.supervisor,self.contract_path,
            self.producer_path,self.producer_contract,self.numerical)}
        self.plan['argv']=[sys.executable,str(self.entry),'--execution-contract',str(self.contract_path),'--out',self.plan['out'],'--execute']
        self.save_plan()

    def save_plan(self):
        self.write(self.path,self.plan)
        self.write(self.launch,{'schema':'rphi.input_preservation_cache_audit_launch_admission.v1','approved':True,
            'scope':'one_completed_cache_CPU_audit_no_training','role':self.plan['role'],'plan_sha256':R.sha(self.path),
            'runner_sha256':R.sha(SOURCE),'execution_contract_sha256':R.sha(self.contract_path),
            'entry_sha256':self.plan['entry_sha256'],'physical_card':None})

    def terminal(self):
        out=Path(self.plan['out']);out.mkdir(parents=True)
        report={**R.report_requirements(self.plan['role']),'producer_manifest':R.read(self.contract_path)['producer_manifest'],
            'runtime':{'device':'cpu','threads':2,'torch_npu_imported_at_end':False,'versions':{'torch_npu':None}}}
        self.write(out/'audit.json',report); archive=out/'evidence.tar.gz';archive.write_bytes(b'fixture small archive')
        self.write(out/'archive_receipt.json',{'path':str(archive),'sha256':R.sha(archive),'bytes':archive.stat().st_size,
            'payload_files':5,'NPZ_or_checkpoint_arrays_included':False})
        return report

    def test_valid_train_and_dev_have_fixed_separate_budgets(self):
        self.assertEqual(R.validate(self.path)['resource']['rss_bytes'],32<<30)
        self.plan.update(role='dev',resource=copy.deepcopy(R.LIMITS['dev']),report_requires=R.report_requirements('dev'))
        self.producer.update(role='dev',completed_rows=2880,usable_for_training=False)
        self.producer['execution_contract']['role']='dev';self.save()
        self.assertEqual(R.validate(self.path)['resource']['rss_bytes'],4<<30)

    def test_metadata_only_or_incomplete_producer_rejected(self):
        for key,value in [('complete',False),('status','metadata_inspected_no_model_no_cache'),('completed_rows',15999),('usable_for_training',False)]:
            before=self.producer[key];self.producer[key]=value;self.save()
            with self.assertRaises(ValueError):R.validate(self.path)
            self.producer[key]=before

    def test_launch_and_numerical_permissions_exact(self):
        a=R.read(self.launch);a['scope']='training';self.write(self.launch,a)
        with self.assertRaises(ValueError):R.validate(self.path)
        self.save();a=R.read(self.numerical);a['out']=str(self.root/'other');self.write(self.numerical,a)
        with self.assertRaises(ValueError):R.validate(self.path)

    def test_no_extra_argv_card_or_budget_expansion(self):
        for key,value in [('physical_card',1),('resource',{**R.LIMITS['train'],'wall_seconds':901}),('argv',self.plan['argv']+['--role','dev'])]:
            before=self.plan[key];self.plan[key]=value;self.save_plan()
            with self.assertRaises(ValueError):R.validate(self.path)
            self.plan[key]=before

    def test_pin_mutation_rejected_and_unicode_hash_matches_contract(self):
        self.assertEqual(R.canonical_sha({'字':'值'}),R.hashlib.sha256(b'{"\xe5\xad\x97":"\xe5\x80\xbc"}').hexdigest())
        self.entry.write_text('# changed')
        with self.assertRaises(ValueError):R.validate(self.path)

    def test_terminal_exact_counts_types_no_NPU_and_archive_sha(self):
        original=self.terminal();R.verify_terminal(self.plan);out=Path(self.plan['out'])
        for key,value in [('complete',1),('actual_rows',15999),('head_training_admitted',True)]:
            bad=copy.deepcopy(original);bad[key]=value;self.write(out/'audit.json',bad)
            with self.assertRaises(ValueError):R.verify_terminal(self.plan)
        bad=copy.deepcopy(original);bad['runtime']['torch_npu_imported_at_end']=True;self.write(out/'audit.json',bad)
        with self.assertRaises(ValueError):R.verify_terminal(self.plan)
        self.write(out/'audit.json',original);(out/'evidence.tar.gz').write_bytes(b'changed')
        with self.assertRaises(ValueError):R.verify_terminal(self.plan)

    def fake_run(self,mode='complete'):
        calls=[]
        def supervise(command,root,env,log,status,status_path,**kwargs):
            calls.append((command,env,kwargs));status.update(producer_started=True,state='worker_completed')
            if mode=='complete':self.terminal()
            elif mode=='worker_failure':status.update(state='failed',complete=False,error={'type':'MemoryError'})
            elif mode=='mutate':self.terminal();self.entry.write_text('# changed during run')
            else:raise TimeoutError('fixture')
        sup=SimpleNamespace(signal_guard=nullcontext,supervise=supervise,write=self.write)
        with patch.object(R,'load_module',return_value=sup) as load,patch.object(R,'resources',return_value={}), \
             patch.object(R.os,'sched_getaffinity',return_value={2,3,4},create=True), \
             patch.object(R.os,'sched_setaffinity',create=True),patch.object(R.os,'nice'):
            result=R.run(self.path)
        self.assertEqual(load.call_count,1)
        return result,calls

    def test_one_CPU_audit_and_fresh_directory_refuses_relaunch(self):
        result,calls=self.fake_run();self.assertTrue(result['complete']);self.assertEqual(len(calls),1)
        env=calls[0][1];self.assertEqual(env['TORCH_DEVICE_BACKEND_AUTOLOAD'],'0')
        self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'],'');self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(calls[0][0],self.plan['argv']);self.assertEqual(calls[0][2],{'wall':900,'rss_limit':32<<30,'grace':30})
        with self.assertRaises(FileExistsError):R.run(self.path)

    def test_owned_supervisor_failure_remains_failed(self):
        result,calls=self.fake_run('worker_failure');self.assertFalse(result['complete'])
        self.assertEqual(result['state'],'failed');self.assertEqual(len(calls),1)

    def test_exception_after_worker_start_remains_failed(self):
        result,calls=self.fake_run('exception');self.assertFalse(result['complete'])
        self.assertEqual(result['state'],'failed');self.assertEqual(result['error']['type'],'TimeoutError')

    def test_source_mutation_after_success_revokes_acceptance(self):
        result,calls=self.fake_run('mutate');self.assertFalse(result['complete'])
        self.assertFalse(result['pins_unchanged']);self.assertEqual(result['state'],'failed')


if __name__=='__main__':unittest.main(verbosity=2)
