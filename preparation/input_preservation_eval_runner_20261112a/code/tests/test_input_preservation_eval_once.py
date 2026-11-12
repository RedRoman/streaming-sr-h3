"""Seven stdlib metadata/terminal fixtures, no head/LPIPS/cache/device run."""
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
SOURCE=Path(__file__).resolve().parents[1]/'run_input_preservation_eval_once.py'
spec=importlib.util.spec_from_file_location('eval_once',SOURCE);R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


class EvalRunner(unittest.TestCase):
    def write(self,p,v):Path(p).write_text(json.dumps(v))
    def item(self,p):return {'path':str(p),'sha256':R.sha(p)}
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();(self.root/'runs').mkdir()
        self.entry=self.root/'eval_input_preservation_heads.py';self.entry.write_text('#fixture')
        self.sup=self.root/'owned.py';self.sup.write_text('#fixture supervisor')
        for obj,key,value in [(R,'ENTRY_SHA',R.sha(self.entry)),(R.B,'SUPERVISOR_SHA',R.sha(self.sup))]:
            p=patch.object(obj,key,value);p.start();self.addCleanup(p.stop)
        self.cp=self.root/'execution.json';self.ap=self.root/'numeric.json';self.pp=self.root/'plan.json';self.lp=self.root/'launch.json'
        files={}
        for name in ('proposal','screen','train_cache_manifest','train_cache_audit','dev_cache_manifest','dev_cache_audit','matched_checkpoint','shuffled_checkpoint'):
            p=self.root/(name+'.json');self.write(p,{'fixture':name});files[name]=self.item(p)
        weights=[]
        for role in ('alexnet','linear'):
            p=self.root/(role+'.weights');p.write_bytes(role.encode());weights.append({'role':role,**self.item(p)})
        ready={'schema':'rphi.input_preservation_metric_readiness.v1','complete':True,'frozen':True,'evaluator':self.item(self.entry),
            'implementation_sha256':{str(self.entry):R.sha(self.entry)},'LPIPS':{'available':True,'network':'alex','version':'0.1','downloads':False,'weights':weights},
            'runtime':{'expected_versions':{'torch_npu':None}}}
        self.rp=self.root/'readiness.json';self.write(self.rp,ready);files['metric_readiness']=self.item(self.rp)
        self.c={'schema':'rphi.input_preservation_heads_evaluation_execution.v1','root':str(self.root),
            'resources':{'wall_seconds':1800,'CPU_cores':2,'RSS_limit_bytes':4<<30,'max_output_bytes':256<<20},
            'files':files,'implementation_sha256':{str(self.entry):R.sha(self.entry)},'admission':{'path':str(self.ap),'sha256':None}}
        self.p={'schema':R.SCHEMA,'root':str(self.root),'resource':R.RESOURCE,'physical_card':None,
            'run':str(self.root/'runs/eval'),'out':str(self.root/'runs/eval/output'),'python':sys.executable,
            'entry':str(self.entry),'supervision_source':str(self.sup),'execution_contract':str(self.cp),'admission':str(self.lp),
            'training_reports':{},'training_execution_contracts':{}}
        common={k:v for k,v in files.items() if not k.endswith('checkpoint')}
        self.reports={}
        for arm in R.ARMS:
            tc={'schema':'rphi.input_preservation_head_training_execution.v1','mode':'research','arm':arm,'steps':1000,'batch_size':16,
                'implementation_sha256':{'/fixture/train_input_preservation_heads.py':R.TRAINER_SHA},'files':common,'admission':None}
            path=self.root/(arm+'_contract.json');self.write(path,tc);self.p['training_execution_contracts'][arm]=self.item(path)
            report={'schema':'rphi.input_preservation_head_training.v1','complete':True,'mode':'research','arm':arm,
                'status':'final_head_ready_for_bound_evaluation','completed_step':1000,'row_cursor':16000,'strict_checkpoint_reload':True,
                'NPU_calls':0,'new_restorer_forwards':0,'dev_label_arrays_read':0,'source_NPZ_reads':0,'resumed':False,'quality_accepted':False,
                'counts':{k:1000 for k in ('forward_attempts','forwards_completed','backward_attempts','backwards_completed','optimizer_attempts','optimizer_updates')},
                'execution_payload_sha256':R.semantic({k:v for k,v in tc.items() if k!='admission'}),'initial_state_sha256':'sameinit',
                'final_state_sha256':'final_'+arm,'binding':{'files':common},
                'checkpoint':{**files[arm+'_checkpoint'],'artifact_role':'research_final_head','model_sha256':'final_'+arm}}
            path=self.root/(arm+'_report.json');self.write(path,report);self.p['training_reports'][arm]=self.item(path);self.reports[arm]=report
        self.extra=[Path(w['path']) for w in weights];self.save()

    def save(self):
        self.write(self.ap,{'schema':'rphi.input_preservation_heads_evaluation_admission.v1','approved':True,
            'execution_sha256':R.semantic({k:v for k,v in self.c.items() if k!='admission'}),'entry_sha256':R.ENTRY_SHA,'out':self.p['out']})
        self.c['admission']['sha256']=R.sha(self.ap);self.write(self.cp,self.c)
        items=[SOURCE,R._base,self.entry,self.sup,self.cp,self.ap,*self.extra,*[Path(b['path']) for b in self.c['files'].values()],
            *[Path(b['path']) for b in self.p['training_reports'].values()],*[Path(b['path']) for b in self.p['training_execution_contracts'].values()]]
        self.p['pins']={str(p):R.sha(p) for p in items};self.p['argv']=[sys.executable,str(self.entry),'--execution-contract',str(self.cp),'--out',self.p['out'],'--execute'];self.save_plan()

    def save_plan(self):
        self.write(self.pp,self.p);self.write(self.lp,{'schema':'rphi.input_preservation_evaluation_launch_admission.v1','approved':True,
            'scope':'one_CPU_fixed_development_evaluation_no_training','plan_sha256':R.sha(self.pp),'runner_sha256':R.sha(SOURCE),
            'execution_contract_sha256':R.sha(self.cp),'physical_card':None})

    def terminal(self,verdict='failed_dev_screen'):
        out=Path(self.p['out']);out.mkdir(parents=True);r={'schema':'rphi.input_preservation_heads_evaluation.v1','complete':True,
            'status':'complete_fixed_development_screen','completed_rows':2880,'original_history_headroom_remains_failed':True,
            'head_or_restorer_training_admitted':False,'quality_proven':False,'LPIPS_frozen_exact':True,'counts':dict(R.COUNTS),
            'observed_model_invocations':{'matched':180,'shuffled':180,'LPIPS':720},'execution_contract':self.c,
            'execution_contract_file_sha256':R.sha(self.cp),'runtime':{'torch_npu':None},
            'lpips':{'available':True,'network':'alex','device':'cpu','downloads':False},
            'frozen_checks':{arm:{'state_dict_exact':True,'versions_exact':True,'eval_frozen_no_grads':True} for arm in R.ARMS},
            'checkpoints':{arm:{'file':self.c['files'][arm+'_checkpoint'],'completed_step':1000} for arm in R.ARMS},'temporal':{'pairs':2850},
            'screen':{'status':verdict,'required_metric_coverage_complete':True,'quality_proven':False,'independent_test':False,
                'automatic_training_or_fullframe_admission':False}}
        for name,n in [('records',2880),('pairs',2850),('batches',180)]:
            p=out/(name+'.jsonl');p.write_text('{}\n'*n);r[name]={**self.item(p),'bytes':p.stat().st_size}
        self.write(out/'report.json',r);return r

    def test_valid_exact_plan_requires_two_final1000(self):
        R.validate(self.pp);self.reports['matched']['completed_step']=999
        q=Path(self.p['training_reports']['matched']['path']);self.write(q,self.reports['matched']);self.p['training_reports']['matched']=self.item(q);self.save()
        with self.assertRaises(ValueError):R.validate(self.pp)

    def test_unready_metric_and_changed_source_fail(self):
        d=R.read(self.rp);d['complete']=False;self.write(self.rp,d);self.c['files']['metric_readiness']=self.item(self.rp);self.save()
        with self.assertRaises(ValueError):R.validate(self.pp)
        self.entry.write_text('#different')
        with self.assertRaises(ValueError):R.validate(self.pp)

    def test_extra_argv_card_or_wall_expansion_rejected(self):
        for k,v in [('argv',self.p['argv']+['--repeat']),('physical_card',1),('resource',{**R.RESOURCE,'wall_seconds':1801})]:
            old=self.p[k];self.p[k]=v;self.save_plan()
            with self.assertRaises(ValueError):R.validate(self.pp)
            self.p[k]=old

    def test_quality_failure_and_pending_are_completed_execution(self):
        r=self.terminal();self.assertEqual(R.terminal(self.p)['scientific_verdict'],'failed_dev_screen')
        r['screen']['status']='pending_review';self.write(Path(self.p['out'])/'report.json',r)
        self.assertEqual(R.terminal(self.p)['scientific_verdict'],'pending_review')

    def test_missing_counts_or_artifact_rows_rejected(self):
        r=self.terminal();r['counts']['LPIPS_rows_completed']=11519;self.write(Path(self.p['out'])/'report.json',r)
        with self.assertRaises(ValueError):R.terminal(self.p)
        r['counts']=dict(R.COUNTS);q=Path(r['pairs']['path']);q.write_text('{}\n'*2849);r['pairs'].update(sha256=R.sha(q),bytes=q.stat().st_size);self.write(Path(self.p['out'])/'report.json',r)
        with self.assertRaises(ValueError):R.terminal(self.p)

    def fake(self,failed=False):
        calls=[]
        def supervise(cmd,root,env,log,status,sp,**kw):
            calls.append((cmd,env,kw));status.update(producer_started=True,state='failed' if failed else 'worker_completed')
            if not failed:self.terminal()
        sup=SimpleNamespace(signal_guard=nullcontext,supervise=supervise,write=self.write)
        with patch.object(R.B,'load_supervisor',return_value=sup),patch.object(R.B,'resources',return_value={}), \
             patch.object(R.os,'sched_getaffinity',return_value={0,1},create=True),patch.object(R.os,'sched_setaffinity',create=True),patch.object(R.os,'nice'):
            return R.run(self.pp),calls

    def test_single_CPU_eval_no_training_retry_and_fresh_claim(self):
        r,calls=self.fake();self.assertTrue(r['complete']);self.assertFalse(r['training_or_fullframe_launched']);self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][2],{'wall':1800,'rss_limit':4<<30,'grace':30});self.assertEqual(calls[0][1]['TORCH_DEVICE_BACKEND_AUTOLOAD'],'0')
        with self.assertRaises(FileExistsError):R.run(self.pp)

    def test_worker_failure_preserved(self):
        r,calls=self.fake(failed=True);self.assertFalse(r['complete']);self.assertEqual(r['state'],'failed');self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main(verbosity=2)
