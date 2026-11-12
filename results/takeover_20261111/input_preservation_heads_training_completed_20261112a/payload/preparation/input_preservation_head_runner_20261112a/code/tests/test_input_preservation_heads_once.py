"""Stdlib small runner fixtures; no real head/model/device/cache execution."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
SOURCE=Path(__file__).resolve().parents[1]/'run_input_preservation_heads_once.py'
s=importlib.util.spec_from_file_location('head_once',SOURCE);R=importlib.util.module_from_spec(s);s.loader.exec_module(R)


class HeadRunner(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();(self.root/'runs').mkdir()
        self.entry=self.root/R.ENTRY;self.entry.write_text('# harmless fixture');self.sup=self.root/'owned.py';self.sup.write_text('# fixture')
        for key,value in [('ENTRY_SHA',R.sha(self.entry)),('SUPERVISOR_SHA',R.sha(self.sup))]:
            p=patch.object(R,key,value);p.start();self.addCleanup(p.stop)
        self.cp=self.root/'contract.json';self.ap=self.root/'admission.json';self.lp=self.root/'launch.json';self.pp=self.root/'plan.json'
        self.plan={'schema':R.SCHEMA,'root':str(self.root),'mode':'research','arm':'matched','resource':dict(R.RESOURCE),'physical_card':None,
            'run':str(self.root/'runs/once'),'out':str(self.root/'runs/once/output'),'entry':str(self.entry),'python':sys.executable,
            'supervision_source':str(self.sup),'execution_contract':str(self.cp),'admission':str(self.lp)}
        self.c={'schema':'rphi.input_preservation_head_training_execution.v1','root':str(self.root),'arm':'matched','mode':'research',
            'resources':{'wall_seconds':900,'CPU_cores':2,'RSS_limit_bytes':4<<30,'maximum_output_bytes':32<<20},
            'implementation_sha256':{str(self.entry):R.sha(self.entry)},'files':{},'backend_smokes':{},
            'runtime':{'expected_versions':{'torch_npu':None}},'admission':{'path':str(self.ap),'sha256':None}}
        self.save()

    def write(self,p,v):Path(p).write_text(json.dumps(v))

    def save(self):
        self.write(self.ap,{'schema':'rphi.input_preservation_head_training_admission.v1','approved':True,'arm':self.c['arm'],'mode':self.c['mode'],
            'execution_payload_sha256':R.semantic({k:v for k,v in self.c.items() if k!='admission'}),'entry_sha256':R.ENTRY_SHA,'out':self.plan['out']})
        self.c['admission']['sha256']=R.sha(self.ap);self.write(self.cp,self.c)
        self.plan['pins']={str(p):R.sha(p) for p in (SOURCE,self.entry,self.sup,self.cp,self.ap)}
        self.plan['argv']=[sys.executable,str(self.entry),'--execution-contract',str(self.cp),'--out',self.plan['out'],'--execute'];self.save_plan()

    def save_plan(self):
        self.write(self.pp,self.plan);self.write(self.lp,{'schema':'rphi.input_preservation_head_launch_admission.v1','approved':True,
            'scope':'one_CPU_input_preservation_head_arm_no_evaluation','arm':self.plan['arm'],'mode':self.plan['mode'],
            'plan_sha256':R.sha(self.pp),'runner_sha256':R.sha(SOURCE),'execution_contract_sha256':R.sha(self.cp),'physical_card':None})

    def terminal(self):
        out=Path(self.plan['out']);out.mkdir();steps=1000 if self.plan['mode']=='research' else 2
        r={'schema':'rphi.input_preservation_head_training.v1','complete':True,'mode':self.plan['mode'],'arm':self.plan['arm'],
            'status':'final_head_ready_for_bound_evaluation' if steps==1000 else 'CPU_backend_smoke_complete_not_research_resume',
            'completed_step':steps,'row_cursor':steps*16,'strict_checkpoint_reload':True,'NPU_calls':0,'new_restorer_forwards':0,
            'dev_label_arrays_read':0,'source_NPZ_reads':0,'resumed':False,'quality_accepted':False,'original_history_headroom_remains_failed':True,
            'counts':{k:steps for k in ('forward_attempts','forwards_completed','backward_attempts','backwards_completed','optimizer_attempts','optimizer_updates')},
            'runtime':{'device':'cpu','threads':2,'dtype':'float32','autocast':False,'grad_scaler':False,'versions':{'torch_npu':None}},
            'execution_payload_sha256':R.semantic({k:v for k,v in self.c.items() if k!='admission'}),'final_state_sha256':'fixture'}
        for name in ('checkpoint','initial_state','metrics'):
            p=out/name;p.write_bytes(name.encode());r[name]={'path':str(p),'sha256':R.sha(p)}
        r['checkpoint'].update(artifact_role='research_final_head' if steps==1000 else 'backend_smoke_not_research_resume',model_sha256='fixture')
        r['metrics']['rows']=steps;self.write(out/'report.json',r);return r

    def test_exact_permission_and_scope(self):
        self.assertEqual(R.validate(self.pp)['arm'],'matched')
        for key,value in [('arm','shuffled'),('physical_card',0),('resource',{**R.RESOURCE,'wall_seconds':901})]:
            before=self.plan[key];self.plan[key]=value;self.save_plan()
            with self.assertRaises(ValueError):R.validate(self.pp)
            self.plan[key]=before

    def test_entry_mutation_and_extra_eval_argv_rejected(self):
        self.plan['argv']+=['--eval'];self.save_plan()
        with self.assertRaises(ValueError):R.validate(self.pp)
        self.save();self.entry.write_text('# changed')
        with self.assertRaises(ValueError):R.validate(self.pp)

    def test_failed_or_wrong_step_or_integer_true_terminal_rejected(self):
        Path(self.plan['run']).mkdir();r=self.terminal();R.terminal(self.plan)
        for key,v in [('complete',1),('completed_step',2),('NPU_calls',1),('strict_checkpoint_reload',False)]:
            old=r[key];r[key]=v;self.write(Path(self.plan['out'])/'report.json',r)
            with self.assertRaises(ValueError):R.terminal(self.plan)
            r[key]=old

    def test_smoke_artifact_cannot_be_research_checkpoint(self):
        Path(self.plan['run']).mkdir();r=self.terminal();r['checkpoint']['artifact_role']='backend_smoke_not_research_resume';self.write(Path(self.plan['out'])/'report.json',r)
        with self.assertRaises(ValueError):R.terminal(self.plan)

    def fake(self,failed=False,mutate=False):
        calls=[]
        def supervise(cmd,root,env,log,status,path,**kw):
            calls.append((cmd,env,kw));status.update(producer_started=True,state='failed' if failed else 'worker_completed')
            if not failed:self.terminal()
            if mutate:self.entry.write_text('# changed')
        sup=SimpleNamespace(signal_guard=nullcontext,supervise=supervise,write=self.write)
        with patch.object(R,'load_supervisor',return_value=sup),patch.object(R,'resources',return_value={}), \
             patch.object(R.os,'sched_getaffinity',return_value={1,2,3},create=True), \
             patch.object(R.os,'sched_setaffinity',create=True),patch.object(R.os,'nice'):
            r=R.run(self.pp)
        return r,calls

    def test_single_CPU_launch_does_not_dispatch_evaluation(self):
        r,calls=self.fake();self.assertTrue(r['complete']);self.assertFalse(r['evaluation_launched']);self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],self.plan['argv']);self.assertEqual(calls[0][1]['TORCH_DEVICE_BACKEND_AUTOLOAD'],'0')
        self.assertEqual(calls[0][1]['ASCEND_RT_VISIBLE_DEVICES'],'');self.assertEqual(calls[0][2],{'wall':900,'rss_limit':4<<30,'grace':30})
        with self.assertRaises(FileExistsError):R.run(self.pp)

    def test_worker_failure_preserved_without_retry(self):
        r,calls=self.fake(failed=True);self.assertFalse(r['complete']);self.assertEqual(r['state'],'failed');self.assertEqual(len(calls),1)

    def test_post_success_source_mutation_rejected(self):
        r,calls=self.fake(mutate=True);self.assertFalse(r['complete']);self.assertFalse(r['pins_unchanged'])


if __name__=='__main__':unittest.main(verbosity=2)
