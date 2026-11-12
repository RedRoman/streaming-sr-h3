import importlib.util,json,os,sys,tarfile,tempfile,unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('watch',HERE/'finalize_teacher_once.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)

class FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name).resolve()
    def tearDown(self):self.tmp.cleanup()
    def put(self,name,obj):
        p=self.p/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return p
    def binding(self,p):return {'path':str(p),'sha256':F.sha(p)}
    def fixture(self,state='completed'):
        run=self.p/'run';run.mkdir();out=run/'output';out.mkdir()
        report=self.put('run/output/report.json',{'complete':True})
        self.put('run/output/metrics.jsonl',{'step':1000})
        (out/'teacher_final.pt').write_bytes(b'excluded checkpoint')
        contract=self.put('execution.json',{'steps':1000,'phase':'teacher','resume':None})
        plan=self.put('plan.json',{'kind':'teacher1000','execution_contract':str(contract),'run':str(run),'out':str(out),'pins':{str(contract):F.sha(contract)}})
        self.put('run/status.json',{'state':state,'complete':state=='completed','pins_unchanged':True,'plan_sha256':F.sha(plan),'report_sha256':F.sha(report)})
        auditor=self.p/'fake_auditor.py';auditor.write_text("import json,pathlib,sys\np=pathlib.Path(sys.argv[sys.argv.index('--out')+1]);p.write_text(json.dumps({'complete':True,'fixture_only':True}))\n")
        c={'schema':'rphi.history_utility_teacher_terminal_watch.v1','kind':'teacher1000','resources':{'wait_seconds':7500,'poll_seconds':30,'audit_seconds':120,'cleanup_seconds':10,'CPU':2,'AS_bytes':4<<30},'source_pins':{str(Path(F.__file__).resolve()):F.sha(F.__file__),str(auditor):F.sha(auditor)},'auditor':self.binding(auditor),'producer_plan':self.binding(plan),'producer_execution_contract':self.binding(contract),'producer_status':str(run/'status.json'),'report':str(report),'out':str(self.p/'finalized'),'python':sys.executable}
        ad=self.put('admission.json',{'approved':True,'scope':'teacher_final_checkpoint_audit_only','execution_payload_sha256':F.value_sha(c),'finalizer_sha256':F.sha(F.__file__)})
        c['admission']=self.binding(ad);cp=self.put('config.json',c)
        return cp,c
    def test_not_launched_terminal(self):
        p=self.put('s.json',{'state':'not_launched'})
        s,_,reads=F.wait_terminal(p,sleep=lambda _:self.fail('must not sleep'));self.assertEqual(s['state'],'not_launched');self.assertEqual(reads,1)
    def test_wait_timeout_is_bounded(self):
        t=[0.];sleeps=[]
        def sleep(x):sleeps.append(x);t[0]+=x
        with self.assertRaises(TimeoutError):F.wait_terminal(self.p/'missing',seconds=35,poll=30,clock=lambda:t[0],sleep=sleep)
        self.assertEqual(sleeps,[30,5])
    def test_failed_producer_preserved_no_audit_and_one_shot(self):
        cp,c=self.fixture('failed');r=F.finalize(cp,c)
        self.assertFalse(r['complete']);self.assertEqual(r['audit_invocations'],0)
        with tarfile.open(self.p/'finalized/evidence.tar.gz') as t:self.assertFalse(any(n.endswith('.pt') for n in t.getnames()))
        with self.assertRaisesRegex(ValueError,'one-shot'):F.finalize(cp,c)
    def test_success_invokes_once_and_small_archive(self):
        cp,c=self.fixture();r=F.finalize(cp,c);self.assertTrue(r['complete']);self.assertEqual(r['audit_invocations'],1)
        manifest=F.read(self.p/'finalized/MANIFEST.json');self.assertFalse(manifest['checkpoint_or_arrays_included'])
        self.assertIn('producer/metrics.jsonl',[x['path'] for x in manifest['files']])
    def test_source_pin_mismatch_rejected_before_output(self):
        cp,c=self.fixture();Path(c['auditor']['path']).write_text('changed')
        with self.assertRaises(ValueError):F.finalize(cp,c)
        self.assertFalse(Path(c['out']).exists())
    def test_archive_rejects_tensor_and_member_traversal(self):
        p=self.p/'tensor.pt';p.write_bytes(b'weights');out=self.p/'pack';out.mkdir()
        with self.assertRaises(ValueError):F.pack(out,{'tensor.pt':p})
        p=self.put('safe.json',{})
        with self.assertRaises(ValueError):F.pack(out,{'../escape.json':p})

if __name__=='__main__':unittest.main(verbosity=2)
