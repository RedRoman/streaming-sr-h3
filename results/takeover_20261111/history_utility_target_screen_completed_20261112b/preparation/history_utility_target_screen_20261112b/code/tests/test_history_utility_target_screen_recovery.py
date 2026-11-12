import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import recover_history_utility_target_screen as R
import materialize_mixed_temporal_pilot as N


class RecoveryTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def log(self,source_count=80,target_count=96):
  p=self.base/'decode.log';lines=['[showinfo@source] config in time_base: 1/20','[showinfo@target] config in time_base: 1/24']
  for tag,count,fps in [('source',source_count,20),('target',target_count,24)]:
   lines += ['[showinfo@%s] n:%d pts:%d pts_time:%.8f'%(tag,i,i,i/fps) for i in range(count)]
  p.write_text('\n'.join(lines));return p
 def test_exact_frozen_source_count_error_only_and_real_parser(self):
  p=self.log()
  with self.assertRaisesRegex(ValueError,'fewer than 96'):N.parse_timestamps(p.read_text())
  r=R.source_count_rejection(ValueError(R.COUNT_ERROR),p,N)
  self.assertEqual(r['decoded_source_count'],80);self.assertEqual(r['decoded_target_count'],96)
  self.assertIsNone(R.source_count_rejection(ValueError('some other failure'),p,N))
  self.assertIsNone(R.source_count_rejection(RuntimeError(R.COUNT_ERROR),p,N))
 def test_invalid_count_log_cannot_be_reclassified(self):
  for source_count,target_count in [(96,96),(80,95),(0,96)]:
   with self.assertRaises(ValueError):R.source_count_rejection(ValueError(R.COUNT_ERROR),self.log(source_count,target_count),N)
  with self.assertRaisesRegex(ValueError,'missing persistent'):R.source_count_rejection(ValueError(R.COUNT_ERROR),self.base/'absent',N)
 def test_source_count_must_not_hide_second_target_grid_failure(self):
  p=self.log();p.write_text(p.read_text().replace('[showinfo@target] n:95 pts:95','[showinfo@target] n:95 pts:96'))
  # The frozen parser raises the earlier count error; recovery must also verify
  # the downstream unchanged grid rule before classifying this exact exception.
  with self.assertRaisesRegex(ValueError,'fewer than 96'):N.parse_timestamps(p.read_text())
  with self.assertRaisesRegex(ValueError,'also violates frozen24fps'):
   R.source_count_rejection(ValueError(R.COUNT_ERROR),p,N)
 def test_log_saved_before_exception_cleanup_and_method_restored(self):
  dest=self.base/'persisted.log'
  def decoder(old,source,spec,scratch,ffmpeg,budget):
   (scratch/'target_decode.log').write_text('real original bytes');raise ValueError(R.COUNT_ERROR)
  module=types.SimpleNamespace(decode_target_owned=decoder)
  with tempfile.TemporaryDirectory(dir=self.base) as d:
   with self.assertRaises(ValueError):
    with R.preserve_decode_log(module,dest):module.decode_target_owned(None,None,None,Path(d),None,None)
  self.assertEqual(dest.read_text(),'real original bytes');self.assertIs(module.decode_target_owned,decoder)
 def fixture(self):
  c={'reused_receipts':{'reuse%02d'%i:{} for i in range(17)},'attempt_order':['new%02d'%i for i in range(22)],'files':{'prior_source_failure':{'path':'oldfailure','sha256':'f'*64}}};p=self.base/'contract.json';R.write(p,c);return c,p
 def test_17reuse22attempt1prior_ineligible_accounting_and_no_rerun(self):
  c,p=self.fixture();out=self.base/'run';calls=[]
  def launch(args,**kw):
   sid=args[-3];mode=args[-1];calls.append((sid,mode,kw['start_new_session']));dest=out/'sources'/sid;dest.mkdir(parents=True)
   bad=sid=='new03';status='reused_actual_pixels_verified' if mode=='reuse' else ('ineligible_source_count' if bad else 'new_target_completed')
   R.write(dest/'recovery_receipt.json',{'complete':True,'source_id':sid,'pid':100+len(calls),'new_decode_attempts':int(mode=='attempt'),'target_eligible':not bad,'status':status});return mock.Mock(returncode=0)
  with mock.patch.object(R,'validate'),mock.patch.object(R.shutil,'disk_usage',return_value=mock.Mock(free=8<<30)),mock.patch.object(R.subprocess,'run',side_effect=launch):r=R.execute(p,c,out)
  self.assertEqual(len(calls),39);self.assertNotIn('DtgdD48804M',[x[0] for x in calls]);self.assertTrue(all(not x[2] for x in calls))
  self.assertEqual((r['reused_complete_targets'],r['new_target_decode_attempts'],r['target_eligible_count'],r['ineligible_source_count']),(17,22,38,2))
  self.assertEqual(r['accounted_source_count'],40);self.assertEqual(r['source_outcomes']['new03']['state'],'ineligible')
  self.assertFalse(r['scientific_continuity_approved'])
  with self.assertRaises(FileExistsError):R.execute(p,c,out)
 def test_unexpected_error_preserves_error_noattempt_and_stops(self):
  c,p=self.fixture();out=self.base/'failed'
  with mock.patch.object(R,'validate'),mock.patch.object(R.shutil,'disk_usage',return_value=mock.Mock(free=8<<30)),mock.patch.object(R.subprocess,'run',return_value=mock.Mock(returncode=1)) as call:
   with self.assertRaisesRegex(ValueError,'fresh recovery worker failed'):R.execute(p,c,out)
  r=R.read(out/'failure.json');self.assertEqual(call.call_count,1);self.assertFalse(r['complete']);self.assertFalse((out/'complete.json').exists())
  self.assertEqual(r['source_outcomes']['reuse00']['state'],'error');self.assertEqual(r['source_outcomes']['new21']['state'],'noattempt');self.assertEqual(r['new_target_decode_attempts'],0)
 def test_reuse_actual_pixels_strict_and_zero_decode(self):
  c={'reused_receipts':{'sid':{'path':str(self.base/'prior.json'),'sha256':'unused'}},'attempt_order':[]}
  frames=np.zeros((2,3,4,3),np.uint8);p=self.base/'target.npz';np.savez(p,frames=frames);old={'files':{'numeric_source':{'path':'numeric'}}};spec={'execution_contract_sha256':N.value_sha(old)}
  target={'path':str(p),'pixel_sha256':N.pixel_sha(frames)};R.write(self.base/'prior.json',{'target':target,'target_specification':spec,'window_key':N.value_sha(spec)})
  numeric=types.SimpleNamespace(Budget=lambda *a:mock.Mock(),pixel_sha=N.pixel_sha,value_sha=N.value_sha)
  module=types.SimpleNamespace(load_numeric=lambda path:numeric,D=types.SimpleNamespace(SHAPE=frames.shape),check_no_models=lambda:{})
  screen=types.SimpleNamespace(load_original=lambda a:module,worker=mock.Mock(side_effect=AssertionError('must not decode')))
  out=self.base/'reuse'
  with mock.patch.object(R,'validate',return_value=(screen,{},old,{})):result=R.worker(c,out,'sid','reuse')
  self.assertTrue(result['actual_pixel_SHA_verified']);self.assertEqual(result['new_decode_attempts'],0);screen.worker.assert_not_called()
  np.savez(p,frames=frames+1)
  with mock.patch.object(R,'validate',return_value=(screen,{},old,{})):
   with self.assertRaisesRegex(ValueError,'actual pixels'):R.worker(c,self.base/'tampered','sid','reuse')

if __name__=='__main__':unittest.main()
