import json
import importlib.util
import signal
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import screen_history_utility_targets as S


class TargetScreenTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_target_diagnostics_fixed_indices_and_scores_not_cut_approval(self):
        frames=np.zeros((96,24,32,3),np.uint8);frames[50:]=255
        times=[{'index':i,'pts_time_seconds':(i+1)/24} for i in range(96)]
        r=S.diagnostics(frames,times,self.base,'synthetic',[72])
        self.assertEqual(len(r['all95_pair_scores']),95);self.assertEqual(len(r['top6']),6)
        self.assertEqual(r['top6'][0],{'new_frame_index':50,'score_mean_abs_RGB_0to1':1.0})
        self.assertFalse(r['scientific_continuity_approved']);self.assertEqual(len(r['artifacts']['full_resolution_pairs']),1)
        self.assertEqual(r['artifacts']['full_resolution_pairs'][0]['sha256'],S.sha(self.base/'full_pair_71_72.png'))

    def test_diagnostics_constant_frames_ties_deterministic_and_invalid_scope(self):
        frames=np.ones((96,24,32,3),np.uint8)*77;times=[{'index':i,'pts_time_seconds':i/24} for i in range(96)]
        r=S.diagnostics(frames,times,self.base,'constant',[])
        self.assertEqual([x['new_frame_index'] for x in r['top6']],list(range(1,7)))
        self.assertTrue(all(x['score_mean_abs_RGB_0to1']==0 for x in r['all95_pair_scores']))
        with self.assertRaisesRegex(ValueError,'96 frames'):S.diagnostics(frames[:-1],times,self.base,'bad',[])

    def test_coordinator_fresh_owned_workers_36decode4reuse_no_overwrite(self):
        order=sorted(list(S.REUSE)+['new%02d'%i for i in range(36)])
        c={'source_order':order};contract=self.base/'contract.json';S.write(contract,c);out=self.base/'run';calls=[]
        def subprocess_run(args,**kwargs):
            sid=args[-1];calls.append((sid,kwargs['start_new_session']));dest=out/'sources'/sid;dest.mkdir(parents=True)
            S.write(dest/'complete.json',{'complete':True,'pid':2000+len(calls),'reused':sid in S.REUSE,'new_target_decodes':int(sid not in S.REUSE)})
            return mock.Mock(returncode=0)
        with mock.patch.object(S,'validate',return_value=({},{})),mock.patch.object(S.shutil,'disk_usage',return_value=mock.Mock(free=8<<30)),mock.patch.object(S.subprocess,'run',side_effect=subprocess_run):
            r=S.execute(contract,c,out)
        self.assertEqual(r['new_target_decodes'],36);self.assertEqual(r['reused_targets'],4)
        self.assertEqual(calls,[(sid,False) for sid in order]);self.assertFalse(r['scientific_continuity_approved'])
        self.assertEqual(r['degradation_codec_passes'],0)
        with self.assertRaises(FileExistsError):S.execute(contract,c,out)

    def test_fail_stop_preserves_false_complete_and_disk_limit(self):
        out=self.base/'failed';c={'source_order':['x']};p=self.base/'c.json';S.write(p,c)
        with mock.patch.object(S,'validate',return_value=({},{})),mock.patch.object(S.shutil,'disk_usage',return_value=mock.Mock(free=8<<30)),mock.patch.object(S.subprocess,'run',return_value=mock.Mock(returncode=1)) as call:
            with self.assertRaisesRegex(ValueError,'fresh target worker failed'):S.execute(p,c,out)
        self.assertEqual(call.call_count,1);self.assertFalse(S.read(out/'failure.json')['complete']);self.assertFalse((out/'complete.json').exists())
        with mock.patch.object(S,'validate',return_value=({},{})),mock.patch.object(S.shutil,'disk_usage',return_value=mock.Mock(free=5<<30)):
            with self.assertRaisesRegex(ValueError,'need6GiB'):S.execute(p,c,self.base/'notstarted')
        self.assertFalse((self.base/'notstarted').exists())

    def test_supervisor_external_signal_cleans_only_owned_group_and_restores_handlers(self):
        path=Path(__file__).resolve().parents[2]/'preparation/history_utility_target_screen_20261112a/run_once.py'
        spec=importlib.util.spec_from_file_location('_target_screen_supervisor_test',path)
        supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
        process=mock.Mock(pid=987654);process.poll.return_value=None
        before={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        with mock.patch.object(supervisor,'stop_owned') as stop:
            with self.assertRaises(InterruptedError):
                with supervisor.owned_signal_guard(process):
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        stop.assert_called_once_with(process,grace=30)
        self.assertEqual({sig:signal.getsignal(sig) for sig in before},before)


if __name__=='__main__':unittest.main()
