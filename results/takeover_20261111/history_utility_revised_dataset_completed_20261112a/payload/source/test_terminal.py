"""CPU fixtures: metadata/real numeric schedule/owned guards, no cohort scan."""
import copy
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import audit_terminal as A
import finalize_once as F
import small_archive as S


class Clock:
    def __init__(self): self.now = 0
    def clock(self): return self.now
    def sleep(self, value): self.now += value


def terminal_fixture():
    train = ['4EKWTiWjPTQ'] + ['train%02d' % i for i in range(18)]
    dev = ['6UZVovkR3IU', '6rJrrfRO3cA'] + ['dev%02d' % i for i in range(4)]
    config = {'producer_contract': {'path': '/unused/contract.json', 'sha256': '1' * 64},
              'producer_freeze': {'path': '/unused/freeze.json', 'sha256': '2' * 64},
              'producer_admission': {'path': '/unused/admission.json', 'sha256': '3' * 64}}
    cohort = {'path': '/unused/cohort.json', 'sha256': A.COHORT_SHA}
    contract = {'files': {'cohort': cohort}, 'selected_source_ids': {'train': train, 'dev': dev}}
    rows = []
    for sid in train + dev:
        reused = sid in A.REUSE
        entry = {'path': '/unused/' + sid + '.json', 'sha256': A.D.value_sha(sid)}
        if reused: entry['revised_authority'] = {'path': '/unused/' + sid + '_reuse.json', 'sha256': '4' * 64}
        rows.append({'source_id': sid, 'entry': entry, 'mode': 'reused_complete' if reused else 'new_materialization',
                     'new_target_decodes': 0, 'new_codec_passes': 0 if reused else 6,
                     'verification': {'old_loader_actual_files_pixels_metadata_verified': 4, 'legacy_research_ready': False}})
    report = {'schema': 'openh3.history_utility_revised_materialization.v1', 'complete': True,
        'status': 'materialized_revised_primary_pending_dataset_validation',
        'execution_contract': config['producer_contract'], 'cohort': cohort, 'new_target_decodes': 0,
        'new_codec_passes_completed': 132, 'model_calls': 0, 'NPU_calls': 0, 'legacy_research_ready': False,
        'revised_dataset_validation_pending': True, 'training_admitted': False,
        'import_guard': {'forbidden_modules': [], 'forbidden_import_attempts': []}, 'source_results': rows,
        'receipt_indices': {'train': [r['entry'] for r in rows[:19]], 'dev': [r['entry'] for r in rows[19:]]}}
    status = {'schema': 'openh3.history_utility_cpu_supervision.v1', 'state': 'completed', 'complete': True,
        'returncode': 0, 'report_sha256': '5' * 64, 'freeze_sha256': config['producer_freeze']['sha256'],
        'admission_sha256': config['producer_admission']['sha256'], 'owned_group_only': True,
        'wall_seconds': 3600, 'grace_seconds': 30, 'memory_mib': 768,
        'elapsed_seconds': 100, 'peak_owned_group_rss_bytes': 200 << 20, 'cpu_affinity': [3]}
    return status, report, contract, config


class TerminalTests(unittest.TestCase):
    def test_exact25_new22_reuse3_and_indices(self):
        s, r, c, cfg = terminal_fixture()
        self.assertEqual(len(A.validate_terminal(s, r, c, cfg, '5' * 64)), 25)
        r['receipt_indices']['train'].reverse()
        with self.assertRaisesRegex(ValueError, 'ordered exact'): A.validate_terminal(s, r, c, cfg, '5' * 64)

    def test_fail_closed_failed_duplicate_budget_authority(self):
        for change in ('failed', 'duplicate', 'budget', 'authority', 'promote', 'admission'):
            s, r, c, cfg = terminal_fixture()
            if change == 'failed': s['state'] = 'failed'
            if change == 'duplicate': r['source_results'][-1] = r['source_results'][0]
            if change == 'budget': s['peak_owned_group_rss_bytes'] = (768 << 20) + 1
            if change == 'authority': r['source_results'][0]['entry'].pop('revised_authority')
            if change == 'promote': r['training_admitted'] = True
            if change == 'admission': s['admission_sha256'] = '0' * 64
            with self.subTest(change=change), self.assertRaises(ValueError): A.validate_terminal(s, r, c, cfg, '5' * 64)

    def test_wait_accepts3630_not_early3600(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            def sleep(value):
                clock.sleep(value)
                if clock.now >= 3630: path.write_text(json.dumps({'state': 'completed'}))
            result, digest, elapsed = F.wait_terminal(path, clock=clock.clock, sleep=sleep)
            self.assertEqual((result['state'], elapsed), ('completed', 3630))
            self.assertEqual(digest, F.sha(path))

    def test_wait4200_timeout_and_failed_no_retry(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            with self.assertRaises(TimeoutError): F.wait_terminal(path, clock=clock.clock, sleep=clock.sleep)
            self.assertEqual(clock.now, 4200)
            path.write_text(json.dumps({'state': 'failed'}))
            self.assertEqual(F.wait_terminal(path)[0]['state'], 'failed')

    def test_memory_and_time_limits_are_actual_not_declarations(self):
        process = SimpleNamespace(pid=123, poll=lambda: None)
        with self.assertRaises(MemoryError):
            F.monitor(process, {}, group_members=lambda _: [{'rss_bytes': (4 << 30) + 1}])
        clock = Clock()
        with self.assertRaises(TimeoutError):
            F.monitor(process, {}, worker_seconds=1, clock=clock.clock, sleep=clock.sleep, group_members=lambda _: [])
        self.assertGreaterEqual(clock.now, 1)

    def test_archive_uses_remaining_budget_not_new900(self):
        self.assertEqual(F.archive_remaining(10, clock=lambda: 900), 10)
        with self.assertRaisesRegex(TimeoutError, 'not attempted'):
            F.archive_remaining(10, clock=lambda: 910)

    @unittest.skipUnless(sys.platform == 'linux', 'actual procfs owned-group tests run in Linux CPU bundle')
    def test_owned_context_cleans_only_owned_group_on_exception(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        proc = None
        try:
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with F.spawned_owned([sys.executable, '-c', 'import time; time.sleep(30)']) as proc:
                    time.sleep(.05); self.assertTrue(F.members(proc.pid)); raise RuntimeError('fixture')
            self.assertIsNotNone(proc.poll()); self.assertEqual(F.members(proc.pid), [])
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate(); unrelated.wait(timeout=3)

    @unittest.skipUnless(sys.platform == 'linux', 'actual procfs owned-group tests run in Linux CPU bundle')
    def test_external_term_cleans_and_restores_handler(self):
        before = signal.getsignal(signal.SIGTERM)
        proc = None
        with self.assertRaises(InterruptedError):
            with F.spawned_owned([sys.executable, '-c', 'import time; time.sleep(30)']) as proc:
                signal.raise_signal(signal.SIGTERM)
        self.assertIsNotNone(proc.poll()); self.assertEqual(F.members(proc.pid), [])
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    @unittest.skipUnless(sys.platform == 'linux', 'finalizer failure path requires actual Linux procfs')
    def test_archive_failure_revokes_outer_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root / 'output'
            admission = root / 'admission.json'
            admission.write_text(json.dumps({'approved': True, 'scope': 'revised_dataset_terminal_audit_only',
                                             'host': '31984', 'freeze_sha256': 'a' * 64}))
            cfg = {'out': str(out), 'producer_status': str(root / 'producer_status.json')}
            argv = ['finalize_once.py', '--freeze', str(root / 'freeze.json'), '--freeze-sha256', 'a' * 64,
                    '--execute', '--admission', str(admission), '--admission-sha256', F.sha(admission)]
            # Failed producer invokes no dataset; exercise bounded failure archive
            # and ensure its own failure cannot leave an apparent success.
            with patch.object(sys, 'argv', argv), patch.object(F.sys, 'platform', 'linux'), \
                 patch.object(F, 'validate_freeze', return_value=({}, cfg)), \
                 patch.object(F, 'wait_terminal', return_value=({'state': 'failed', 'complete': False}, 'b'*64, 30)), \
                 patch.object(signal, 'alarm'), patch.object(signal, 'setitimer'), \
                 patch.object(S, 'pack', side_effect=ValueError('fixture archive failure')):
                with self.assertRaisesRegex(ValueError, 'fixture archive'): F.main()
            status = json.loads((out / 'status.json').read_text())
            self.assertEqual(status['state'], 'failed'); self.assertFalse(status['complete'])
            self.assertFalse(status['revised_pair_ready']); self.assertEqual(status['audit_invocations'], 0)
            self.assertFalse(status['retry_permitted']); self.assertIn('archive_error', status)

    def test_archive_allowlist_excludes_npz_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root / 'audit'; out.mkdir(); run = root / 'run'; run.mkdir()
            (out / 'status.json').write_text('{}'); (out / 'admission.json').write_text('{}')
            (out / 'hidden.npz').write_bytes(b'array-not-allowed')
            source = root / 'src'; source.mkdir(); (source / 'audit_terminal.py').write_text('# fixture')
            cfg = {'producer_run': str(run)}
            for key in ('producer_contract', 'producer_freeze', 'producer_admission'):
                path = root / (key + '.json'); path.write_text('{}'); cfg[key] = {'path': str(path)}
            receipt = S.pack(out, cfg, source)
            self.assertFalse(receipt['NPZ_included'])
            manifest = json.loads((out / 'MANIFEST.json').read_text())
            self.assertFalse(any(r['path'].endswith('.npz') for r in manifest['files']))
            with self.assertRaises(FileExistsError): S.pack(out, cfg, source)

    def test_archive_rejects_disguised_npz_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root / 'audit'; out.mkdir(); run = root / 'run'; run.mkdir()
            npz = root / 'bad.npz'; npz.write_bytes(b'not-json')
            (run / 'complete.json').write_text(json.dumps({'source_results': [{'source_id': 'x', 'entry': {'path': str(npz)}}]}))
            cfg = {'producer_run': str(run)}
            for key in ('producer_contract', 'producer_freeze', 'producer_admission'):
                path = root / (key + '.json'); path.write_text('{}'); cfg[key] = {'path': str(path)}
            with self.assertRaisesRegex(ValueError, 'non-small'): S.pack(out, cfg, root)


class NumericScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Actual frozen D.schedule/_row code, with precomputed synthetic hashes;
        # does not create or inspect any real cohort pixels.
        numeric = A.D.HistoryUtilityDataset.__new__(A.D.HistoryUtilityDataset)
        numeric.verify_pixels = True; numeric.split = 'train'; numeric.scope = 'engineering'; numeric.research_ready = False
        numeric.dataset_sha256 = '6' * 64; numeric.source_ids = ['source%02d' % i for i in range(19)]
        numeric.sources = {}; numeric._control_sha = {}; registry = {}
        for sid in numeric.source_ids:
            source = {'source_id': sid, 'source_sha256': A.D.value_sha(sid), 'source_group_id': 'group:' + sid, 'split': 'train'}
            registry[sid] = source; numeric._control_sha[sid] = A.D.value_sha(sid + ':near')
            numeric.sources[sid] = {'receipt': {'source': source, 'window_key': A.D.value_sha(sid + ':window'),
                'target': {'pixel_sha256': A.D.value_sha(sid + ':target')},
                'observations': {pid: {'artifact': {'pixel_sha256': A.D.value_sha(sid + pid)}} for pid in A.D.PROFILES}}}
        cls.data = SimpleNamespace(_numeric=numeric, dataset_sha256='7' * 64, cohort_sha256=A.COHORT_SHA,
                                   source_ids=numeric.source_ids, registry=registry)
        cls.schedules = {role: A.R.RevisedHistoryUtilityDataset.schedule(cls.data, role, 1000) for role in A.D.SEEDS}

    def test_real_numeric_two1000x16_seeds_and_contract(self):
        for role, schedule in self.schedules.items():
            result = A.schedule_summary(schedule, self.data)
            self.assertEqual(result['rows'], 16000); self.assertEqual(result['seed'], A.D.SEEDS[role])
            self.assertEqual(len(result['source_counts']), 19)
            self.assertEqual(result['geometry_contract']['observed'], [16, 3, 3, 96, 96])
        self.assertNotEqual(self.schedules['teacher']['schedule_sha256'], self.schedules['student']['schedule_sha256'])

    def test_schedule_rejects_cross_split_or_noncausal_even_rehashed(self):
        for mutation in ('split', 'future'):
            schedule = copy.deepcopy(self.schedules['student']); row = schedule['numeric_schedule']['rows'][0]
            if mutation == 'split': row['split'] = 'dev'
            else: row['history_indices'][-1] = row['frame_index'] + 1
            row['row_sha256'] = A.D.value_sha({k: v for k, v in row.items() if k != 'row_sha256'})
            numeric = schedule['numeric_schedule']
            numeric['schedule_sha256'] = A.D.value_sha({k: v for k, v in numeric.items() if k != 'schedule_sha256'})
            schedule['schedule_sha256'] = A.D.value_sha({k: v for k, v in schedule.items() if k != 'schedule_sha256'})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): A.schedule_summary(schedule, self.data)


if __name__ == '__main__': unittest.main(verbosity=2)
