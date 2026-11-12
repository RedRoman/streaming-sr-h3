"""Synthetic CPU artifacts only; no real degradation codec or video decoder."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import history_utility_data as D
import materialize_history_utility_revised as M

ROOT = Path(__file__).resolve().parents[2]


def put(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True)); return M.bound(path)


class RevisedMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.sid = '2BdYaGuVAKo'  # synthetic fixture only; production cohort excludes it.
        cls.plan = D._json(ROOT / 'configs/openh3_20261111/mixed_temporal_data_plan_20261111.json')
        original = D._json(ROOT / 'results/takeover_20261111/mixed_temporal_single_source_smoke_completed/complete.json')
        source = cls.root / (cls.sid + '.mp4'); source.write_bytes(b'synthetic byte hash fixture; not a video')
        manifests = {}
        for split in ('train', 'dev'):
            rows = [json.loads(x) for x in (ROOT / f'results/takeover_20261111/materialized/{split}_materialized.jsonl').read_text().splitlines()]
            for row in rows:
                if row['source_id'] == cls.sid: row['source_sha256'] = M.sha(source)
            path = cls.root / (split + '.jsonl'); path.write_text('\n'.join(json.dumps(x) for x in rows))
            manifests[split] = M.bound(path)
            cls.plan['source_roles']['existing_' + split]['manifest_sha256'] = M.sha(path)
        plan_ref = put(cls.root / 'plan.json', cls.plan)
        cls.registry = D.load_source_registry(cls.plan, manifests['train']['path'], manifests['dev']['path'])
        binding = cls.registry[cls.sid]
        receipt = copy.deepcopy(original); receipt['source'] = binding
        receipt['registry']['sources'] = [cls.registry[k] for k in sorted(cls.registry)]
        receipt['registry_sha256'] = D.value_sha(receipt['registry'])
        receipt['plan_file_sha256'] = plan_ref['sha256']
        receipt['execution_contract'].update(ffmpeg='unused-fixture-ffmpeg', ffprobe='unused-fixture-ffprobe')
        receipt['target_specification'].update(binding)
        receipt['target_specification']['execution_contract_sha256'] = D.value_sha(receipt['execution_contract'])
        receipt['window_key'] = receipt['target_specification_sha256'] = D.value_sha(receipt['target_specification'])
        cls.profiles = [next(p for p in cls.plan['profiles'] if p['profile_id'] == pid) for pid in D.PROFILES]
        spec = importlib.util.spec_from_file_location('_fixture_frozen_numeric', ROOT / 'code/materialize_mixed_temporal_pilot.py')
        cls.old = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.old)
        pixels = np.zeros(D.SHAPE, np.uint8)
        target_sha = D.array_sha(pixels)
        common = {'schema': D.MATERIALIZER_SCHEMA, **binding, 'window_key': receipt['window_key'],
                  'target_pixel_sha256': target_sha,
                  **{k: receipt[k] for k in ('plan_file_sha256', 'registry_sha256', 'materializer_sha256', 'software_sha256')}}
        cls.fixture_dir = cls.root / 'fixture'; cls.fixture_dir.mkdir()
        receipt['target'] = cls.old.save_npz(cls.fixture_dir / 'target.npz', pixels,
            {**common, 'role': 'target', 'target_specification': receipt['target_specification']})
        for profile in cls.profiles:
            pid = profile['profile_id']; obs = receipt['observations'][pid]
            obs['artifact'] = cls.old.save_npz(cls.fixture_dir / (pid + '.npz'), pixels,
                {**common, 'role': 'observation', 'profile': profile,
                 'parameter_schedule_sha256': D.value_sha(profile['frame_parameters']),
                 'noise_stream': profile['noise_stream'], 'blind_model_condition': 'not_supplied_new_loader_required'})
            for replay in obs['replays']:
                replay['observation_pixel_sha256'] = target_sha
                for i, noise in enumerate(replay['noise_frames']):
                    noise['seed_uint64'] = cls.old.noise_seed(binding['source_sha256'], receipt['window_key'], profile['noise_stream'], i)
                replay['noise_schedule_sha256'] = D.value_sha(replay['noise_frames'])
        del pixels
        cls.receipt = receipt; cls.entry = put(cls.fixture_dir / 'complete.json', receipt)
        cls.contract = {'schema': M.CONTRACT_SCHEMA, 'files': {
            'cohort': put(cls.root / 'synthetic_cohort.json', {'fixture': 'not real cohort'}),
            'plan': plan_ref, 'train_manifest': manifests['train'], 'dev_manifest': manifests['dev'],
            'source_inventory': put(cls.root / 'inventory.json', {'fixture': True}),
            'numerical_window_contract': put(cls.root / 'numeric_contract.json', receipt['execution_contract'])},
            'software_sha256': receipt['software_sha256'], 'selected_source_ids': {'train': [cls.sid], 'dev': []},
            'reuse_receipts': {cls.sid: cls.entry}, 'new_sources': []}
        cls.contract_path = cls.root / 'execution_contract.json'; put(cls.contract_path, cls.contract)
        cls.context = {'D': D, 'registry': cls.registry, 'profiles': cls.profiles, 'plan': cls.plan,
            'original': receipt['execution_contract'],
            'source_files': {cls.sid: {'path': str(source), 'bytes': source.stat().st_size, 'probe': {'fixture': True}}},
            'cohort': {'decisions': {cls.sid: {'target_receipt': cls.entry}}},
            'evidence': {cls.sid: {'target': receipt['target'], 'target_specification': receipt['target_specification'],
                'numerical': {'target': receipt['target'], 'target_specification': receipt['target_specification'],
                              'window_key': receipt['window_key'], 'decode': receipt['decode']}}}}

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=self.root); self.out = Path(self.case.name) / 'output'

    def tearDown(self): self.case.cleanup()

    def fake_numeric(self, mismatch=False):
        old = SimpleNamespace(**vars(self.old)); old.software_receipt = mock.Mock(return_value=self.receipt['software'])
        old.source_probe = mock.Mock(side_effect=AssertionError('video probe forbidden'))
        old.decode_target = mock.Mock(side_effect=AssertionError('video decode forbidden'))
        calls = []
        def observation(clean, profile, binding, window, scratch, replay, budget):
            calls.append((profile['profile_id'], replay, window))
            detail = copy.deepcopy(self.receipt['observations'][profile['profile_id']]['replays'][replay])
            if mismatch and replay == 1: detail['observation_pixel_sha256'] = 'f' * 64
            raw = scratch / ('fixture_%s_%d.raw' % (profile['profile_id'], replay)); raw.touch()
            return clean, raw, detail
        old.observation = mock.Mock(side_effect=observation)
        return old, calls

    def test_target_metadata_bridge_and_actual_old_loader_compatibility(self):
        old, calls = self.fake_numeric()
        before = M.sha(self.entry['path'])
        result = M.materialize_source(self.contract_path, self.contract, self.context, old,
                                     self.sid, self.out, mock.Mock())
        receipt = M.read(result['entry']['path'])
        self.assertEqual(calls, [(pid, replay, self.receipt['window_key']) for pid in D.PROFILES for replay in (0, 1)])
        self.assertEqual(result['new_target_decodes'], 0); self.assertEqual(result['new_codec_passes'], 6)
        self.assertEqual(receipt['target']['pixel_sha256'], self.receipt['target']['pixel_sha256'])
        self.assertNotEqual(receipt['target']['file_sha256'], self.receipt['target']['file_sha256'])
        self.assertEqual(receipt['target_specification'], self.receipt['target_specification'])
        self.assertEqual(receipt['execution_contract'], self.receipt['execution_contract'])
        self.assertFalse(receipt['scientific_continuity_approved'])
        self.assertEqual(receipt['verification']['old_loader_actual_files_pixels_metadata_verified'], 4)
        for artifact in [receipt['target']] + [receipt['observations'][p]['artifact'] for p in D.PROFILES]:
            with np.load(artifact['path'], allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata'].item()))
                self.assertEqual(metadata['execution_authority'], receipt['execution_authority'])
                self.assertEqual(metadata['revised_cohort_sha256'], M.COHORT_SHA)
        self.assertEqual(M.sha(self.entry['path']), before)
        old.source_probe.assert_not_called(); old.decode_target.assert_not_called()
        with self.assertRaises(FileExistsError):
            M.materialize_source(self.contract_path, self.contract, self.context, old, self.sid, self.out, mock.Mock())

    def test_reuse_actual_profile_pixels_no_codec_and_immutable_receipt(self):
        before = M.sha(self.entry['path'])
        result = M.reuse_source(self.contract_path, self.contract, self.context, self.sid, self.out)
        self.assertEqual(result['entry']['path'], self.entry['path'])
        self.assertEqual(result['new_codec_passes'], 0)
        self.assertEqual(M.sha(self.entry['path']), before)
        sidecar = M.read(result['entry']['revised_authority']['path'])
        self.assertTrue(sidecar['original_receipt_unchanged']); self.assertTrue(sidecar['target_pixels_unchanged'])
        self.assertEqual(sidecar['execution_authority']['execution_contract_value_sha256'], D.value_sha(self.contract))
        self.assertFalse(any(self.out.glob('*.npz')))

    def test_bad_pixel_bytes_or_metadata_fail_before_any_codec(self):
        context = {**self.context, 'evidence': copy.deepcopy(self.context['evidence'])}
        path = Path(self.receipt['target']['path']); raw = path.read_bytes()
        try:
            path.write_bytes(raw + b'tamper')
            with self.assertRaisesRegex(ValueError, 'target file'): M.read_target(context, self.sid)
        finally: path.write_bytes(raw)
        with np.load(path, allow_pickle=False) as data:
            pixels = data['frames']; metadata = json.loads(str(data['metadata'].item()))
        metadata['window_key'] = 'a' * 64
        bad = self.out.parent / 'wrong_metadata.npz'
        target = self.old.save_npz(bad, pixels, metadata)
        context['evidence'][self.sid]['numerical']['target'] = target
        with self.assertRaisesRegex(ValueError, 'metadata identity'): M.read_target(context, self.sid)

    def test_replay_failure_preserves_counts_never_complete(self):
        old, _ = self.fake_numeric(mismatch=True)
        with self.assertRaisesRegex(ValueError, 'double replay'):
            M.materialize_source(self.contract_path, self.contract, self.context, old, self.sid, self.out, mock.Mock())
        self.assertFalse((self.out / 'complete.json').exists())
        failed = M.read(self.out / 'failure.json')
        self.assertEqual(failed['status'], 'failed_not_complete')
        self.assertEqual(failed['new_codec_passes_started'], 2)
        self.assertEqual(failed['new_codec_passes_completed'], 2)

    def test_nonprimary_worker_rejected_without_read_or_codec(self):
        with mock.patch.object(M, 'load_context', return_value=self.context):
            with self.assertRaisesRegex(ValueError, 'non-primary'):
                M.worker(self.contract_path, self.contract, self.root, 'not_selected')

    def test_inspection_only_and_authority_admission_strict(self):
        contract = copy.deepcopy(self.contract); contract.update(budget=M.BUDGET)
        with mock.patch.object(M, 'load_context', return_value=self.context):
            report = M.inspect(contract)
        self.assertFalse(report['actual_target_pixel_verification'])
        self.assertEqual(report['codec_passes_executed'], 0)
        with self.assertRaisesRegex(ValueError, 'admission'): M.validate_admission(None, self.contract_path)
        admission = {'schema': 'openh3.history_utility_revised_materialization_admission.v1', 'approved': True,
                     'contract_sha256': M.sha(self.contract_path), 'entry_sha256': M.sha(M.__file__)}
        path = self.out.parent / 'admission.json'; put(path, admission)
        M.validate_admission(path, self.contract_path)
        admission['contract_sha256'] = '0' * 64; put(path, admission)
        with self.assertRaisesRegex(ValueError, 'admission differs'): M.validate_admission(path, self.contract_path)

    def test_failed_fresh_worker_stops_no_retry_and_preserves_original_output(self):
        contract = copy.deepcopy(self.contract)
        calls = []
        class Process:
            returncode = 1
            def __init__(self, command, **kw): calls.append((command, kw['start_new_session']))
            def wait(self, **kw): return self.returncode
            def poll(self): return self.returncode
        with mock.patch.object(M.subprocess, 'Popen', Process), mock.patch.object(M, 'disk_check', return_value=0):
            with self.assertRaisesRegex(ValueError, 'source worker failed'):
                M.execute(self.contract_path, contract, self.out, self.root / 'admission.json')
        self.assertEqual(len(calls), 1); self.assertFalse(calls[0][1])
        self.assertIn('--admission', calls[0][0])
        self.assertFalse(M.read(self.out / 'failure.json')['complete'])
        with self.assertRaisesRegex(ValueError, 'output already exists'):
            M.execute(self.contract_path, contract, self.out, self.root / 'admission.json')

    def test_new_supervisor_exact_3600_limit_and_pins(self):
        path = ROOT / 'preparation/history_utility_revised_materializer_20261112a/run_once.py'
        spec = importlib.util.spec_from_file_location('_revised_supervisor_fixture', path)
        supervisor = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervisor)
        root = self.root.resolve()
        command = [sys.executable, M.__file__, '--contract', str(self.contract_path),
                   '--out', str(root / 'runs' / 'worker')]
        config = {'schema': 'openh3.history_utility_cpu_freeze.v1', 'host': '31984',
            'wall_seconds': 3600, 'grace_seconds': 30, 'memory_mib': 768,
            'pins': {str(path): M.sha(path), M.__file__: M.sha(M.__file__), str(self.contract_path): M.sha(self.contract_path)},
            'root': str(root), 'run': str(root / 'runs' / 'worker'),
            'supervisor': str(root / 'runs' / 'supervisor'), 'command': command,
            'execution_contract': str(self.contract_path)}
        freeze = self.out.parent / 'freeze.json'; put(freeze, config)
        self.assertEqual(supervisor.validate_freeze(freeze, M.sha(freeze))['wall_seconds'], 3600)
        config['wall_seconds'] = 1800; put(freeze, config)
        with self.assertRaisesRegex(ValueError, 'CPU limits'): supervisor.validate_freeze(freeze, M.sha(freeze))

    def test_new_supervisor_exception_cleans_only_owned_group(self):
        path = ROOT / 'preparation/history_utility_revised_materializer_20261112a/run_once.py'
        spec = importlib.util.spec_from_file_location('_revised_supervisor_cleanup_fixture', path)
        supervisor = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervisor)
        process = mock.Mock(); process.poll.return_value = None
        with mock.patch.object(supervisor, 'stop_owned') as stop:
            with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                with supervisor.owned_signal_guard(process): raise RuntimeError('fixture interruption')
        stop.assert_called_once_with(process, grace=30)

    def test_spawn_to_handler_pending_termination_closes_owned_process(self):
        path = ROOT / 'preparation/history_utility_revised_materializer_20261112a/run_once.py'
        spec = importlib.util.spec_from_file_location('_revised_spawn_fixture', path)
        supervisor = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervisor)
        signal = supervisor.signal; events = []; process = mock.Mock(); process.poll.return_value = None
        previous_handler = signal.getsignal(signal.SIGTERM)
        def mask(operation, values):
            events.append(('mask', operation, set(values)))
            if operation == signal.SIG_SETMASK and events.count(('spawn',)) == 1:
                events.append(('pending_delivered',)); events.remove(('spawn',))
                handler = signal.getsignal(signal.SIGTERM)
                self.assertNotEqual(handler, previous_handler)
                handler(signal.SIGTERM, None)
            return set()
        def spawn(command, **kwargs):
            self.assertEqual(events[0], ('mask', signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT}))
            self.assertTrue(kwargs['start_new_session']); self.assertTrue(callable(kwargs['preexec_fn']))
            events.append(('spawn',)); return process
        with mock.patch.object(signal, 'pthread_sigmask', side_effect=mask), \
             mock.patch.object(supervisor.subprocess, 'Popen', side_effect=spawn), \
             mock.patch.object(supervisor, 'stop_owned') as stop:
            with self.assertRaisesRegex(InterruptedError, 'supervisor received'):
                with supervisor.spawned_owned(['fixture']): self.fail('pending TERM must precede body')
        stop.assert_called_once_with(process, grace=30)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)
        self.assertIn(('pending_delivered',), events)

    def test_owned_RSS_ignores_exited_zombies(self):
        path = ROOT / 'preparation/history_utility_revised_materializer_20261112a/run_once.py'
        spec = importlib.util.spec_from_file_location('_revised_zombie_fixture', path)
        supervisor = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervisor)
        class Proc:
            def __init__(self, name, state): self.name, self.state = name, state
            def __truediv__(self, child):
                if child == 'stat': text = f'{self.name} (fixture) {self.state} 1 999 0 0'
                else:
                    if self.state == 'Z': raise AssertionError('zombie RSS should not be read')
                    text = 'VmRSS:\t20 kB\n'
                return SimpleNamespace(read_text=lambda: text)
        root = SimpleNamespace(iterdir=lambda: [Proc('12', 'Z'), Proc('13', 'S')])
        with mock.patch.object(supervisor, 'Path', return_value=root):
            self.assertEqual(supervisor.group_memory(999), [{'pid': 13, 'rss_bytes': 20480}])


if __name__ == '__main__': unittest.main()
