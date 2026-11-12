"""Synthetic fullshape NPZ/visual evidence, never real cohort continuity labels."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import history_utility_data as D
import history_utility_revised_data as R
import test_history_utility_data as F


class RevisedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse frozen synthetic fixture creation only, not its prior test suite.
        F.HistoryUtilityDataTests.setUpClass()
        cls.fixture = F.HistoryUtilityDataTests
        cls.root = cls.fixture.directory
        cls.registry = D.load_source_registry(D._json(F.PLAN), F.TRAIN, F.DEV)
        cls.sid = cls.fixture.sid
        cls.dev_sid = next(s for s, r in cls.registry.items() if r['split'] == 'dev')
        for name in ('target96.png', 'candidate_pairs.png'):
            (cls.root / name).write_bytes(('synthetic hash fixture ' + name).encode())
        images = [{'path': str(cls.root / n), 'sha256': D.file_sha(cls.root / n)}
                  for n in ('target96.png', 'candidate_pairs.png')]
        original = cls.fixture.receipt
        cls.numerical = {'schema': 'openh3.history_utility_target_only_receipt.v1',
            'complete': True, 'status': 'target_only_complete_not_observation_dataset',
            'source': original['source'], 'window_key': original['window_key'],
            'target': original['target'], 'target_specification': original['target_specification'],
            'numerical_window_contract_file_sha256': R.NUMERIC_WINDOW_SHA,
            'decode': {'decoded_target_count': 96, 'decoded_source_count': 96, 'timestamps': {
                name: [{'index': i, 'pts': i + 1, 'time_base': '1/24'} for i in range(96)]
                for name in ('target', 'source')}},
            'diagnostics': {'artifacts': {'all_frame_sheet': images[0], 'ranked_candidate_pairs': images[1]}}}
        cls.visual = {'policy_sha256': R.POLICY_SHA, 'rows': [{
            **original['source'], 'window_key': original['window_key'],
            'target_pixel_sha256': original['target']['pixel_sha256'],
            'target_file_sha256': original['target']['file_sha256'],
            'decision': 'primary', 'reviewed_frames': [0, 95],
            'reviewed_ranked_pairs': 'all6 displayed in candidate_pairs.png', 'images_reviewed': images}]}
        num = cls.write('numerical.json', cls.numerical)
        visual = cls.write('visual.json', cls.visual)
        decisions = {sid: {**binding, 'disposition': 'diagnostic', 'window_key': None,
                          'target_pixel_sha256': None} for sid, binding in cls.registry.items()}
        for sid in (cls.sid, cls.dev_sid):
            decisions[sid].update(disposition='primary', window_key=original['window_key'],
                target_pixel_sha256=original['target']['pixel_sha256'], target_receipt=num,
                visual_review={**visual, 'decision_key': 'rows'})
        # The dev row only tests exact full-registry accounting. This train-only
        # fixture intentionally cannot claim dev physical/visual readiness.
        cls.cohort = {'schema': R.SCHEMA, 'policy_sha256': R.POLICY_SHA,
            'decisions': decisions, 'primary_source_ids': {'train': [cls.sid], 'dev': [cls.dev_sid]}}
        ineligible = [sid for sid in cls.registry if sid not in (cls.sid, cls.dev_sid)][-2:]
        audit_rows = []
        for sid, binding in cls.registry.items():
            ok = sid not in ineligible
            if not ok: decisions[sid]['disposition'] = 'ineligible'
            audit_rows.append({**binding, 'state': 'target_protocol_eligible_continuity_unreviewed' if ok else 'protocol_ineligible',
                'window_key': original['window_key'], 'target_pixel_sha256': original['target']['pixel_sha256'],
                'numerical_receipt': num})
        audit = {'schema': 'openh3.history_utility_target_screen_actual_terminal_audit.v1',
            'complete': True, 'source_count': 40, 'source_files_actual_SHA_verified': 40,
            'target_protocol_eligible_count': 38, 'ineligible_count': 2,
            'target_files_actual_SHA_and_numpy_pixel_SHA_verified': 38, 'source_rows': audit_rows}
        cls.cohort['target_terminal_audit'] = cls.write('audit.json', audit)
        cls.binding = cls.write('cohort.json', cls.cohort)
        numeric_path = F.ROOT / 'preparation/history_utility_materializer_20261112a/execution_contract.json'
        execution = {'schema': 'openh3.history_utility_revised_materialization_execution.v1',
            'files': {'cohort': cls.binding,
                      'numerical_window_contract': {'path': str(numeric_path), 'sha256': R.NUMERIC_WINDOW_SHA}}}
        cls.authority = {'schema': 'openh3.history_utility_revised_materialization_authority.v1',
            'execution_contract': cls.write('new_execution.json', execution),
            'execution_contract_value_sha256': D.value_sha(execution), 'cohort': cls.binding,
            'input_target_receipt': num, 'input_target': original['target']}
        sidecar = {'schema': 'openh3.history_utility_revised_reuse_admission.v1',
            'reused_receipt': {k: cls.fixture.entry[k] for k in ('path', 'sha256')},
            'execution_authority': cls.authority, 'original_receipt_unchanged': True,
            'target_pixels_unchanged': True, 'source': original['source'],
            'window_key': original['window_key'], 'target_pixel_sha256': original['target']['pixel_sha256']}
        cls.entry = {**cls.fixture.entry, 'revised_authority': cls.write('reuse_admission.json', sidecar)}
        cls.data = cls.dataset()

    @classmethod
    def tearDownClass(cls):
        F.HistoryUtilityDataTests.tearDownClass()

    @classmethod
    def write(cls, name, value):
        path = cls.root / name
        path.write_text(json.dumps(value, sort_keys=True))
        return {'path': str(path), 'sha256': D.file_sha(path)}

    @classmethod
    def dataset(cls, cohort=None, receipts=None):
        return R.RevisedHistoryUtilityDataset(receipts if receipts is not None else [cls.entry],
            cohort=cohort or cls.binding, plan=F.PLAN, train_manifest=F.TRAIN,
            dev_manifest=F.DEV, split='train', evidence_root=cls.root)

    def test_full_registry_exact_membership_no_reassignment(self):
        plan = D._json(F.PLAN)
        R.validate_numeric_plan(plan)
        plan['profiles'][0]['frame_parameters'][0]['frame'] = 1
        with self.assertRaisesRegex(ValueError, 'operator/profile'): R.validate_numeric_plan(plan)
        for mutate in (lambda c: c['decisions'].pop(self.dev_sid),
                       lambda c: c['decisions'][self.sid].update(split='dev'),
                       lambda c: c['primary_source_ids']['train'].append(self.sid),
                       lambda c: c['decisions'][self.sid].update(disposition='diagnostic')):
            c = copy.deepcopy(self.cohort); mutate(c)
            with self.assertRaises(ValueError): R.validate_cohort(c, self.registry)
        with self.assertRaisesRegex(ValueError, 'membership'): self.dataset(receipts=[])
        with self.assertRaisesRegex(ValueError, 'membership'):
            self.dataset(receipts=[self.fixture.entry, self.fixture.entry])

    def test_distinct_readiness_namespace_and_old_gate_unchanged(self):
        self.assertEqual(len(self.data.registry), 40)
        self.assertTrue(self.data.revised_cohort_ready)
        self.assertFalse(self.data.research_ready)
        self.assertFalse(self.data._numeric.research_ready)
        self.assertFalse(self.data.binding['training_admitted'])
        self.assertFalse(self.data.binding['original_32_8_completion_claim'])
        self.assertEqual(D.file_sha(D.__file__), R.LOADER_SHA)
        with self.assertRaisesRegex(ValueError, 'research needs'):
            self.fixture.dataset(scope='research')

    def test_visual_and_numerical_binding_negative_cases(self):
        for field, value in [('window_key', '1' * 64), ('target_pixel_sha256', '2' * 64), ('split', 'dev')]:
            visual = copy.deepcopy(self.visual); visual['rows'][0][field] = value
            decision = copy.deepcopy(self.cohort['decisions'][self.sid])
            decision['visual_review'] = {**self.write('bad_visual.json', visual), 'decision_key': 'rows'}
            with self.assertRaises(ValueError):
                R.validate_primary_evidence(self.sid, decision, self.registry, self.root)
        visual = copy.deepcopy(self.visual); visual['rows'][0]['decision'] = 'diagnostic'
        decision = copy.deepcopy(self.cohort['decisions'][self.sid])
        decision['visual_review'] = self.write('diagnostic_visual.json', visual)
        with self.assertRaisesRegex(ValueError, 'visual review'):
            R.validate_primary_evidence(self.sid, decision, self.registry, self.root)
        numerical = copy.deepcopy(self.numerical); numerical['decode']['timestamps']['target'][2]['pts'] += 1
        decision['visual_review'] = self.cohort['decisions'][self.sid]['visual_review']
        decision['target_receipt'] = self.write('bad_pts.json', numerical)
        with self.assertRaisesRegex(ValueError, 'timestamp grid'):
            R.validate_primary_evidence(self.sid, decision, self.registry, self.root)

    def test_actual_image_bytes_and_receipt_digest_fail_closed(self):
        decision = copy.deepcopy(self.cohort['decisions'][self.sid])
        decision['target_receipt']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'evidence file changed'):
            R.validate_primary_evidence(self.sid, decision, self.registry, self.root)
        path = self.root / 'candidate_pairs.png'; before = path.read_bytes()
        try:
            path.write_bytes(before + b'tamper')
            with self.assertRaisesRegex(ValueError, 'evidence file changed'):
                R.validate_primary_evidence(self.sid, self.cohort['decisions'][self.sid], self.registry, self.root)
        finally: path.write_bytes(before)

    def test_real_npz_verification_not_metadata_only(self):
        path = self.root / 'target.npz'; before = path.read_bytes()
        try:
            path.write_bytes(before + b'tamper')
            with self.assertRaisesRegex(ValueError, 'NPZ bytes/SHA'): self.dataset()
        finally: path.write_bytes(before)

    def test_both_1000step_schedules_exact_v1_numerics(self):
        for role in ('teacher', 'student'):
            revised = self.data.schedule(role)
            legacy = self.data._numeric.schedule(role, 1000)
            self.assertEqual(revised['numeric_schedule'], legacy)
            self.assertEqual(len(legacy['rows']), 16000)
            self.assertEqual(revised['seed'], D.SEEDS[role])
            self.assertFalse(revised['training_admitted'])
            self.assertFalse(legacy['research_ready'])
            for start in (0, 15984):
                profiles = [r['profile_id'] for r in legacy['rows'][start:start + 16]]
                for pid in D.PROFILES: self.assertEqual(profiles.count(pid), 4)
                self.assertEqual(profiles.count('identity'), 2)
                self.assertEqual(profiles.count('near_clean'), 2)
        with self.assertRaisesRegex(ValueError, 'exactly1000'): self.data.schedule(steps=1)

    def test_crop_history_and_teacher_masks_bitexact_no_oracle(self):
        rows = [self.data._numeric._row(self.sid, pid, t, [4, 8, 96, 96], [0, 1],
                {'kind': 'local_soft', 'seed': 19}, index)
                for index, (pid, t) in enumerate([(D.PROFILES[0], 0), ('identity', 1), ('near_clean', 2)])]
        a, b = self.data.batch(rows), self.fixture.data.batch(rows)
        self.assertEqual(set(a['model_inputs']), {'observed', 'availability'})
        for key in a['model_inputs']:
            np.testing.assert_array_equal(a['model_inputs'][key], b['model_inputs'][key])
        for key in ('target', 'control', 'teacher_weights'):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(a['metadata'], b['metadata'])

    def test_relative_evidence_path_no_fallback(self):
        relative = {'path': 'cohort.json', 'sha256': self.binding['sha256']}
        data = self.dataset(cohort=relative)
        self.assertEqual(data.dataset_sha256, self.data.dataset_sha256)
        with tempfile.TemporaryDirectory() as missing:
            with self.assertRaises(FileNotFoundError):
                R.bound_json(relative, missing)

    def test_pair_scope_requires_both_splits_and_same_cohort(self):
        dev = SimpleNamespace(split='dev', revised_cohort_ready=True,
            cohort_sha256=self.data.cohort_sha256, registry=self.data.registry,
            source_ids=[self.dev_sid])
        result = R.assert_revised_pair(self.data, dev)
        self.assertTrue(result['revised_pair_ready'])
        self.assertFalse(result['training_admitted'])
        dev.cohort_sha256 = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'cohort'): R.assert_revised_pair(self.data, dev)
        dev.cohort_sha256 = self.data.cohort_sha256
        dev.source_ids = [self.sid]
        with self.assertRaisesRegex(ValueError, 'crosses'): R.assert_revised_pair(self.data, dev)

    def test_old_reuse_and_new_inline_authority_are_separate(self):
        with self.assertRaisesRegex(ValueError, 'sidecar'):
            self.dataset(receipts=[self.fixture.entry])
        receipt = copy.deepcopy(self.fixture.receipt)
        receipt.update(producer_schema='openh3.history_utility_revised_materialization.v1',
            revised_cohort_sha256=self.binding['sha256'], execution_authority=self.authority)
        kwargs = dict(cohort=self.cohort, cohort_sha256=self.binding['sha256'],
            decision=self.cohort['decisions'][self.sid], evidence=self.data.evidence[self.sid], root=self.root)
        kind, authority = R.validate_source_authority(self.fixture.entry, receipt, **kwargs)
        self.assertEqual(kind, 'new_materialization')
        self.assertEqual(authority, self.authority)
        with self.assertRaisesRegex(ValueError, 'NPZ lacks'):
            R.validate_new_metadata(self.data._numeric.sources[self.sid], authority, self.binding['sha256'])
        receipt['execution_authority'] = copy.deepcopy(self.authority)
        receipt['execution_authority']['execution_contract_value_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'authority changed'):
            R.validate_source_authority(self.fixture.entry, receipt, **kwargs)

    def test_terminal_audit_eligibility_and_receipt_binding(self):
        R.validate_terminal_audit(self.cohort, self.registry, self.root)
        c = copy.deepcopy(self.cohort)
        a = D._json(c['target_terminal_audit']['path']); a['source_rows'][0]['split'] = 'invalid'
        c['target_terminal_audit'] = self.write('bad_audit.json', a)
        with self.assertRaisesRegex(ValueError, 'source/role'):
            R.validate_terminal_audit(c, self.registry, self.root)


if __name__ == '__main__': unittest.main()
