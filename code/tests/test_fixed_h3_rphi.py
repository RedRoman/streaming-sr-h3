import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eval_rphi import TASKS, canonical_sha, causal_indices, sample_id, sha_file
from export_rphi_guides import array_sha
from export_rphi_guides_stage2 import verify_checkpoint_implementation
from run_fixed_h3_rphi import contained_file, generate_pair, inspect_inputs, main, validate_h3_geometry
from test_rphi_export import fixture


def export_fixture(root):
    manifest, _ = fixture(root / 'data')
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    directory = root / 'export'
    directory.mkdir()
    export = {'schema': 'rphi.guide_export.v1', 'complete': True, 'samples': 4,
              'source_ids': ['dev-source'], 'checkpoint_sha256': 'a' * 64,
              'model_config': {'variant': 'blind', 'observation_encoder': 'native', 'k': 3},
              'checkpoint_step': 3000,
              'checkpoint_implementation_sha256': {'code/rphi_condition_v2.py': 'c' * 64},
              'stage2_export_proof': {'checkpoint_sha256': 'a' * 64,
                  'inference_implementation_verified': True,
                  'implementation_sha256': {'code/rphi_condition_v2.py': 'c' * 64}},
              'condition_mode': 'observations_only', 'dev_manifest_sha256': sha_file(manifest),
              'training_contract': {'dev_manifest_sha256': sha_file(manifest),
                  'target': 'clean_original_recording', 'task_order': list(TASKS), 'steps': 3000,
                  'train_source_ids': ['train-source'], 'train_source_sha256': ['b' * 64]},
              'per_sample': []}
    for record in records:
        sid = sample_id(record)
        rgb = np.full((22, 8, 12, 3), .5, np.float32)
        arrays = {'rgb': rgb, 'raw_rgb': rgb}
        path = directory / (sid + '.npz')
        np.savez_compressed(path, **arrays)
        row = {key: record[key] for key in ('source_id', 'source_sha256', 'target_key', 'task',
            'split', 'target_pixel_sha256', 'observation_pixel_sha256')}
        row.update({key: export[key] for key in ('checkpoint_sha256', 'model_config',
            'condition_mode', 'dev_manifest_sha256')})
        row.update(sample_id=sid, record_sha256=canonical_sha(record),
            arrays={key: {'shape': list(value.shape), 'dtype': 'float32',
                          'array_sha256': array_sha(value)} for key, value in arrays.items()},
            guide_npz=path.name, guide_file_sha256=sha_file(path), frame_indices=list(range(22)),
            history_indices=causal_indices(list(range(22)), 3, 22).tolist(),
            target_file_sha256=sha_file(record['target_npz']),
            observation_file_sha256=sha_file(record['observation_npz']))
        sidecar = directory / (sid + '.json')
        sidecar.write_text(json.dumps(row))
        export['per_sample'].append({**row, 'provenance_json': sidecar.name,
                                     'provenance_file_sha256': sha_file(sidecar)})
    path = directory / 'manifest.json'
    path.write_text(json.dumps(export))
    return path, manifest, export


class InputContractTests(unittest.TestCase):
    def test_complete_export_and_content_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path, manifest, export = export_fixture(Path(temp))
            _, _, rows = inspect_inputs(path, manifest)
            self.assertEqual(len(rows), 4)
            guide = Path(rows[0]['guide_file'])
            guide.write_bytes(guide.read_bytes() + b'tamper')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                inspect_inputs(path, manifest)

    def test_missing_task_and_foreign_record_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, manifest, export = export_fixture(Path(temp))
            export['per_sample'].pop()
            export['samples'] = 3
            path.write_text(json.dumps(export))
            with self.assertRaisesRegex(ValueError, 'every task/window'):
                inspect_inputs(path, manifest)
            export['per_sample'][0]['record_sha256'] = 'c' * 64
            path.write_text(json.dumps(export))
            with self.assertRaisesRegex(ValueError, 'record mismatch'):
                inspect_inputs(path, manifest)

    def test_clean_target_cannot_replace_observation_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            path, manifest, export = export_fixture(Path(temp))
            export['per_sample'][0]['observation_pixel_sha256'] = export['per_sample'][0]['target_pixel_sha256']
            path.write_text(json.dumps(export))
            with self.assertRaisesRegex(ValueError, 'source binding mismatch'):
                inspect_inputs(path, manifest)

    def test_missing_frames_and_known_condition_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, manifest, export = export_fixture(Path(temp))
            export['condition_mode'] = 'known'
            path.write_text(json.dumps(export))
            with self.assertRaisesRegex(ValueError, 'observations-only'):
                inspect_inputs(path, manifest)
            export['condition_mode'] = 'observations_only'
            row = export['per_sample'][0]
            row['frame_indices'] = list(range(0, 22, 4))
            sidecar = path.parent / row['provenance_json']
            sidecar.write_text(json.dumps({k: v for k, v in row.items()
                                          if k not in ('provenance_json', 'provenance_file_sha256')}))
            row['provenance_file_sha256'] = sha_file(sidecar)
            path.write_text(json.dumps(export))
            with self.assertRaisesRegex(ValueError, 'every consecutive'):
                inspect_inputs(path, manifest)

    def test_path_escape_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'contained'):
                contained_file(root, '../outside.npz')
            with self.assertRaises(FileExistsError):
                main(['--guide-manifest', 'absent', '--dev-manifest', 'absent',
                      '--pretrained-manifest', 'absent', '--out', temp])
        validate_h3_geometry((22, 480, 832, 3))
        with self.assertRaisesRegex(ValueError, 'divisible by 32'):
            validate_h3_geometry((22, 16, 32, 3))

    def test_early_checkpoint_and_unbound_native_code_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, manifest, export = export_fixture(Path(temp))
            export['checkpoint_step'] = 250
            path.write_text(json.dumps(export))
            with self.assertRaisesRegex(ValueError, 'final contracted'):
                inspect_inputs(path, manifest)
            export['checkpoint_step'] = 3000
            export['stage2_export_proof']['implementation_sha256'] = {}
            path.write_text(json.dumps(export))
            with self.assertRaisesRegex(ValueError, 'encoder SHA'):
                inspect_inputs(path, manifest)

    def test_exporter_rejects_model_code_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ['rphi_model.py', 'rphi_data.py', 'rphi_condition_v2.py']
            for name in names:
                (root / name).write_text(name)
            payload = {'step': 3000, 'training_contract': {'steps': 3000},
                       'model_config': {'variant': 'blind', 'k': 3, 'observation_encoder': 'native'},
                       'implementation_sha256': {'code/' + name: sha_file(root / name) for name in names}}
            self.assertEqual(len(verify_checkpoint_implementation(payload, root)), 3)
            (root / 'rphi_condition_v2.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'implementation mismatch'):
                verify_checkpoint_implementation(payload, root)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'existing CPU torch required')
class FixedPairTests(unittest.TestCase):
    def test_failed_post_export_does_not_leave_complete_manifest(self):
        import torch
        import export_rphi_guides_stage2 as wrapper
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'checkpoint.pt'
            torch.save({}, checkpoint)
            out = root / 'output'
            def fake_export(**kwargs):
                out.mkdir()
                manifest = {'complete': True, 'checkpoint_sha256': 'wrong'}
                (out / 'manifest.json').write_text(json.dumps(manifest))
                return manifest
            with patch.object(wrapper, 'verify_checkpoint_implementation', return_value={}), \
                 patch.object(wrapper, 'export_guides', side_effect=fake_export):
                with self.assertRaisesRegex(ValueError, 'different checkpoint'):
                    wrapper.main(['--checkpoint', str(checkpoint), '--dev-manifest', 'unused', '--out', str(out)])
            self.assertFalse(json.loads((out / 'manifest.json').read_text())['complete'])
            self.assertTrue((out / 'stage2_failure.json').exists())

    def fake_h3(self, mutate=False):
        import torch
        class FakeH3:
            calls = []
            def encode_guide_pixels(self, rgb, hw):
                return torch.full((1, 24, 7, hw[0] // 16, hw[1] // 16), float(rgb.mean()))
            def sample(self, guide, mode, **kw):
                self.calls.append((guide.clone(), kw['warm_start'].clone(), mode, kw['seed'], kw['audio_init']))
                if mutate:
                    kw['warm_start'].add_(1)
                return {'video': guide}, {'mock': True}
            def decode(self, latent):
                return torch.full((22, 32, 32, 3), float(latent.mean()))
        return FakeH3()

    def test_fixed_draft_same_seed_only_guide_changes(self):
        import torch
        raw, rphi = torch.full((22, 32, 32, 3), .2), torch.full((22, 32, 32, 3), .8)
        h3 = self.fake_h3()
        outputs, report = generate_pair(h3, raw, rphi, 101)
        self.assertEqual(report['h3_sample_calls'], 2)
        self.assertFalse(torch.equal(h3.calls[0][0], h3.calls[1][0]))
        self.assertTrue(torch.equal(h3.calls[0][1], h3.calls[1][1]))
        self.assertEqual(h3.calls[0][2:], ('base', 101, 'zeros'))
        self.assertEqual(h3.calls[0][2:], h3.calls[1][2:])
        self.assertEqual(report['calls']['raw']['warm_start_latent_sha256_float32'],
                         report['calls']['rphi']['warm_start_latent_sha256_float32'])
        self.assertTrue(torch.allclose(outputs['raw'], raw))
        self.assertTrue(torch.allclose(outputs['rphi'], rphi))

    def test_mutated_draft_and_geometry_fail_closed(self):
        import torch
        rgb = torch.full((22, 32, 32, 3), .5)
        with self.assertRaisesRegex(ValueError, 'mutated'):
            generate_pair(self.fake_h3(mutate=True), rgb, rgb, 101)
        with self.assertRaisesRegex(ValueError, '22 frames'):
            generate_pair(self.fake_h3(), rgb[:6], rgb[:6], 101)
        with self.assertRaisesRegex(ValueError, 'geometries differ'):
            generate_pair(self.fake_h3(), rgb, rgb[:6], 101)
        with self.assertRaisesRegex(ValueError, 'divisible by 32'):
            generate_pair(self.fake_h3(), rgb[:, :16], rgb[:, :16], 101)


if __name__ == '__main__':
    unittest.main()
