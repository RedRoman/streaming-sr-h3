"""CPU-only regressions; no H3, model weights or accelerator runtime needed.

Run: python -m unittest discover -s code/tests -p 'test_restoration_metrics.py' -v
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import struct

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restoration_metrics import (METRIC_SCHEMA_VERSION, SCALAR_ALIASES, add_aliases,
                                 assert_compatible_output, band_error_terms,
                                 flow_visibility_mask, masked_warp_mae, metric_metadata,
                                 paired_counts, remap_target_to_source)


class FlowMetricTests(unittest.TestCase):
    def test_known_translation_requires_pulling_second_frame_back(self):
        rng = np.random.default_rng(10)
        a = rng.random((16, 20, 3), dtype=np.float32)
        b = np.zeros_like(a)
        b[:, 2:] = a[:, :-2]
        fw = np.zeros((16, 20, 2), dtype=np.float32)
        fw[..., 0] = 2
        bw = -fw
        mask = flow_visibility_mask(fw, bw)
        self.assertTrue(mask[:, :-2].all())
        self.assertFalse(mask[:, -2:].any())
        self.assertAlmostEqual(masked_warp_mae(a, b, fw, mask), 0, places=7)
        old_wrong_error = np.abs(remap_target_to_source(a, fw) - b)[mask].mean()
        self.assertGreater(old_wrong_error, .1)

    def test_identity_and_consistency_reject_occluded_region(self):
        fw = np.zeros((10, 12, 2), dtype=np.float32)
        bw = fw.copy()
        self.assertTrue(flow_visibility_mask(fw, bw).all())
        bw[2:5, 3:7, 0] = 4
        mask = flow_visibility_mask(fw, bw)
        self.assertFalse(mask[2:5, 3:7].any())
        self.assertEqual(int(mask.sum()), 120 - 12)

    def test_invalid_flow_and_empty_visibility(self):
        fw = np.full((4, 5, 2), 100, dtype=np.float32)
        fw[0, 0] = np.nan
        mask = flow_visibility_mask(fw, -fw)
        self.assertFalse(mask.any())
        img = np.zeros((4, 5, 3), dtype=np.float32)
        self.assertIsNone(masked_warp_mae(img, img, fw, mask))

    def test_mask_is_applied_to_prediction_and_gt_equally(self):
        gt = np.zeros((5, 6, 3), dtype=np.float32)
        pred = gt.copy()
        pred[0, :, :] = 100
        mask = np.ones((5, 6), dtype=bool)
        mask[0] = False
        zero = np.zeros((5, 6, 2), dtype=np.float32)
        self.assertEqual(masked_warp_mae(gt, pred, zero, mask), 0)


class BandErrorTests(unittest.TestCase):
    def test_nonorthogonal_split_needs_cross_term(self):
        gt = np.zeros((2, 3))
        pred = np.ones_like(gt)
        d = band_error_terms(gt, pred, .8 * gt, .8 * pred)
        self.assertAlmostEqual(d['mse_actual'], 1)
        self.assertAlmostEqual(d['mse_lowband'] + d['mse_detail'], .68)
        self.assertAlmostEqual(d['mse_cross_term'], .32)
        self.assertAlmostEqual(d['mse_reconstructed'], d['mse_actual'])
        self.assertLess(d['mse_identity_abs_error'], 1e-12)

    def test_oracle_lowband_replacement_can_worsen_mse(self):
        from error_decomposition import describe_terms
        gt = np.full((2, 3), .2)
        pred = np.full_like(gt, .4)
        d = describe_terms(band_error_terms(gt, pred, 3 * gt, 3 * pred))
        self.assertLess(d['mse_cross_term'], 0)
        self.assertAlmostEqual(d['mse_actual'], .04)
        self.assertAlmostEqual(d['mse_gt_lowband_substitution'], .16)
        self.assertAlmostEqual(d['mse_gt_lowband_substitution_clipped'], .04)
        self.assertLess(d['headroom_dB'], 0)
        self.assertAlmostEqual(d['psnr'], -10 * np.log10(.04))

    def test_actual_bicubic_band_identity(self):
        import torch
        from error_decomposition import split
        rng = np.random.default_rng(11)
        gt = torch.from_numpy(rng.random((1, 16, 24, 3), dtype=np.float32))
        pred = (gt + .1 * torch.from_numpy(rng.standard_normal(gt.shape).astype(np.float32))).clamp(0, 1)
        gl, _ = split(gt)
        pl, _ = split(pred)
        d = band_error_terms(gt.numpy(), pred.numpy(), gl.numpy(), pl.numpy())
        self.assertLess(d['mse_identity_abs_error'], 1e-12)
        self.assertGreater(abs(d['mse_cross_term']), 1e-7)


class ResultContractTests(unittest.TestCase):
    def test_metric_directions_and_non_quality_diagnostics(self):
        deltas = [-2, 0, 1, 3]
        self.assertEqual(paired_counts(deltas, 'psnr')['a_better'], 2)
        self.assertEqual(paired_counts(deltas, 'null_cos')['a_better'], 2)
        self.assertEqual(paired_counts(deltas, 'lpips')['a_better'], 1)
        self.assertEqual(paired_counts(deltas, 'p8.terr_null')['a_better'], 1)
        d = paired_counts(deltas, 'null_energy')
        self.assertIsNone(d['a_better'])
        self.assertEqual((d['a_lower'], d['a_higher'], d['ties']), (1, 2, 1))

    def test_aliases_are_exact_and_schema_warns_against_mixing(self):
        old = {'null_energy': .8, 'null_cos': .4, 'psnr_gtrange': 22, 'lpips_gtrange': .2}
        new = add_aliases(old.copy(), SCALAR_ALIASES)
        for key, alias in SCALAR_ALIASES.items():
            self.assertEqual(new[key], new[alias])
        self.assertFalse(metric_metadata()['compatible_with_unversioned_results'])
        self.assertEqual(metric_metadata()['version'], METRIC_SCHEMA_VERSION)

    def test_historical_result_cannot_be_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'judge.json'
            original = json.dumps({'summary': {'lpips': .2}})
            path.write_text(original)
            with self.assertRaisesRegex(ValueError, 'Historical metrics cannot be upgraded'):
                assert_compatible_output(path)
            self.assertEqual(path.read_text(), original)
            path.write_text(json.dumps({'metric_schema_version': METRIC_SCHEMA_VERSION}))
            assert_compatible_output(path)

    def test_static_absolute_errors_are_retained_and_paired(self):
        from deploy_eval import paired, summarise
        def arm(error):
            return {'psnr': 20., 'lpips': .2,
                    'p8': {'static': True, 'terr_null': error, 'terr_range': error / 2,
                           'warp': None, 'flicker': None}}
        rows = [{'clip': 'static', 'degradation': 'realistic', 'leaked': False,
                 'arms': {'a': arm(.1), 'b': arm(.2)}}]
        summary = summarise(rows, ['a'])
        self.assertEqual(summary['all']['a']['p8.terr_null']['n'], 1)
        self.assertNotIn('p8.warp', summary['all']['a'])
        comparison = paired(rows, 'a', 'b', ['p8.terr_null', 'p8.warp', 'psnr'])
        self.assertEqual(comparison['p8.terr_null']['a_better'], 1)
        self.assertNotIn('p8.warp', comparison)
        self.assertEqual(comparison['psnr']['ties'], 1)

    def test_rgb_reader_interface_uses_original_target(self):
        import torch
        from unittest.mock import patch
        from deploy_eval import _eval
        import revisit_metrics
        gt = torch.full((5, 16, 16, 3), .2)
        pred = torch.full_like(gt, .3)
        with patch.object(revisit_metrics, 'lpips_dist', side_effect=lambda x, y: float((x-y).square().mean())):
            result = _eval(('rgb', {'model': 'model'}), frame_reader=lambda arm, name, t: pred[t],
                           gt_reader=lambda t: gt[t], clip_meta={'frames': 5, 'degradation': {'kind': 'realistic'}},
                           frame_stride=2)
        self.assertAlmostEqual(result['arms']['model']['psnr'], 20, places=5)
        self.assertEqual(result['arms']['model']['n_frames'], 3)
        self.assertEqual(result['frame_stride'], 2)


class PilotInputContractTests(unittest.TestCase):
    def test_w0_source_and_rgb_latent_metadata_are_checked(self):
        from run_openh3_pilot_eval import select_dev_windows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = {'id': 'source', 'start_s': .5, 'degradation': {'kind': 'realistic'}}
            header = {'guide_raw': {'shape': [1, 24, 7, 1, 1]}, 'target': {'shape': [1, 24, 7, 1, 1]},
                      '__metadata__': {'meta': json.dumps(meta)}}
            payload = json.dumps(header).encode()
            latent = root / 'source_w0.safetensors'
            latent.write_bytes(struct.pack('<Q', len(payload)) + payload)
            manifest = root / 'dev.json'
            manifest.write_text(json.dumps({'samples': [{'source_id': 'source', 'latent_path': str(latent)}]}))
            hr = np.zeros((22, 16, 16, 3), dtype=np.uint8)
            lr = np.zeros((22, 4, 4, 3), dtype=np.uint8)
            rgb = root / 'source_w0.npz'
            np.savez(rgb, hr=hr, lr=lr, dx=lr, meta=json.dumps(meta))
            selected = select_dev_windows(manifest, root, root, 1)
            self.assertEqual(selected[0]['source_id'], 'source')
            self.assertEqual(len(selected[0]['rgb_sha256']), 64)
            meta['start_s'] = .7
            np.savez(rgb, hr=hr, lr=lr, dx=lr, meta=json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'start_s mismatch'):
                select_dev_windows(manifest, root, root, 1)


if __name__ == '__main__':
    unittest.main()
