import copy
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_utility_headroom import RULE, SCHEMA, assess_records


class HeadroomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = {'schema': SCHEMA, 'rule': RULE, 'dev_source_ids': list('abcdef'),
                        'profile_ids': ['stable', 'smooth', 'abrupt', 'identity', 'near_clean']}
        cls.rows = []
        for s in cls.protocol['dev_source_ids']:
            for p in cls.protocol['profile_ids']:
                for t in range(96):
                    avail = [int(t >= 2), int(t >= 1)]
                    u = np.zeros((2, 6, 6))
                    for j, v in enumerate(avail):
                        if v:
                            u[j, :, :3] = .001
                            u[j, :, 3:] = -.001
                    cls.rows.append({'source_id': s, 'profile_id': p, 'frame_index': t,
                        'crop_tlhw': [192, 368, 96, 96], 'availability': avail,
                        'all_history_mae': .01, 'current_only_mae': .011, 'raw_utility': u.tolist()})

    def test_complete_support_and_boundary_counts(self):
        r = assess_records(self.rows, self.protocol)
        self.assertTrue(r['headroom_rule_passed'])
        self.assertFalse(r['student_training_admitted'])
        self.assertEqual(r['valid_tiles'], 6 * 5 * (94 + 95) * 36)

    def test_no_history_gain_fails_without_relabeling_quality(self):
        rows = [dict(x, current_only_mae=.009) for x in self.rows]
        r = assess_records(rows, self.protocol)
        self.assertFalse(r['headroom_rule_passed'])
        self.assertFalse(r['gates']['source_balanced_degraded_gain'])

    def test_degenerate_labels_fail(self):
        rows = [dict(x, raw_utility=np.zeros((2, 6, 6)).tolist()) for x in self.rows]
        r = assess_records(rows, self.protocol)
        self.assertFalse(r['headroom_rule_passed'])
        self.assertEqual(r['neutral_tile_fraction'], 1.)

    def test_source_equal_not_dominated_by_single_source(self):
        rows = [dict(x, current_only_mae=.05 if x['source_id'] == 'a' else .009) for x in self.rows]
        r = assess_records(rows, self.protocol)
        self.assertTrue(r['gates']['source_balanced_degraded_gain'])
        self.assertFalse(r['gates']['source_support'])

    def test_partial_duplicate_and_unavailable_rejected(self):
        for rows in (self.rows[:-1], self.rows[:-1] + [self.rows[0]]):
            with self.assertRaises(ValueError): assess_records(rows, self.protocol)
        rows = copy.deepcopy(self.rows)
        rows[0]['raw_utility'][0][0][0] = 1.
        with self.assertRaises(ValueError): assess_records(rows, self.protocol)

    def test_rule_change_rejected(self):
        p = copy.deepcopy(self.protocol)
        p['rule']['minimum_mean_degraded_mae_gain'] = 0.
        with self.assertRaises(ValueError): assess_records(self.rows, p)


if __name__ == '__main__':
    unittest.main()
