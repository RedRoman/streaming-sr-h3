import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import diagnose_cached_masks as D

def errors(values):return np.stack([np.full((6,6),v,np.float64) for v in values])
class DiagnosticTests(unittest.TestCase):
    def test_exact_ties_fixed_orders_and_loo_keeps(self):
        s=D.diagnostic_tiles(errors([1,1,1,1]),np.ones((6,6)),[1,1])
        self.assertEqual(s['selection_fraction']['oracle4']['00'],1.)
        self.assertEqual(s['selection_fraction']['oracle5']['input'],1.)
        self.assertEqual(s['selection_fraction']['loo_zero']['11'],1.)
        self.assertEqual(s['loo_zero_differs_from_oracle4_fraction'],1.)
        self.assertEqual(s['loo_zero_strict_positive_regret_fraction'],0.)
    def test_joint_removal_can_harm_despite_both_loo_signs(self):
        s=D.diagnostic_tiles(errors([3,.5,.5,1]),np.full((6,6),4.),[1,1])
        self.assertEqual(s['selection_fraction']['loo_zero']['00'],1.)
        self.assertEqual(s['selection_fraction']['oracle4']['10'],1.)
        self.assertEqual(s['mae']['loo_zero'],3.);self.assertEqual(s['regret_vs_oracle4']['loo_zero'],2.5)
        self.assertEqual(s['interaction']['signed_mean'],3.)
    def test_joint_removal_can_help_despite_both_loo_keep_signs(self):
        s=D.diagnostic_tiles(errors([0,2,2,1]),np.full((6,6),3.),[1,1])
        self.assertEqual(s['selection_fraction']['loo_zero']['11'],1.)
        self.assertEqual(s['selection_fraction']['oracle4']['00'],1.)
        self.assertEqual(s['regret_vs_oracle4']['loo_zero'],1.);self.assertEqual(s['interaction']['signed_mean'],-3.)
    def test_boundary_candidates_and_interaction_exclusion(self):
        s=D.diagnostic_tiles(errors([.5,0,.2,.2]),np.ones((6,6)),[0,1])
        self.assertEqual(s['selection_fraction']['oracle4']['01'],1.);self.assertEqual(s['feasible_mask_fraction']['10'],0.);self.assertIsNone(s['interaction'])
        s=D.diagnostic_tiles(errors([.5,0,0,0]),np.ones((6,6)),[0,0])
        self.assertEqual(s['selection_fraction']['oracle4']['00'],1.);self.assertEqual(s['selection_fraction']['loo_zero']['00'],1.);self.assertIsNone(s['interaction'])
    def test_primary_zero_and_secondary_margin_are_distinct(self):
        s=D.diagnostic_tiles(errors([1.2,.9999,.9999,1]),np.ones((6,6)),[1,1])
        self.assertEqual(s['selection_fraction']['loo_zero']['00'],1.);self.assertEqual(s['selection_fraction']['loo_margin']['11'],1.)
    def test_raw_input_secondary_does_not_change_four_mask_result(self):
        s=D.diagnostic_tiles(errors([.125,.125,.125,.125]),np.zeros((6,6)),[1,1])
        self.assertEqual(s['mae']['oracle4'],.125);self.assertEqual(s['mae']['oracle5'],0.);self.assertEqual(s['oracle5_gain_over_oracle4'],.125)
        self.assertEqual(s['selection_fraction']['oracle5']['input'],1.);self.assertEqual(s['harm_vs_input']['oracle4'],.125)
        with self.assertRaises(ValueError):D.diagnostic_tiles(errors([np.nan,0,0,0]),np.zeros((6,6)),[1,1])
    def test_equal_source_and_profile_summary(self):
        p={'dev_source_ids':[str(i) for i in range(6)],'profile_ids':['a','b','c','identity','near_clean']};rows=[]
        for i,sid in enumerate(p['dev_source_ids']):
            for j,pid in enumerate(p['profile_ids']):
                for t in range(96):rows.append({'source_id':sid,'profile_id':pid,'statistics':{'value':float(i*10+j),'interaction':None if t<2 else {'signed_mean':float(i+j)}}})
        result=D.summarize(rows,p)
        self.assertEqual(result['primary_degraded_equal_source_equal_profile']['value'],26.)
        self.assertEqual(result['identity_equal_source']['value'],28.)
        self.assertEqual(result['near_clean_equal_source']['value'],29.)
        self.assertTrue(all(r['interaction_frames']==94 for r in result['per_source_profile']))
if __name__=='__main__':unittest.main(verbosity=2)
