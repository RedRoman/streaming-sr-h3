import unittest
import numpy as np
from diagnose_rphi_residual_mix import scan_mix, metric_summary, assert_quality_match


class ResidualMixTests(unittest.TestCase):
    def test_analytic_pooled_mse_oracle_and_endpoints(self):
        target=np.full((2,2,2,3),102,np.uint8)
        r=np.full(target.shape,.2,np.float32);g=np.full(target.shape,.6,np.float32)
        out=scan_mix(r,g,target)
        self.assertAlmostEqual(out['oracle']['t'],.5,places=6)
        self.assertLess(out['oracle']['scored']['post_pixel_clip_float32']['mse'],1e-14)
        self.assertAlmostEqual(out['grid'][0]['pre_pixel_clip_float64']['mse'],.04,places=6)
        self.assertAlmostEqual(out['grid'][-1]['pre_pixel_clip_float64']['mse'],.04,places=6)
        for row in out['grid']:
            self.assertAlmostEqual(row['analytic_preclip_pooled_mse'],row['pre_pixel_clip_float64']['mse'],places=12)

    def test_clipping_is_separate_from_oracle_objective(self):
        target=np.full((1,2,2,3),255,np.uint8)
        r=np.zeros(target.shape,np.float32);g=np.full(target.shape,2,np.float32)
        out=scan_mix(r,g,target)
        self.assertEqual(out['oracle']['t'],.5)
        last=out['grid'][-1]
        self.assertEqual(last['clipped_component_fraction'],1)
        self.assertEqual(last['pre_pixel_clip_float64']['mse'],1)
        self.assertEqual(last['post_pixel_clip_float32']['mse'],0)

    def test_zero_residual_and_harm_choose_zero(self):
        target=np.full((1,2,2,3),51,np.uint8)
        r=np.full(target.shape,np.float32(51)/np.float32(255),np.float32)
        out=scan_mix(r,r.copy(),target)
        self.assertTrue(out['oracle']['zero_residual']);self.assertEqual(out['oracle']['t'],0)
        g=r+.1;out=scan_mix(r,g,target)
        self.assertEqual(out['oracle']['t'],0)
        self.assertGreater(out['grid'][1]['post_pixel_clip_float32']['mse'],0)

    def test_native_anchor_t075_and_psnr_objective(self):
        target=np.full((1,2,2,3),100,np.uint8)
        r=np.full(target.shape,.3,np.float32);g=np.full(target.shape,.8,np.float32)
        out=scan_mix(r,g,target)
        mixed=next(row for row in out['grid'] if row['t']==.75)
        expected=(g.astype(np.float64)*.75+r.astype(np.float64)*.25).clip(0,1).astype(np.float32)
        y=target.astype(np.float32)/np.float32(255)
        mse=float(np.mean((expected.astype(np.float64)-y)**2))
        self.assertAlmostEqual(mixed['post_pixel_clip_float32']['mse'],mse,places=14)
        vals=metric_summary([.01,.0001],[.1,.01])
        self.assertAlmostEqual(vals['psnr'],30)
        self.assertNotAlmostEqual(vals['psnr'],vals['psnr_from_mean_mse'])
        with self.assertRaises(ValueError):assert_quality_match(vals,{**vals,'mse':1},'bad cache')

if __name__=='__main__': unittest.main()
