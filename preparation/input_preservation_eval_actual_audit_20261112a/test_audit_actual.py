import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import audit_actual as A


class ActualAuditTests(unittest.TestCase):
    def test_signed_zero_mask_and_clip_bytes(self):
        raw=np.full((3,96,96),np.float32(-0.0));k1=np.ones_like(raw)
        score=np.full((1,6,6),np.float32(-0.0));mask=np.zeros_like(score,dtype=np.uint8)
        p=A.compose(raw,k1,score,mask)
        self.assertEqual(A.asha(p),A.asha(raw));self.assertEqual(A.asha(A.clipped(p)),A.asha(raw))
        score[0,0,0]=1;mask[0,0,0]=1;p=A.compose(raw,k1,score,mask)
        self.assertTrue((p[:,:16,:16]==1).all());self.assertEqual(A.asha(p[:,:,16:]),A.asha(raw[:,:,16:]))
        mask[0,0,0]=0
        with self.assertRaisesRegex(ValueError,'threshold'):A.compose(raw,k1,score,mask)
        score[0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'score'):A.compose(raw,k1,score,mask)

    def test_rgb_math_clipping_and_known_seam(self):
        raw=np.zeros((3,96,96),np.float32);p=raw.copy();p[:,:,:16]=2
        m=A.metrics(p,raw,raw,np.ones((1,6,6),np.uint8),.125)
        self.assertAlmostEqual(m['clipped_mae'],1/6);self.assertAlmostEqual(m['clipped_mse'],1/6)
        self.assertAlmostEqual(m['unclamped_mae'],1/3);self.assertAlmostEqual(m['unclamped_mse'],2/3)
        self.assertAlmostEqual(m['residual_boundary_jump'],.1);self.assertEqual(m['residual_nonboundary_jump'],0)
        self.assertEqual(m['lpips'],.125);self.assertAlmostEqual(m['clipped_pixel_fraction'],1/6)
        exact=A.metrics(raw,raw,raw,np.zeros((1,6,6),np.uint8),0.)
        self.assertIsNone(exact['clipped_psnr']);self.assertTrue(exact['clipped_exact_match'])

    def test_npz_actual_file_array_and_dtype_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'arrays').mkdir();p=root/'arrays/x.npz';a=np.zeros((16,4),np.float64)
            np.savez(p,lpips=a)
            e={'path':str(p),'sha256':A.fsha(p),'bytes':p.stat().st_size,
               'arrays':{'lpips':{'shape':[16,4],'dtype':'float64','sha256':A.asha(a)}}}
            self.assertEqual(A.asha(A.load_npz(e,root,{'lpips':((16,4),'float64')})['lpips']),A.asha(a))
            bad=copy.deepcopy(e);bad['arrays']['lpips']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'bytes'):A.load_npz(bad,root,{'lpips':((16,4),'float64')})
            with self.assertRaisesRegex(ValueError,'type'):A.load_npz(e,root,{'lpips':((16,4),'float32')})
            p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'SHA'):A.load_npz(e,root,{'lpips':((16,4),'float64')})

    def test_tolerance_does_not_weaken_bytes_null_or_bool(self):
        err={};A.compare({'m':1.+5e-13,'clipped_psnr':20.+5e-11},{'m':1.,'clipped_psnr':20.},err)
        self.assertGreater(err['metric'],0)
        for a,b in [(1,True),('a','b'),(0.,None),(1.+2e-12,1.)]:
            with self.assertRaises(ValueError):A.compare(a,b,{})

    def test_complete_stride1_and_phase_support(self):
        pairs=[];details=[]
        for s in range(6):
            for p in range(5):
                sid='s'+str(s);profile='p'+str(p)
                for t in range(1,96):pairs.append({'source_id':sid,'profile_id':profile,'from_frame':t-1,'to_frame':t,
                    'declared_phase_transition':t==32,'arms':{a:{k:0. for k in A.PAIR_KEYS} for a in A.ARMS}})
                for support,n in [('all',95),('declared_transition',1),('same_phase',94)]:
                    details.append({'source_id':sid,'profile_id':profile,'support':support,'pairs':n,'arms':{a:{k:0. for k in A.PAIR_KEYS} for a in A.ARMS}})
        temporal={'per_source_profile':details,'pairs':2850,'declared_phase_transition_pairs':30}
        A.temporal_check(pairs,temporal,{})
        pairs[5]['to_frame']=7
        with self.assertRaisesRegex(ValueError,'stride1'):A.temporal_check(pairs,temporal,{})


if __name__=='__main__':unittest.main()
