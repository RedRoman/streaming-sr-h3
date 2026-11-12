import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit_arrays.py'));A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)

class ArrayAuditTests(unittest.TestCase):
    def test_zero_mse_null_clipping_and_actual_error(self):
        z=np.zeros((3,96,96),np.float32);p=np.full_like(z,-.125);m=A.metric(p,z,z)
        self.assertIsNone(m['clipped_psnr']);self.assertTrue(m['clipped_exact_match']);self.assertEqual(m['unclamped_mae'],.125);self.assertEqual(m['clipped_mse'],0.);self.assertEqual(m['clip_fraction'],1.)
    def fixture(self):
        shape=(1,3,96,96);z=np.zeros(shape,np.float32)
        arrays={'current':z.copy(),'target':z.copy(),'all_history':np.full(shape,.125,np.float32),'current_only':np.full(shape,.25,np.float32),'remove_0':np.full(shape,.1875,np.float32),'remove_1':np.full(shape,.0625,np.float32)}
        row={'profile_id':'test','target_crop_sha256_float32':A.asha(z[0]),'availability':[1,1]}
        modes={m:A.metric(arrays['current' if m=='input' else m][0],z[0],z[0]) for m in A.MODES}
        row.update(metrics=modes,raw_utility=[np.full((6,6),.0625).tolist(),np.full((6,6),-.0625).tolist()],all_history_mae=.125,current_only_mae=.25,all_history_mae_harm_vs_input=.125,current_only_mae_harm_vs_input=.25)
        return arrays,row
    def test_signed_utility_and_wrong_sign_rejected(self):
        arrays,row=self.fixture();A.recompute_row(arrays,0,row,{})
        row['raw_utility'][0][0][0]*=-1
        with self.assertRaisesRegex(ValueError,'numerical mismatch'):A.recompute_row(arrays,0,row,{})
    def test_missing_history_cache_exact_and_zero(self):
        arrays,row=self.fixture();row['availability'][0]=0;row['raw_utility'][0]=np.zeros((6,6)).tolist()
        with self.assertRaisesRegex(ValueError,'exactly reuse'):A.recompute_row(arrays,0,row,{})
        arrays['remove_0']=arrays['all_history'].copy();row['metrics']['remove_0']=row['metrics']['all_history'];A.recompute_row(arrays,0,row,{})
        row['raw_utility'][0][0][0]=1e-15
        with self.assertRaisesRegex(ValueError,'exactly zero'):A.recompute_row(arrays,0,row,{})
    def test_cross_reduction_tolerance_is_predeclared_and_gate_exact(self):
        rng=np.random.default_rng(7);p=rng.standard_normal((3,96,96)).astype(np.float32);y=rng.random((3,96,96),dtype=np.float32)
        expected=np.abs(p.astype(np.float64)-y.astype(np.float64)).mean(0).reshape(6,16,6,16).mean((1,3))
        self.assertLess(float(np.max(np.abs(A.tiles(p,y)-expected))),A.ATOL)
        with self.assertRaises(ValueError):A.compare({'gate':True},{'gate':False},{})
        with self.assertRaises(ValueError):A.compare(float('nan'),0.,{})
    def test_actual_npz_hash_and_finite_checks(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d).resolve();(out/'arrays').mkdir();p=out/'arrays/batch_0000.npz';arrays={n:np.zeros((16,3,96,96),np.float32) for n in A.NAMES}
            def save():
                np.savez(p,**arrays)
                return {'path':str(p),'bytes':p.stat().st_size,'sha256':A.sha(p),'arrays':{n:{'dtype':'float32','shape':[16,3,96,96],'sha256':A.asha(a)} for n,a in arrays.items()}}
            e=save();self.assertEqual(len(A.load_batch(e,out)),6);bad=copy.deepcopy(e);bad['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'bytes/SHA'):A.load_batch(bad,out)
            arrays['all_history'][0,0,0,0]=np.nan;e=save()
            with self.assertRaisesRegex(ValueError,'nonfinite'):A.load_batch(e,out)
    def test_source_balanced_gate_and_control_separation(self):
        protocols={'dev_source_ids':[str(i) for i in range(6)],'profile_ids':['a','b','c','identity','near_clean'],'rule':A.RULE};rows=[]
        for sid in protocols['dev_source_ids']:
            for pid in protocols['profile_ids']:
                for t in range(96):
                    av=[int(t>=2),int(t>=1)];u=np.zeros((2,6,6),np.float64)
                    for slot in range(2):
                        if av[slot]:u[slot,:,::2]=.001;u[slot,:,1::2]=-.001
                    gain=.001 if pid in ('a','b','c') else -.1
                    rows.append({'source_id':sid,'profile_id':pid,'frame_index':t,'availability':av,'raw_utility':u.tolist(),'all_history_mae':.2,'current_only_mae':.2+gain})
        a=A.assessment(rows,protocols);self.assertTrue(a['headroom_rule_passed']);self.assertEqual(a['valid_tiles'],204120);self.assertAlmostEqual(a['source_balanced_degraded_mae_gain'],.001);self.assertAlmostEqual(a['positive_tile_fraction'],.5);self.assertEqual(a['sources_with_positive_degraded_gain'],6)
        self.assertLess(a['per_source_profile'][-1]['history_mae_gain'],0.)

if __name__=='__main__':unittest.main(verbosity=2)
