import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import audit_arrays as A

class AuditTests(unittest.TestCase):
    def test_array_exact_then_wrong_file_hash_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.npy';a=np.arange(4,dtype=np.float32)
            np.save(p,a,allow_pickle=False)
            row={'path':'a.npy','file_sha256':A.sha_file(p),'content_sha256':A.array_sha(a),'shape':[4],'dtype':'float32'}
            self.assertTrue(A.Arrays(td,{'a':row}).verify_all()['all_finite'])
            row['file_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'file SHA'):A.Arrays(td,{'a':row}).verify_all()
    def test_nan_rejected_even_with_valid_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.npy';a=np.array([np.nan],np.float32);np.save(p,a,allow_pickle=False)
            row={'path':'a.npy','file_sha256':A.sha_file(p),'content_sha256':A.array_sha(a),'shape':[1],'dtype':'float32'}
            with self.assertRaisesRegex(ValueError,'nonfinite'):A.Arrays(td,{'a':row}).verify_all()
    def test_utility_sign_and_missing_support(self):
        target=np.zeros((1,3,16,16),np.float32);normal=np.full_like(target,.1);removed0=np.full_like(target,.11);removed1=np.full_like(target,.2)
        available=np.array([[1,0]],np.float32);valid=available.astype(bool)
        raw=np.array([[[[float(np.float64(np.float32(.11))-np.float64(np.float32(.1)))]],[[0.]]]],np.float32)
        normalized=np.clip(raw/(.25/255),-4,4)
        result=A.check_utility(normal,removed0,removed1,target,available,raw,normalized,valid)
        self.assertLessEqual(result['raw_max_abs_error'],A.RAW_ATOL)
        with self.assertRaisesRegex(ValueError,'raw utility'):A.check_utility(normal,removed0,removed1,target,available,-raw,-normalized,valid)
        bad=raw.copy();bad[:,1]=1e-12
        with self.assertRaisesRegex(ValueError,'exactly zero'):A.check_utility(normal,removed0,removed1,target,available,bad,normalized,valid)

if __name__=='__main__':unittest.main()
