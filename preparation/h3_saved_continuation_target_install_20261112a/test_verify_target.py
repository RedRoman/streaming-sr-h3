import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verify_target as V

class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'data';self.path.write_bytes(b'fixed')
        self.row={'path':str(self.path),'roles':['fixture'],'expected_bytes':5,'expected_sha256':hashlib.sha256(b'fixed').hexdigest()}
    def test_stat_does_not_read_but_full_detects_same_size_tamper(self):
        self.path.write_bytes(b'wrong')
        with patch.object(V,'sha_file',side_effect=AssertionError('no read in stat-only')):
            self.assertEqual(V.check_file(self.row,False)['status'],'stat_only_SHA_pending')
        self.assertEqual(V.check_file(self.row,True)['status'],'failed')
        self.path.write_bytes(b'fixed');self.assertEqual(V.check_file(self.row,True)['status'],'SHA_verified')
    def test_missing_and_symlink_fail_closed(self):
        self.path.unlink();self.assertEqual(V.check_file(self.row,True)['status'],'failed')
        other=self.path.with_name('other');other.write_bytes(b'fixed');os.symlink(other,self.path)
        self.assertEqual(V.check_file(self.row,True)['status'],'failed')
    def test_runtime_metadata_is_exact_and_no_framework_import(self):
        expected={'python':V.sys.version,'torch':'t','torch_npu':'n'}
        with patch.object(V.importlib.metadata,'version',side_effect=lambda x:{'torch':'t','torch_npu':'n'}[x]):
            r=V.runtime_metadata(expected);self.assertTrue(r['exact_match']);self.assertEqual(r['forbidden_modules_imported'],[])
            self.assertFalse(V.runtime_metadata(dict(expected,torch='different'))['exact_match'])

if __name__=='__main__':unittest.main(verbosity=2)
