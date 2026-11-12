import hashlib,pathlib,tempfile,unittest
from reconstruct_local import reconstruct
H=lambda b:hashlib.sha256(b).hexdigest()
class Reconstruction(unittest.TestCase):
 def test_exact_and_preserve(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);a=p/'prefix';b=p/'suffix';a.write_bytes(b'abc');b.write_bytes(b'def');o=p/'whole'
   r=reconstruct((a,3,H(b'abc')),[(b,3,3,H(b'def'))],o,6,H(b'abcdef'))
   self.assertEqual(o.read_bytes(),b'abcdef');self.assertEqual(a.read_bytes(),b'abc');self.assertEqual(r['bytes'],6)
   with self.assertRaises(AssertionError):reconstruct((a,3,H(b'abc')),[(b,3,3,H(b'def'))],o,6,H(b'abcdef'))
 def test_wrong_prefix_or_gap_rejected(self):
  for badhash,offset in [(True,3),(False,4)]:
   with tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d);a=p/'prefix';b=p/'suffix';a.write_bytes(b'abc');b.write_bytes(b'def');o=p/'whole'
    with self.assertRaises(AssertionError):reconstruct((a,3,H(b'bad' if badhash else b'abc')),[(b,offset,3,H(b'def'))],o,6,H(b'abcdef'))
    self.assertFalse(o.exists());self.assertEqual(a.read_bytes(),b'abc')
if __name__=='__main__':unittest.main()
