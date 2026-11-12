import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from download_refiner import exact_copy, publish_verified


class DownloadTests(unittest.TestCase):
    def test_exact_body_limit_and_early_EOF(self):
        out = io.BytesIO(); count, digest = exact_copy(io.BytesIO(b"abc"), out, 3, lambda n: None)
        self.assertEqual((count, digest), (3, hashlib.sha256(b"abc").hexdigest()))
        for body in (b"ab", b"abcd"):
            out = io.BytesIO()
            with self.assertRaises(ValueError): exact_copy(io.BytesIO(body), out, 3, lambda n: None)
            self.assertLessEqual(len(out.getvalue()), 3)

    def test_verified_publication_never_overwrites_and_wrong_hash_preserves_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            partial, ready = Path(directory) / "x.partial", Path(directory) / "x.ready"
            partial.write_bytes(b"abc"); digest = hashlib.sha256(b"abc").hexdigest()
            with self.assertRaises(ValueError): publish_verified(partial, ready, 3, "0" * 64)
            self.assertTrue(partial.exists()); self.assertFalse(ready.exists())
            ready.write_bytes(b"old")
            with self.assertRaises(FileExistsError): publish_verified(partial, ready, 3, digest)
            self.assertEqual(ready.read_bytes(), b"old"); ready.unlink()
            publish_verified(partial, ready, 3, digest)
            self.assertFalse(partial.exists()); self.assertEqual(ready.read_bytes(), b"abc")


if __name__ == "__main__": unittest.main()
