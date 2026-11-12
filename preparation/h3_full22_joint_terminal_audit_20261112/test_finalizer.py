"""Bounded terminal wait and actual small-RGB archive fixtures; no real queue."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
import finalize_once
import small_archive
from audit_arrays import file_sha


class FinalizerTests(unittest.TestCase):
    def test_wait_is_bounded_and_failed_terminal_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"queue.json";p.write_text('{"state":"running"}')
            elapsed=[0.]
            record=finalize_once.wait_terminal(p,lambda *args:None,max_wait=65,poll=30,
                clock=lambda:elapsed[0],sleep=lambda s:elapsed.__setitem__(0,elapsed[0]+s))
            self.assertFalse(record["terminal"]);self.assertEqual(elapsed[0],65)
            self.assertEqual(json.loads(p.read_text())["state"],"running")
            p.write_text('{"state":"failed"}')
            record=finalize_once.wait_terminal(p,lambda *args:None,sleep=lambda s:self.fail("unexpected sleep"))
            self.assertTrue(record["terminal"]);self.assertEqual(record["record"]["state"],"failed")
            p.write_text('{"state":"not_launched"}')
            record=finalize_once.wait_terminal(p,lambda *args:None,sleep=lambda s:self.fail("must not wait after admission refusal"))
            self.assertTrue(record["terminal"]);self.assertEqual(record["record"]["state"],"not_launched")

    def test_full_latents_pack_but_rgb_and_gradients_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();prep,run,own,out=[root/x for x in ("prep","run","own","out")]
            for p in (prep/"code/tests",run/"eval",run/"logs",own,out):p.mkdir(parents=True)
            for p in (run/"queue_status.json",out/"finalize_receipt.json",out/"terminal_audit.json",prep/"execution_contract.json"):
                p.write_text("{}")
            arrays={}
            for label in ("reference_video","candidate_video"):
                p=run/"eval"/(label+".npy");np.save(p,np.zeros((1,24,7,30,52),np.float32))
                arrays[label]={"path":str(p),"file_sha256":file_sha(p)}
            (run/"eval/adapter_gradient.npy").write_bytes(b"must remain remote")
            result=small_archive.package(root,prep,run,own,out,True,{"archive_arrays":arrays})
            with tarfile.open(result["path"]) as t:
                manifest=json.load(t.extractfile("MANIFEST.json"))
                self.assertFalse(any("adapter_gradient" in x for x in t.getnames()))
                self.assertEqual(len([x for x in t.getnames() if x.endswith(".npy")]),2)
                for name,row in manifest["files"].items():
                    data=t.extractfile(name).read();self.assertEqual(small_archive.sha(data),row["sha256"])
            self.assertLess(result["uncompressed_source_bytes"],16<<20)
            self.assertEqual(file_sha(result["path"]),result["sha256"])
            with self.assertRaisesRegex(ValueError,"unapproved"):
                small_archive.package(root,prep,run,own,out,True,{"archive_arrays":{"reference_rgb":next(iter(arrays.values()))}})
            with self.assertRaises(FileExistsError):
                small_archive.package(root,prep,run,own,out,True,{"archive_arrays":arrays})


if __name__=="__main__":unittest.main()
