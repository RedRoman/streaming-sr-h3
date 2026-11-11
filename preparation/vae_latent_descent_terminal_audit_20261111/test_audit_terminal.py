"""Minimal array/update/state integrity fixtures; no Torch or model imports."""
import copy
import json
import tarfile
from pathlib import Path
import tempfile
import unittest

import numpy as np
import audit_terminal as audit


def cast(a, b):
    x, y = a.astype(np.float16), b.astype(np.float16)
    delta = y.astype(np.float64) - x.astype(np.float64)
    return {"before_sha256": audit.array_sha(x), "after_sha256": audit.array_sha(y), "dtype": "float16",
            "changed_coordinates": int(np.count_nonzero(delta)), "elements": delta.size,
            "changed_fraction": float(np.count_nonzero(delta) / delta.size),
            "actual_cast_step_rms": float(np.sqrt(np.mean(delta ** 2))),
            "actual_cast_max_abs_change": float(np.max(np.abs(delta)))}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = np.arange(24, dtype=np.float32).reshape(6, 4) / np.float32(100) + np.float32(1)
        self.protocol = {"input_shape": [6, 4], "RGB_shape": [22,480,832,3], "input_sha256": audit.array_sha(self.source)}
        self.r = {"schema": "openh3.vae_latent_descent_probe.v1", "complete": True, "protocol_complete": True,
                  "phase": "completed_local_descent", "error": None, "contract": self.protocol,
                  "forward_attempts": 8, "forwards_completed": 8, "backward_attempts": 3, "backwards_completed": 3,
                  "arrays": {}, "evaluations": {}, "gradients": {}, "steps": [], "checkpoint": {},
                  "gates": {k: True for k in ("protocol", "frozen", "runtime", "budget", "restoration")},
                  "frozen_before": {"sha256": "a"*64, "parameter_count": 559},
                  "frozen_after": {"sha256": "a"*64, "parameter_count": 559}, "frozen_versions_unchanged": True,
                  "runtime_before": {"jit_compile_false": True, "vae_dtype": "torch.float16"},
                  "runtime_after": {"jit_compile_false": True, "vae_dtype": "torch.float16"}}
        source = self.source.copy(); self.save("latent_0", source)
        self.eval("initial_reference", source, 1.)
        for i in range(1,4):
            label = f"step{i}_gradient"
            raw = np.full(source.shape, 1024., dtype=np.float32)
            gradient = np.ones_like(source)
            self.save(label+"_raw", raw); self.save(label+"_unscaled", gradient)
            self.r["gradients"][label] = {"raw": audit.statistics(raw), "unscaled": audit.statistics(gradient)}
            before, after = 1.-(i-1)*.1, 1.-i*.1
            self.eval(label, source, before, True)
            following = np.ascontiguousarray(source - np.float32(.001))
            self.save(f"latent_{i}", following); self.eval(f"step{i}_next", following, after)
            actual = source.astype(np.float64)-following.astype(np.float64)
            row = {"step":i, "loss_before":before, "loss_after":after, "actual_loss_decrease":before-after,
                   "accepted":True, "gradient_rms_float64":1., "rho":.001,
                   "actual_float32_step_rms":float(np.sqrt(np.mean(actual**2))), "gradient_dot_direction":24.,
                   "predicted_first_order_decrease":float(np.sum(actual)), "float16_cast_change":cast(source,following)}
            if i == 1:
                plus = np.ascontiguousarray(source + np.float32(.001))
                self.save("first_positive_latent",plus); self.eval("first_positive",plus,1.1)
                central = (1.1-after)/.002
                row["directional_check"] = {"loss_positive":1.1, "central_difference":central,"gradient_dot_direction":24.,
                    "central_minus_gradient_dot":central-24., "relative_magnitude_error_descriptive_only":abs(central-24.)/24.,
                    "positive_loss_minus_initial":1.1-before, "positive_loss_increased_observation_only":True,
                    "positive_float16_cast_change":cast(source,plus), "central_positive_gate":True}
            self.r["steps"].append(row)
            self.r["checkpoint"][label] = {"enabled":True,"active":False,"restored":True,"block_count":36,"wrapped":36,
                "use_reentrant":False,"preserve_rng_state":True,"checkpoint_forward_calls":108,"no_grad_forward_calls":0,
                "recompute_calls":108,"output_gradient_guards":108,"backward_guard_calls":108,"recomputed_block_indices":list(range(36))}
            source = following

    def save(self, name, array):
        path = self.root/(name+".npy"); np.save(path,array,allow_pickle=False)
        self.r["arrays"][name] = {"path":path.name,"file_sha256":audit.sha_file(path),"array_sha256":audit.array_sha(array),
                                  "shape":list(array.shape),"dtype":"float32","bytes":path.stat().st_size}

    def eval(self, label, source, loss, grad=False):
        row = {"input_sha256":audit.array_sha(source),"actual_cast_sha256":audit.array_sha(source.astype(np.float16)),
               "actual_cast_dtype":"float16","output_sha256":"b"*64,"output_shape":[22,480,832,3],
               "output_dtype":"torch.float32","finite":True,"loss":loss,"target_sha256":"c"*64}
        if grad: row["parity"]={"allclose":True,"cast_exact":True,"requires_grad":True,"bitwise_equal":True,"max_abs_difference":0.}
        self.r["evaluations"][label]=row

    def check(self, successful=True):
        return audit.validate_report(self.r,self.root,self.protocol,successful)

    def test_small_archive_excludes_arrays_and_manifest_matches_bytes(self):
        prep,run,own,out = [self.root / x for x in (audit.PREP,audit.RUN,"own","out")]
        for p in (prep/"code/tests",run/"eval",run/"logs",own,out):p.mkdir(parents=True,exist_ok=True)
        for p in (prep/"queue.json",run/"queue_status.json",run/"eval/report.json",out/"audit.json"):
            p.write_text("{}")
        (prep/"code/probe.py").write_text("# frozen")
        (run/"eval/raw.npy").write_bytes(b"excluded")
        (own/"audit_terminal.py").write_text("# standalone")
        result=audit.pack(self.root,out,own)
        with tarfile.open(result["path"]) as t:
            self.assertFalse(any(x.endswith(".npy") for x in t.getnames()))
            manifest=json.load(t.extractfile("MANIFEST.json"))
            for name,row in manifest["files"].items():
                data=t.extractfile(name).read()
                self.assertEqual(audit.hashlib.sha256(data).hexdigest(),row["sha256"])
        self.assertEqual(audit.sha_file(result["path"]),result["sha256"])

    def test_fixed_updates_and_bounded_success(self):
        facts=self.check(); self.assertEqual(facts["exact_normalized_updates_verified"],[1,2,3])
        self.assertEqual(len(facts["verified_array_manifest"]),11)

    def test_tampered_hash_and_resealed_wrong_update_rejected(self):
        path=self.root/"latent_1.npy"; saved=path.read_bytes();path.write_bytes(saved[:-1]+b"X")
        with self.assertRaisesRegex(ValueError,"SHA"):self.check()
        path.write_bytes(saved)
        array=np.load(path); array.flat[0]+=np.float32(.01);self.save("latent_1",array)
        with self.assertRaisesRegex(ValueError,"update"):self.check()

    def test_malformed_success_counts_or_checkpoint_rejected(self):
        self.r["forwards_completed"]=7
        with self.assertRaisesRegex(ValueError,"budget"):self.check()
        self.r["forwards_completed"]=8;self.r["checkpoint"]["step1_gradient"]["restored"]=False
        with self.assertRaisesRegex(ValueError,"checkpoint"):self.check()

    def test_first_step_descent_failure_preserved_and_partial_arrays_checked(self):
        self.r.update(complete=False,protocol_complete=False,error="descent gate failed",phase="stopped_evidence_retained",
                      forward_attempts=3,forwards_completed=3,backward_attempts=1,backwards_completed=1)
        self.r["arrays"]={k:v for k,v in self.r["arrays"].items() if k in ("latent_0","latent_1","step1_gradient_raw","step1_gradient_unscaled")}
        self.r["evaluations"]={k:v for k,v in self.r["evaluations"].items() if k in ("initial_reference","step1_gradient","step1_next")}
        self.r["evaluations"]["step1_next"]["loss"]=1.1
        self.r["gradients"]={"step1_gradient":self.r["gradients"]["step1_gradient"]}
        self.r["checkpoint"]={"step1_gradient":self.r["checkpoint"]["step1_gradient"]}
        row=self.r["steps"][0];row.pop("directional_check");row.update(accepted=False,loss_after=1.1,actual_loss_decrease=1.-1.1)
        self.r["steps"]=[row]
        self.assertEqual(self.check(False)["exact_normalized_updates_verified"],[1])
        self.assertFalse(self.r["complete"])
        with self.assertRaisesRegex(ValueError,"failed report"):self.check(True)


if __name__=="__main__":unittest.main()
