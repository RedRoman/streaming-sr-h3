"""Stdlib-only condition, failure durability, and save-count checks."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_evidence_checks", ROOT / "h3_evidence_checks.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class Evidence:
    def __init__(self):
        self.data = {"phase": "optimizer_inventory", "existing": [1, 2]}
        self.saved = []

    def save(self):
        self.saved.append(copy.deepcopy(self.data))


def frozen_require():
    """Execute only the pinned old function AST, never its model imports."""
    source = (ROOT / "check_h3_joint_rgb_grad.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != "852df835da5385e0e1f9761df79a6b4f6cf77384bde5a8adf462f5dc118e44e0":
        raise AssertionError("reference helper changed")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "require")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "frozen_require_AST", "exec"), namespace)
    return namespace["require"]


class EvidenceChecksTests(unittest.TestCase):
    def test_success_bulk_has_no_save_or_evidence_mutation(self):
        old = frozen_require()
        old_evidence, new_evidence = Evidence(), Evidence()
        original = copy.deepcopy(new_evidence.data)
        for _ in range(5600):
            self.assertIsNone(old(True, "unused", old_evidence))
            self.assertIsNone(M.require(True, "unused", new_evidence))
        self.assertEqual(len(old_evidence.saved), 5600)
        self.assertEqual(new_evidence.saved, [])
        self.assertEqual(new_evidence.data, original)
        # No report interface is touched on success.
        self.assertIsNone(M.require(True, "unused", object()))

    def test_truthiness_and_rejection_match_frozen_function(self):
        old = frozen_require()
        for value in (False, None, 0, 0.0, "", [], {}, True, 1, "x", [0]):
            with self.subTest(value=value):
                prior, current = Evidence(), Evidence()
                if value:
                    self.assertIsNone(old(value, "same message", prior))
                    self.assertIsNone(M.require(value, "same message", current))
                    self.assertEqual(current.saved, [])
                else:
                    with self.assertRaises(RuntimeError) as first:
                        old(value, "same message", prior)
                    with self.assertRaises(RuntimeError) as second:
                        M.require(value, "same message", current)
                    self.assertEqual(first.exception.args, second.exception.args)
                    self.assertEqual(len(current.saved), 1)
                    expected = [{"message": "same message", "phase": "optimizer_inventory"}]
                    self.assertEqual(current.saved[0]["gate_failures"], expected)
                    self.assertEqual(current.saved[0]["existing"], [1, 2])

    def test_condition_is_evaluated_exactly_once(self):
        class Condition:
            def __init__(self, result):
                self.calls, self.result = 0, result

            def __bool__(self):
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError("condition evaluated twice")
                return self.result

        for result in (True, False):
            condition, evidence = Condition(result), Evidence()
            if result:
                M.require(condition, "condition", evidence)
            else:
                with self.assertRaisesRegex(RuntimeError, "condition"):
                    M.require(condition, "condition", evidence)
            self.assertEqual(condition.calls, 1)

    def test_truth_test_error_saved_and_same_exception_reraised(self):
        original = ValueError("ambiguous truth value")

        class Condition:
            def __bool__(self):
                raise original

        evidence = Evidence()
        with self.assertRaises(ValueError) as raised:
            M.require(Condition(), "tensor condition", evidence)
        self.assertIs(raised.exception, original)
        self.assertEqual(len(evidence.saved), 1)
        self.assertEqual(evidence.saved[0]["gate_failures"][0]["evaluation_error"],
                         {"type": "ValueError", "message": "ambiguous truth value"})

    def test_save_failure_propagates_without_claiming_gate_persisted(self):
        original = OSError("no space")

        class Broken(Evidence):
            def save(self):
                self.attempted = copy.deepcopy(self.data)
                raise original

        evidence = Broken()
        with self.assertRaises(OSError) as raised:
            M.require(False, "rejected", evidence)
        self.assertIs(raised.exception, original)
        self.assertEqual(evidence.attempted["gate_failures"][0]["message"], "rejected")
        self.assertEqual(evidence.saved, [])

    def test_failure_is_on_disk_before_exception_is_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"

            class Disk(Evidence):
                def save(self):
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(self.data, allow_nan=False))
                    temporary.replace(path)

            evidence = Disk()
            with self.assertRaisesRegex(RuntimeError, "finite gate"):
                M.require(False, "finite gate", evidence)
            self.assertEqual(json.loads(path.read_text())["gate_failures"],
                             [{"message": "finite gate", "phase": "optimizer_inventory"}])

    def test_failures_append_and_phase_flush_remains_explicit(self):
        evidence = Evidence()
        for message in ("first", "second"):
            with self.assertRaises(RuntimeError):
                M.require(False, message, evidence)
        self.assertEqual([r["message"] for r in evidence.saved[-1]["gate_failures"]], ["first", "second"])
        evidence.data["phase"] = "reload_complete"
        M.require(True, "pass", evidence)
        self.assertEqual(evidence.saved[-1]["phase"], "optimizer_inventory")
        evidence.save()
        self.assertEqual(evidence.saved[-1]["phase"], "reload_complete")


if __name__ == "__main__":
    unittest.main()
