"""Pure-stdlib context tests; no Torch, Comfy, accelerator, or decoder execution."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

PROBE_PATH = Path(__file__).resolve().parents[1] / "check_vae_decode_grad.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("_vae_grad_probe_context_test", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def fake_modules(*, include_native_function=True):
    names = ("comfy", "comfy.quant_ops", "comfy_kitchen",
             "comfy_kitchen.backends", "comfy_kitchen.backends.ascend",
             "comfy_kitchen.backends.ascend.rope")
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    original, eager, ascend, inplace = (object() for _ in range(4))
    ck = SimpleNamespace(rms_rope_split_half=original, rms_rope_split_half_=inplace)
    modules["comfy.quant_ops"].ck = ck
    if include_native_function:
        modules["comfy_kitchen.backends.ascend.rope"].rms_rope_split_half = ascend
    events = []

    @contextmanager
    def native_training_math():
        previous = ck.rms_rope_split_half
        events.append(("enter", previous))
        ck.rms_rope_split_half = eager
        try:
            yield
        finally:
            events.append(("exit", ck.rms_rope_split_half))
            ck.rms_rope_split_half = previous

    with patch.dict(sys.modules, modules):
        yield SimpleNamespace(ck=ck, original=original, eager=eager, ascend=ascend,
                              inplace=inplace, native=native_training_math, events=events)


class DecoderMathContextTests(unittest.TestCase):
    def setUp(self):
        self.before_modules = set(sys.modules)
        self.probe = load_probe()

    def tearDown(self):
        added = set(sys.modules) - self.before_modules
        self.assertFalse(any(name == "torch" or name.startswith(("torch.", "torch_npu"))
                             for name in added), added)
        self.assertFalse(any(name == "comfy" or name.startswith(("comfy.", "comfy_kitchen"))
                             for name in added), added)

    def test_eager_retains_native_context_choice_without_importing_ascend_function(self):
        with fake_modules(include_native_function=False) as f:
            with self.probe.decoder_math(f.native, "eager"):
                self.assertIs(f.ck.rms_rope_split_half, f.eager)
                self.assertIs(f.ck.rms_rope_split_half_, f.inplace)
            self.assertIs(f.ck.rms_rope_split_half, f.original)
            self.assertEqual(f.events, [("enter", f.original), ("exit", f.eager)])

    def test_ascend_replaces_only_noninplace_and_restores_in_correct_order(self):
        with fake_modules() as f:
            with self.probe.decoder_math(f.native, "ascend_native"):
                self.assertIs(f.ck.rms_rope_split_half, f.ascend)
                self.assertIs(f.ck.rms_rope_split_half_, f.inplace)
            self.assertIs(f.ck.rms_rope_split_half, f.original)
            self.assertIs(f.ck.rms_rope_split_half_, f.inplace)
            # The inner finally must restore eager before native's outer finally.
            self.assertEqual(f.events, [("enter", f.original), ("exit", f.eager)])

    def test_body_exception_propagates_and_restores_both_layers(self):
        class ProbeFailure(Exception):
            pass
        failure = ProbeFailure("synthetic backward failure")
        with fake_modules() as f:
            with self.assertRaises(ProbeFailure) as caught:
                with self.probe.decoder_math(f.native, "ascend_native"):
                    self.assertIs(f.ck.rms_rope_split_half, f.ascend)
                    raise failure
            self.assertIs(caught.exception, failure)
            self.assertIs(f.ck.rms_rope_split_half, f.original)
            self.assertIs(f.ck.rms_rope_split_half_, f.inplace)
            self.assertEqual(f.events[-1], ("exit", f.eager))

    def test_import_failure_restores_outer_context_without_running_body(self):
        with fake_modules(include_native_function=False) as f:
            with self.assertRaises(ImportError):
                with self.probe.decoder_math(f.native, "ascend_native"):
                    self.fail("body must not run after missing native implementation")
            self.assertIs(f.ck.rms_rope_split_half, f.original)
            self.assertIs(f.ck.rms_rope_split_half_, f.inplace)
            self.assertEqual(f.events[-1], ("exit", f.eager))

    def test_nested_context_restores_outer_ascend_choice_before_final_exit(self):
        with fake_modules() as f:
            with self.probe.decoder_math(f.native, "ascend_native"):
                self.assertIs(f.ck.rms_rope_split_half, f.ascend)
                with self.probe.decoder_math(f.native, "eager"):
                    self.assertIs(f.ck.rms_rope_split_half, f.eager)
                self.assertIs(f.ck.rms_rope_split_half, f.ascend)
                self.assertIs(f.ck.rms_rope_split_half_, f.inplace)
            self.assertIs(f.ck.rms_rope_split_half, f.original)


if __name__ == "__main__":
    unittest.main()
