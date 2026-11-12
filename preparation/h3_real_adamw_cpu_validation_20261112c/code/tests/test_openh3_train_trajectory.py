"""CPU-only trajectory plumbing; optional AST execution of pinned Comfy sources.

Set OPENH3_COMFY_REFERENCE_ROOT to the reviewed Comfy checkout to run the actual
CFGGuider.inner_sample / KSAMPLER.sample / prepare_noise function bodies against
tiny fake model/conditioning objects. No Comfy import, checkpoint loading or NPU
execution occurs. These tests do not establish real-H3 numerical parity.
"""
import ast
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import openh3_train_trajectory as trajectory


def reference_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def fixture():
    """All expensive model operations are fake, while sampler bodies are real AST."""
    root = Path(os.environ["OPENH3_COMFY_REFERENCE_ROOT"])
    for relative, expected in trajectory.COMFY_SHA256.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise AssertionError("source changed: " + relative)
    runtime_state = {"math": False}

    @contextmanager
    def math_context():
        previous = runtime_state["math"]
        runtime_state["math"] = True
        try:
            yield
        finally:
            runtime_state["math"] = previous

    class Nested:
        is_nested = True
        def __init__(self, values): self.values = values
        def unbind(self): return self.values

    ns = {"torch": torch, "comfy": SimpleNamespace(nested_tensor=SimpleNamespace(NestedTensor=Nested))}
    reference_function(root / "comfy/sample.py", "prepare_noise_inner", ns)
    prepare_noise = reference_function(root / "comfy/sample.py", "prepare_noise", ns)
    import math
    util_ns = {"torch": torch, "math": math}
    pack = reference_function(root / "comfy/utils.py", "pack_latents", util_ns)
    unpack = reference_function(root / "comfy/utils.py", "unpack_latents", util_ns)

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4]))
            self.lora_b = torch.nn.Parameter(torch.tensor([0.01, 0.02, 0.03]))
        def forward(self, x, index, transformer_options=None):
            if not runtime_state["math"]:
                raise AssertionError("recompute escaped the training math context")
            return x * self.lora_a[index] * self.lora_b[index]

    class TinyDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(0.65), requires_grad=False)
            self.blocks = torch.nn.ModuleList([TinyBlock()])
            self._openh3_pretrained_record = {"fake": "CPU test only"}
            self.eval()
        def forward(self, x, index):
            # Coupling the full joint mean makes both audio evolution and its
            # gradient relevant to video; resetting audio each step fails parity.
            return self.base * x + x.mean(-1, keepdim=True) * .07 + self.blocks[0](x, index)

    class Flow:
        audio_scale = 4.0
        def noise_scaling(self, sigma, noise, latent, max_denoise):
            return sigma * noise + (1 - sigma) * latent
        def inverse_noise_scaling(self, sigma, value): return value / (1 - sigma)

    class Base:
        def __init__(self):
            self.diffusion_model = TinyDiT()
            self.model_sampling = Flow()
            self.model_lowvram = False
            self.latent_format = SimpleNamespace(scale_factor=1.0)
            self.current_patcher = None
            self.input_scaled = None
        def process_latent_in(self, value):
            video, audio = unpack(value, self.latent_shapes)
            result = pack([video, audio * 4])[0]
            self.input_scaled = result.clone()
            return result
        def process_latent_out(self, value):
            video, audio = unpack(value, self.latent_shapes)
            return pack([video, audio / 4])[0]

    base = Base()

    class Patcher:
        def __init__(self): self.model, self.model_options = base, {"transformer_options": {}}
        def is_dynamic(self): return False
        def pre_run(self): self.model.current_patcher = self

    patcher = Patcher()

    class Executor:
        @staticmethod
        def new_class_executor(fn, _instance, _wrappers): return SimpleNamespace(execute=fn)

    def process_conds(model, noise, conds, device, latent, denoise_mask, seed, latent_shapes):
        if denoise_mask is not None or len(conds["positive"]) != 1 or conds["negative"]:
            raise AssertionError("unexpected conditioning")
        runtime_state["guide"] = conds["positive"][0]["extra"]["minimax_keyframes"][0]["latent"]
        runtime_state["seed"] = seed
        return conds

    class Inpaint:
        def __init__(self, model, sigmas): self.model = model
        def __call__(self, x, sigma, denoise_mask=None, **kwargs):
            if denoise_mask is not None: raise AssertionError("mask not supported")
            return self.model(x, sigma, **kwargs)

    comfy = SimpleNamespace(
        model_patcher=SimpleNamespace(create_model_options_clone=copy.deepcopy),
        patcher_extension=SimpleNamespace(
            WrapperExecutor=Executor, get_all_wrappers=lambda *a, **k: [],
            WrappersMP=SimpleNamespace(SAMPLER_SAMPLE="sampler")))
    sampler_ns = {"torch": torch, "comfy": comfy, "process_conds": process_conds,
                  "KSamplerX0Inpaint": Inpaint, "detail": lambda *a: None}
    inner = reference_function(root / "comfy/samplers.py", "inner_sample", sampler_ns, "CFGGuider")
    sampler_sample = reference_function(root / "comfy/samplers.py", "sample", sampler_ns, "KSAMPLER")

    class Guider:
        inner_sample = inner
        def __init__(self, p):
            self.model_patcher, self.model_options = p, p.model_options
            self.index = 0
        def set_cfg(self, cfg): self.cfg = cfg
        def set_conds(self, positive, negative):
            self.original_conds = {"positive": [{"context": p[0], "extra": p[1]} for p in positive],
                                   "negative": negative}
        def __call__(self, x, sigma, model_options, seed):
            self.index += 1
            return self.inner_model.diffusion_model(x, self.index - 1) + runtime_state["guide"].mean() * .03

    class Sampler:
        sample = sampler_sample
        def __init__(self, fn):
            self.sampler_function, self.extra_options, self.inpaint_options = fn, {}, {}
        def max_denoise(self, model, sigmas): return False

    backend = SimpleNamespace(NestedTensor=Nested, prepare_noise=prepare_noise, pack=pack,
        unpack=unpack, Guider=Guider, Sampler=Sampler, math_context=math_context,
        sources={"test": "actual pinned sampler AST, fake denoiser/conditioning"})
    h3 = SimpleNamespace(patcher=patcher, dev=torch.device("cpu"), pretrained_manifest="test only",
        sigma_schedule=torch.tensor(trajectory.NATIVE_SIGMAS),
        context={"prompt_embeds": torch.zeros(2, 3), "text_token_tags": torch.zeros(2, dtype=torch.int64)})
    return h3, backend, runtime_state


@unittest.skipUnless(torch is not None and os.environ.get("OPENH3_COMFY_REFERENCE_ROOT"),
                     "requires Torch CPU and the pinned Comfy source (no Comfy import)")
class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.h3, self.backend, self.state = fixture()
        self.guide = torch.full((1, 24, 2, 2, 2), .2)
        self.warm = torch.full_like(self.guide, .4)

    def session(self, **kw):
        return trajectory.ResidentH3Trajectory(self.h3, _backend=self.backend, **kw)

    def test_cpu_noise_order_audio_initialization_and_shapes(self):
        packed, noise, shapes, report = trajectory._joint_inputs(self.backend, self.guide, self.warm, 101, "randn")
        generator = torch.Generator().manual_seed(101)
        expected_video = torch.randn(self.warm.shape, generator=generator)
        expected_audio = torch.randn((1, 32, 2, 8), generator=generator)
        self.assertTrue(torch.equal(noise, self.backend.pack([expected_video, expected_audio])[0]))
        expected_initial_audio = torch.randn((1, 32, 2, 8), generator=torch.Generator().manual_seed(101))
        self.assertTrue(torch.equal(self.backend.unpack(packed, shapes)[1], expected_initial_audio))
        self.assertEqual(report["rgb_frames"], 5)
        guide22 = torch.zeros(1, 24, 7, 2, 2)
        _, _, shapes22, report22 = trajectory._joint_inputs(self.backend, guide22, guide22, 101, "zeros")
        self.assertEqual(shapes22[1][-1], 37)
        self.assertEqual(report22["rgb_frames"], 22)

    def test_official_euler_forward_parity_and_all_three_joint_gradients(self):
        events = []
        with self.session() as session:
            result = session.sample(self.guide, self.warm, audio_init="randn", callback=lambda x: events.append(x))
            result["video"].square().mean().backward()
            self.assertEqual(len(events), 3)
            self.assertEqual(result["contract"]["actual_forward_NFE"], 3)
            for parameter in self.h3.patcher.model.diffusion_model.blocks[0].parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertEqual(int(torch.count_nonzero(parameter.grad)), 3)
            shapes = self.h3.patcher.model.latent_shapes
            scaled_audio = self.backend.unpack(self.h3.patcher.model.input_scaled, shapes)[1]
            original_audio = torch.randn(scaled_audio.shape, generator=torch.Generator().manual_seed(101))
            self.assertTrue(torch.equal(scaled_audio, original_audio * 4))
        root = Path(os.environ["OPENH3_COMFY_REFERENCE_ROOT"])
        ns = {"torch": torch, "trange": lambda n, **kw: range(n),
              "to_d": lambda x, s, y: (x - y) / s}
        official = reference_function(root / "comfy/k_diffusion/sampling.py", "sample_euler", ns)
        def official_adapter(model, state, sigmas, extra_args=None, callback=None):
            return official(model, state, sigmas, extra_args=extra_args, callback=callback)
        with torch.no_grad(), patch.object(trajectory, "differentiable_euler", official_adapter), self.session() as session:
            reference = session.sample(self.guide, self.warm, audio_init="randn")
        self.assertTrue(torch.equal(result["video"], reference["video"]))
        self.assertTrue(torch.equal(result["audio"], reference["audio"]))
        self.assertIsNone(self.h3.patcher.model.current_patcher)
        self.assertFalse(self.state["math"])

    def test_checkpoint_forward_and_gradient_parity_and_restoration(self):
        with self.session() as session:
            plain = session.sample(self.guide, self.warm)
            plain["video"].square().mean().backward()
        params = list(self.h3.patcher.model.diffusion_model.blocks[0].parameters())
        expected_grads = [p.grad.clone() for p in params]
        for p in params: p.grad = None
        block = self.h3.patcher.model.diffusion_model.blocks[0]
        self.assertNotIn("forward", block.__dict__)
        with self.session(checkpoint_blocks=True) as session:
            checkpointed = session.sample(self.guide, self.warm)
            checkpointed["video"].square().mean().backward()
        self.assertTrue(torch.equal(plain["video"], checkpointed["video"]))
        for p, expected in zip(params, expected_grads): self.assertTrue(torch.equal(p.grad, expected))
        self.assertNotIn("forward", block.__dict__)

    def test_no_late_backward_no_hidden_hooks_no_schedule_substitution(self):
        with self.session() as session:
            result = session.sample(self.guide, self.warm)
        with self.assertRaisesRegex(RuntimeError, "inside"):
            result["video"].sum().backward()
        self.h3.patcher.model_options["sampler_post_cfg_function"] = [lambda x: x]
        with self.assertRaisesRegex(ValueError, "custom"):
            with self.session(): pass
        self.h3.patcher.model_options = {"transformer_options": {}}
        self.h3.sigma_schedule = torch.tensor([1., .8, .6, 0.])
        with self.assertRaisesRegex(ValueError, "schedule"): self.session()

    def test_zero_adapter_and_unmerged_reload_are_separate_from_merge(self):
        block = self.h3.patcher.model.diffusion_model.blocks[0]
        with torch.no_grad(): block.lora_b.zero_()
        with torch.no_grad(), self.session() as session:
            zero = session.sample(self.guide, self.warm)
        self.assertTrue(zero["contract"]["adapter_B_all_zero_at_session_entry"])
        with torch.no_grad(): block.lora_b.fill_(.02)
        state = copy.deepcopy(block.state_dict())
        with torch.no_grad(), self.session() as session: nonzero = session.sample(self.guide, self.warm)
        with torch.no_grad(): block.lora_b.zero_()
        block.load_state_dict(state, strict=True)
        with torch.no_grad(), self.session() as session: reloaded = session.sample(self.guide, self.warm)
        self.assertTrue(torch.equal(nonzero["video"], reloaded["video"]))
        self.assertFalse(torch.equal(zero["video"], nonzero["video"]))
        self.assertIn("not_assumed", reloaded["contract"]["acceptance"]["nonzero_BF16_merge_parity"])

    def test_bad_input_and_exception_restore_context(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            with self.session() as session: session.sample(self.guide.requires_grad_(), self.warm)
        self.assertFalse(self.state["math"])
        self.assertIsNone(self.h3.patcher.model.current_patcher)
        with self.assertRaisesRegex(TypeError, "float32"):
            trajectory._check_latents(self.guide.detach().half(), self.warm)

    def test_decoder_recompute_context_does_not_change_dit_math(self):
        runtime = {"rope": "dit", "native_calls": 0}

        @contextmanager
        def decoder_context():
            previous = runtime["rope"]
            runtime["rope"] = "vae_native"
            try:
                yield
            finally:
                runtime["rope"] = previous

        class DecoderBlock(torch.nn.Module):
            def forward(self, value):
                self.assert_native()
                runtime["native_calls"] += 1
                return value.sin() * .1 + value
            def assert_native(self):
                if runtime["rope"] != "vae_native": raise AssertionError("wrong VAE recompute math")

        stage = torch.nn.Module()
        stage.decoder = torch.nn.Module()
        stage.decoder.transformer_blocks = torch.nn.ModuleList([DecoderBlock() for _ in range(36)])
        stage.register_parameter("frozen", torch.nn.Parameter(torch.tensor(1.), requires_grad=False))
        stage.eval()
        def decode(video):
            value = video
            for block in stage.decoder.transformer_blocks: value = block(value)
            return value
        self.h3.vae = SimpleNamespace(first_stage_model=stage, device="cpu", offload_device="cpu",
            disable_offload=True, vae_dtype=torch.float32, decode=decode)
        original = self.h3.patcher.model.diffusion_model.blocks[0].forward
        def dit_forward(*args, **kwargs):
            if runtime["rope"] != "dit": raise AssertionError("VAE math leaked into DiT recompute")
            return original(*args, **kwargs)
        self.h3.patcher.model.diffusion_model.blocks[0].forward = dit_forward
        with self.session(checkpoint_blocks=True, checkpoint_decoder=True,
                          decoder_context=decoder_context) as session:
            result = session.sample(self.guide, self.warm)
            pixels = session.decode(result["video"])
            self.assertEqual(runtime["rope"], "dit")
            pixels.square().mean().backward()
            self.assertEqual(runtime["rope"], "dit")
        self.assertGreater(runtime["native_calls"], 36)
        self.assertTrue(all("forward" not in b.__dict__ for b in stage.decoder.transformer_blocks))

    def test_midpatch_failure_and_overlap_restore_all_wrappers(self):
        good = torch.nn.Identity()
        class FailsOnPatch(torch.nn.Identity):
            def __setattr__(self, name, value):
                if name == "forward": raise RuntimeError("synthetic assignment failure")
                super().__setattr__(name, value)
        bad = FailsOnPatch()
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with trajectory._checkpoint_modules([good, bad], True, lambda: True): pass
        self.assertNotIn("forward", good.__dict__)
        self.assertNotIn("forward", bad.__dict__)
        with trajectory._checkpoint_modules([good], True, lambda: True):
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                with trajectory._checkpoint_modules([good], True, lambda: True): pass
        self.assertNotIn("forward", good.__dict__)

    def test_output_guard_rejects_late_backward_without_recompute(self):
        block = torch.nn.Identity()
        source = torch.ones(3, requires_grad=True)
        with trajectory._checkpoint_modules([block], True, lambda: True):
            output = block(source)
        with self.assertRaisesRegex(RuntimeError, "original checkpoint"):
            output.sum().backward()
        session = self.session()
        with session:
            old = session.sample(self.guide, self.warm)
        with session:
            with self.assertRaisesRegex(RuntimeError, "inside"):
                old["video"].sum().backward()


@unittest.skipUnless(torch is not None, "requires Torch CPU only")
class CheckpointSnapshotTests(unittest.TestCase):
    def test_shared_options_are_frozen_at_each_block_call(self):
        class OptionBlock(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float64))
                self.seen_indices = []

            def forward(self, value, transformer_options=None):
                index = transformer_options["block_index"]
                self.seen_indices.append(index)
                # The option influences a saved activation, so a stale index
                # produces incorrect gradients even if forward parity holds.
                result = (value * self.weight * (index + 1)).sin()
                transformer_options["block_index"] = -1000
                return result

        blocks = [OptionBlock(.7), OptionBlock(.9)]
        source = torch.tensor([.2, -.4, .6], dtype=torch.float64)

        def run(checkpointed):
            value = source.clone().requires_grad_()
            leaf = value
            shared_options = {"block_index": 0}
            with trajectory._checkpoint_modules(blocks, checkpointed, lambda: True):
                for index, block in enumerate(blocks):
                    shared_options["block_index"] = index
                    value = block(value, transformer_options=shared_options)
                # Models may reuse the dict for later blocks or later NFE.
                shared_options["block_index"] = 49
                value.square().sum().backward()
            return value.detach(), leaf.grad.clone(), [b.weight.grad.clone() for b in blocks]

        expected = run(False)
        for block in blocks:
            block.weight.grad = None
            block.seen_indices.clear()
        actual = run(True)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))
        for found, wanted in zip(actual[2], expected[2]):
            self.assertTrue(torch.equal(found, wanted))
        self.assertEqual([b.seen_indices for b in blocks], [[0, 0], [1, 1]])
        self.assertTrue(all("forward" not in b.__dict__ for b in blocks))


if __name__ == "__main__":
    unittest.main()
