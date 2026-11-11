"""CPU-only official Euler parity and independent three-step gradient checks.

The copied snippets below were read from the pinned 30794 Comfy source. Only
their AST function definitions are executed, with decorators removed and a tiny
CPU namespace; importing Comfy/H3, loading weights, and NPU calls are forbidden.
Set OPENH3_COMFY_REFERENCE_ROOT to check the actual source files against the
frozen full-file hashes as well. Without it, the frozen snippets are the oracle.
"""
import ast
import hashlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from openh3_differentiable_euler import differentiable_euler, to_d

OFFICIAL_SNAPSHOT = {
    "comfy/k_diffusion/sampling.py": {
        "path": "/cache/zihaohe/omni_openh3_20261111/code/ComfyUI/comfy/k_diffusion/sampling.py",
        "sha256": "cfc4221ed0bc1f84a6f5689d3ec440cad3008c2541ab8f5e2599d1a73fac74ce",
        "functions": {
            "to_d": {
                "first_line": 63,
                "def_line": 63,
                "last_line": 65,
                "sha256": "79b063e3cb9561d840b552ddd71406e9204036cede1eafd025ea0d7b62426a01",
                "source": "def to_d(x, sigma, denoised):\n    \"\"\"Converts a denoiser output to a Karras ODE derivative.\"\"\"\n    return (x - denoised) / utils.append_dims(sigma, x.ndim)\n"
            },
            "sample_euler": {
                "first_line": 189,
                "def_line": 190,
                "last_line": 212,
                "sha256": "d29fcb080c000558a9d94bc545ed64cb71a466ae451d8a1ad83d743e9c5cdff9",
                "source": "@torch.no_grad()\ndef sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):\n    \"\"\"Implements Algorithm 2 (Euler steps) from Karras et al. (2022).\"\"\"\n    extra_args = {} if extra_args is None else extra_args\n    s_in = x.new_ones([x.shape[0]])\n    for i in trange(len(sigmas) - 1, disable=disable):\n        if s_churn > 0:\n            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.\n            sigma_hat = sigmas[i] * (gamma + 1)\n        else:\n            gamma = 0\n            sigma_hat = sigmas[i]\n\n        if gamma > 0:\n            eps = torch.randn_like(x) * s_noise\n            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5\n        denoised = model(x, sigma_hat * s_in, **extra_args)\n        d = to_d(x, sigma_hat, denoised)\n        if callback is not None:\n            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})\n        dt = sigmas[i + 1] - sigma_hat\n        # Euler method\n        x = x + d * dt\n    return x\n"
            }
        }
    },
    "comfy/k_diffusion/utils.py": {
        "path": "/cache/zihaohe/omni_openh3_20261111/code/ComfyUI/comfy/k_diffusion/utils.py",
        "sha256": "fc30630d8a0e4520b2e864c8f84deea3231c9b04d3f1404fec93fc425f629272",
        "functions": {
            "append_dims": {
                "first_line": 21,
                "def_line": 21,
                "last_line": 29,
                "sha256": "816349d93f63ced7a88c4fc3967a6c1ba77bffca03536dea4b4e12604b98f5b5",
                "source": "def append_dims(x, target_dims):\n    \"\"\"Appends dimensions to the end of a tensor until it has target_dims dimensions.\"\"\"\n    dims_to_append = target_dims - x.ndim\n    if dims_to_append < 0:\n        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')\n    expanded = x[(...,) + (None,) * dims_to_append]\n    # MPS will get inf values if it tries to index into the new axes, but detaching fixes this.\n    # https://github.com/pytorch/pytorch/issues/84364\n    return expanded.detach().clone() if expanded.device.type == 'mps' else expanded\n"
            }
        }
    }
}


def reference_metadata():
    path = Path(__file__).resolve().parents[1] / "openh3_differentiable_euler.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "OFFICIAL_REFERENCE" for t in n.targets))
    return ast.literal_eval(node.value)


def official_sources():
    """Verify frozen function bytes; optionally also verify actual Comfy files."""
    frozen = reference_metadata()["files"]
    root = os.environ.get("OPENH3_COMFY_REFERENCE_ROOT")
    result = {}
    for rel, record in OFFICIAL_SNAPSHOT.items():
        assert record["sha256"] == frozen[rel]["sha256"]
        if root:
            data = (Path(root) / rel).read_bytes()
            assert hashlib.sha256(data).hexdigest() == record["sha256"], rel
            lines = data.decode().splitlines()
        for name, function in record["functions"].items():
            source = function["source"]
            assert hashlib.sha256(source.encode()).hexdigest() == function["sha256"]
            assert function["sha256"] == frozen[rel]["functions"][name]["sha256"]
            if root:
                actual = "\n".join(lines[function["first_line"] - 1:function["last_line"]]) + "\n"
                assert actual == source, name
            result[name] = source
    return result


def official_cpu_functions():
    sources = official_sources()
    functions = []
    for name in ("append_dims", "to_d", "sample_euler"):
        node = ast.parse(sources[name]).body[0]
        assert isinstance(node, ast.FunctionDef)
        node.decorator_list = []  # Remove only the official inference decorator.
        functions.append(node)
    namespace = {"torch": torch, "trange": lambda n, disable=None: range(n)}
    compiled = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(compiled, "<pinned-official-euler-functions>", "exec"), namespace)
    namespace["utils"] = SimpleNamespace(append_dims=namespace["append_dims"])
    return namespace


class SourceContractTests(unittest.TestCase):
    def test_frozen_source_hashes_and_optional_actual_files(self):
        self.assertEqual(set(official_sources()), {"sample_euler", "to_d", "append_dims"})

    def test_module_does_not_import_comfy_or_disable_or_detach_graph(self):
        path = Path(__file__).resolve().parents[1] / "openh3_differentiable_euler.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else "")
                self.assertNotIn(name, {"detach", "detach_", "no_grad", "inference_mode"})
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name.startswith(("comfy", "h3", "torch_npu")) for n in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith(("comfy", "h3", "torch_npu")))


@unittest.skipIf(torch is None, "CPU torch is required; this skip is not a numerical pass")
class EulerCPUFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = official_cpu_functions()

    def test_derivative_matches_official_scalar_and_batch_sigma(self):
        state = torch.arange(24, dtype=torch.float64).reshape(2, 1, 12) / 23
        prediction = state.sin()
        for sigma in [torch.tensor(.6, dtype=state.dtype),
                      torch.tensor([.6, .3], dtype=state.dtype)]:
            torch.testing.assert_close(to_d(state, sigma, prediction),
                self.official["to_d"](state, sigma, prediction), rtol=0, atol=0)

    def test_step_final_nfe_and_rng_parity_against_official(self):
        schedules = [[1., .6, .2, 0.], [.9, .6, .2, .1], [1., .6, .6, 0.], [.5, 0.]]
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            for schedule in schedules:
                with self.subTest(dtype=dtype, schedule=schedule):
                    initial = torch.linspace(-.8, .9, 26, dtype=dtype).reshape(2, 1, 13)
                    sigma_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                    sigmas = torch.tensor(schedule, dtype=sigma_dtype)
                    results = []
                    for fn in (self.official["sample_euler"], differentiable_euler):
                        states, calls = [], []
                        def model(state, batch_sigma, *, gain):
                            calls.append((state.clone(), batch_sigma.clone()))
                            return state.tanh() * gain + batch_sigma[:, None, None] * .03
                        before_rng = torch.get_rng_state().clone()
                        output = fn(model, initial, sigmas, extra_args={"gain": .45},
                                    callback=lambda values: states.append(values))
                        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
                        self.assertEqual(len(calls), len(schedule) - 1)
                        self.assertEqual(len(states), len(schedule) - 1)
                        self.assertEqual(calls[0][0].device.type, initial.device.type)
                        torch.testing.assert_close(calls[0][0], initial, rtol=0, atol=0)
                        results.append((output, states, calls))
                    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
                    for left, right in zip(results[0][1], results[1][1]):
                        self.assertEqual(left.keys(), right.keys())
                        self.assertEqual(left["i"], right["i"])
                        for key in ("x", "sigma", "sigma_hat", "denoised"):
                            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                    for left, right in zip(results[0][2], results[1][2]):
                        for a, b in zip(left, right):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_three_step_adapter_and_initial_state_gradients_match_closed_form(self):
        # x0_hat_i = theta_i*x_i, so x_final = x_initial*product(a_i),
        # a_i = 1 + ((sigma_next-sigma)/sigma)*(1-theta_i).
        initial = torch.tensor([[[.2, -.4, .7, .1]]], dtype=torch.float64, requires_grad=True)
        params = torch.tensor([.2, .4, .6], dtype=torch.float64, requires_grad=True)
        schedule = [1., .6, .2, 0.]
        sigmas = torch.tensor(schedule, dtype=torch.float64)
        calls = []
        def model(state, sigma):
            index = len(calls)
            calls.append(state)
            return params[index] * state
        output = differentiable_euler(model, initial, sigmas)
        loss = .5 * output.square().sum()
        loss.backward()
        self.assertEqual(len(calls), 3)
        rates = [(schedule[i + 1] - schedule[i]) / schedule[i] for i in range(3)]
        values = [.2, .4, .6]
        factors = [1 + rates[i] * (1 - values[i]) for i in range(3)]
        product = math.prod(factors)
        square_sum = .2 ** 2 + .4 ** 2 + .7 ** 2 + .1 ** 2
        expected_param = [square_sum * product * (-rates[i]) *
                          math.prod(factors[j] for j in range(3) if j != i) for i in range(3)]
        torch.testing.assert_close(output, initial * product, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(params.grad, torch.tensor(expected_param, dtype=params.dtype),
                                   rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(initial.grad, initial * product ** 2, rtol=1e-12, atol=1e-14)
        self.assertTrue(bool(torch.isfinite(params.grad).all()))
        self.assertTrue(bool((params.grad.abs() > 1e-8).all()))
        self.assertTrue(bool((initial.grad.abs() > 1e-8).all()))

    def test_complete_joint_state_and_callback_keep_graph(self):
        # Synthetic coupling only: this does not assert actual H3 audio semantics.
        initial = torch.arange(1, 9, dtype=torch.float64).reshape(1, 1, 8).requires_grad_()
        theta = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        traces = []
        def model(state, sigma):
            video, audio = state[..., :5], state[..., 5:]
            return torch.cat([theta * video + .1 * audio.mean(dim=-1, keepdim=True),
                              theta * audio + .05 * video.mean(dim=-1, keepdim=True)], dim=-1)
        output = differentiable_euler(model, initial,
            torch.tensor([1., .6, .2, 0.], dtype=torch.float64), callback=traces.append)
        output[..., :5].sum().backward()
        self.assertEqual(tuple(output.shape), tuple(initial.shape))
        self.assertIs(traces[0]["x"], initial)  # No hidden rescale/re-noise/copy.
        self.assertTrue(traces[1]["x"].requires_grad)
        self.assertTrue(traces[2]["denoised"].requires_grad)
        self.assertGreater(float(initial.grad[..., 5:].abs().sum()), 0)
        self.assertGreater(float(theta.grad.abs()), 0)

    def test_invalid_schedules_and_churn_fail_before_model_calls(self):
        state = torch.ones(1, 1, 4)
        for values in [[1.], [1., 0., 0.], [1., -.1], [1., 1.1], [float("nan"), 0.]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                differentiable_euler(lambda *args: self.fail("model called"),
                                     state, torch.tensor(values))
        with self.assertRaisesRegex(ValueError, "zero-churn"):
            differentiable_euler(lambda *args: self.fail("model called"),
                                 state, torch.tensor([1., 0.]), s_churn=.1)

    def test_incomplete_joint_prediction_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete joint"):
            differentiable_euler(lambda state, sigma: state[..., :2],
                                 torch.ones(1, 1, 4), torch.tensor([1., 0.]))


if __name__ == "__main__":
    unittest.main()
