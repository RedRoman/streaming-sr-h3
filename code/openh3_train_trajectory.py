"""Resident, differentiable *complete* Open-H3 three-step trajectory candidate.

This module does not load weights, inject/merge adapters, train, or admit an NPU
job. The caller supplies a verified H3 instance with frozen pretrained DiT and
optional independent FP32 LoRA already resident on its device. Keep the session
open through backward. No Comfy outer-sampler cleanup/offload is run.

The official CFGGuider.inner_sample and KSAMPLER retain conditioning, packed
video/audio initialization, audio carry transforms, CFG1 and x0 semantics. Only
the no-grad Euler function is replaced. H3.sample's no-grad wrapper is bypassed.
There is no single-sigma surrogate, re-noising, GT input, DC, or RGB clipping here.

CPU tests establish plumbing/math only. Real H3 zero-adapter parity, nonzero
unmerged save/reload parity and joint memory/backward are separate pending gates.
Existing inference merges FP32 W+BA into BF16: that is a different arithmetic
path and must be measured separately, never assumed equal from zero-LoRA tests.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
from pathlib import Path
import threading
from types import SimpleNamespace

import torch

from openh3_differentiable_euler import differentiable_euler


NATIVE_SIGMAS = (0.909375, 6 / 7, 2 / 3, 0.0)
_CHECKPOINT_BLOCK_IDS = set()
_CHECKPOINT_LOCK = threading.RLock()
COMFY_SHA256 = {
    "comfy/sample.py": "d274ed5b6abf83707b52b87d25d1716f072d5444954cf23ac0ed31a3b803ee1c",
    "comfy/samplers.py": "f2c264ca9d394612f828e3ffe167c856a278a10e1711b269f0ba65dccb66393f",
    "comfy/sampler_helpers.py": "53c82235bfea0c4edf3cd5219f080c44c45ac1c9a6c3171f2aae151e18b56b91",
    "comfy/model_base.py": "1212947ec7c143f23fa411ef280e5f9c257b38379ccfc9298688879e1fa0777d",
    "comfy/model_sampling.py": "8afdc665272589a567792bb652389df7d4a83f69574c2b7e5b3d533d55592990",
    "comfy/model_patcher.py": "7edc88d9eae6a88637625d257dc73684cfc012a29446bf88e70d22e3a6be3994",
    "comfy/utils.py": "9320cf5c624e96acc1d9ec92ff120ed6387a2142707f3194cb21dd0d11ac2500",
    "comfy/conds.py": "72058e9a22c972a9c875819e59d432d30d367fd2f7092ee6c6c45e5a60c959b0",
    "comfy/ldm/minimax/model.py": "15f0395cf0dbe6e5c13fd859ff359eacc7761b4e15f60aa0038853985d0e599b",
    "comfy/k_diffusion/sampling.py": "cfc4221ed0bc1f84a6f5689d3ec440cad3008c2541ab8f5e2599d1a73fac74ce",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_runtime_sources(root):
    """Fail closed before importing the actual Comfy execution dependency set."""
    actual = {}
    for relative, expected in COMFY_SHA256.items():
        path = Path(root) / relative
        found = _sha(path)
        if found != expected:
            raise ValueError("Unreviewed Comfy source: " + str(path))
        actual[str(path)] = found
    for name in ("openh3_train_trajectory.py", "openh3_differentiable_euler.py",
                 "h3_comfy.py", "lora.py", "openh3_runtime.py", "check_vae_decode_grad.py"):
        path = Path(__file__).resolve().parent / name
        actual[str(path)] = _sha(path)
    return actual


def _runtime():
    # h3_comfy is imported only for an explicitly constructed resident session,
    # not by CPU math/test imports of this module.
    from h3_comfy import COMFY, native_training_math
    sources = verify_runtime_sources(COMFY)
    import comfy.sample
    import comfy.samplers
    import comfy.nested_tensor
    import comfy.utils
    return SimpleNamespace(
        NestedTensor=comfy.nested_tensor.NestedTensor,
        prepare_noise=comfy.sample.prepare_noise,
        pack=comfy.utils.pack_latents, unpack=comfy.utils.unpack_latents,
        Guider=comfy.samplers.CFGGuider, Sampler=comfy.samplers.KSAMPLER,
        math_context=native_training_math, sources=sources,
    )


def _has_content(value):
    if isinstance(value, dict):
        return any(_has_content(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_content(v) for v in value)
    return value is not None and value is not False


def _tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _check_latents(guide, warm_start):
    for name, value in (("guide", guide), ("raw warm start", warm_start)):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise TypeError(name + " must be an explicit float32 latent tensor")
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError(name + " must be a frozen CPU latent cache")
        if value.ndim != 5 or value.shape[:2] != (1, 24):
            raise ValueError(name + " must have shape [1,24,T,H,W]")
        if not torch.isfinite(value).all():
            raise ValueError(name + " contains nonfinite values")
    if guide.shape != warm_start.shape:
        raise ValueError("guide and raw warm start geometry must match")
    _, _, latent_t, latent_h, latent_w = guide.shape
    if latent_t < 2 or (latent_t - 2) % 5 or min(latent_h, latent_w) < 2:
        raise ValueError("unsupported native H3 temporal/spatial geometry")
    if latent_h % 2 or latent_w % 2:
        raise ValueError("RGB geometry must be divisible by 32; no implicit padding")
    return (latent_t - 2) // 5 * 17 + 5


def _joint_inputs(runtime, guide, warm_start, seed, audio_init):
    frames = _check_latents(guide, warm_start)
    audio_t = round(frames / 24 * 40)
    if audio_init == "zeros":
        audio = torch.zeros(1, 32, 2, audio_t, dtype=torch.float32)
    elif audio_init == "randn":
        audio = torch.randn(1, 32, 2, audio_t,
                            generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    else:
        raise ValueError("audio_init must be zeros or randn")
    latent = runtime.NestedTensor((warm_start, audio))
    # Intentionally use the actual Comfy CPU/global-generator behavior: video
    # first, then audio from the SAME generator. A device RNG is not equivalent.
    noise = runtime.prepare_noise(latent, seed)
    packed, shapes = runtime.pack(list(latent.unbind()))
    packed_noise, noise_shapes = runtime.pack(list(noise.unbind()))
    if shapes != noise_shapes:
        raise ValueError("noise and warm-start joint layouts differ")
    return packed, packed_noise, shapes, {
        "rgb_frames": frames, "audio_t": audio_t, "audio_init": audio_init,
        "guide_latent_sha256_float32": _tensor_sha(guide),
        "raw_warm_start_latent_sha256_float32": _tensor_sha(warm_start),
        "packed_cpu_noise_sha256_float32": _tensor_sha(packed_noise),
        "packed_initial_latent_sha256_float32": _tensor_sha(packed),
        "joint_shapes": [list(shape) for shape in shapes],
    }


@contextmanager
def _checkpoint_modules(blocks, enabled, active, math_context=None):
    """Scoped checkpointing with a fresh math context for *each* recomputation.

    In joint backward the outer session uses DiT math. A decoder context_fn must
    therefore restore native VAE math only while its own block is recomputed.
    Never leave native VAE RoPE globally active over a DiT recomputation.
    """
    if not enabled:
        yield
        return
    from torch.utils.checkpoint import checkpoint
    blocks = tuple(blocks)
    ids = {id(block) for block in blocks}
    if len(ids) != len(blocks) or not all(isinstance(b, torch.nn.Module) for b in blocks):
        raise ValueError("checkpoint blocks must be distinct torch modules")
    with _CHECKPOINT_LOCK:
        if ids & _CHECKPOINT_BLOCK_IDS:
            raise RuntimeError("nested or overlapping checkpoint modules")
        _CHECKPOINT_BLOCK_IDS.update(ids)
    saved = []
    state = {"active": True}

    def guard(gradient):
        if not state["active"] or not active():
            raise RuntimeError("backward must remain inside the original checkpoint/session scope")
        return gradient

    def guard_outputs(value):
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                value.register_hook(guard)
        elif isinstance(value, dict):
            for item in value.values(): guard_outputs(item)
        elif isinstance(value, (list, tuple)):
            for item in value: guard_outputs(item)
        return value

    try:
        for block in blocks:
            original = block.forward

            def wrapped(*inputs, _original=original, **kwargs):
                if not state["active"] or not active():
                    raise RuntimeError("checkpoint session is no longer active")
                if not torch.is_grad_enabled():
                    return _original(*inputs, **kwargs)

                def recompute(*args):
                    if not state["active"] or not active():
                        raise RuntimeError("backward must remain inside the trajectory session")
                    copied = dict(kwargs)
                    if "transformer_options" in copied:
                        copied["transformer_options"] = dict(copied["transformer_options"])
                    return _original(*args, **copied)

                factory = nullcontext if math_context is None else math_context
                result = checkpoint(recompute, *inputs, use_reentrant=False,
                                    context_fn=lambda: (factory(), factory()))
                return guard_outputs(result)

            saved.append((block, original, "forward" in block.__dict__))
            block.forward = wrapped
        yield
    finally:
        state["active"] = False
        errors = []
        for block, original, had_instance_override in reversed(saved):
            try:
                if had_instance_override:
                    block.forward = original
                elif "forward" in block.__dict__:
                    del block.forward
            except BaseException as error:
                errors.append(error)
        with _CHECKPOINT_LOCK:
            _CHECKPOINT_BLOCK_IDS.difference_update(ids)
        if errors:
            raise RuntimeError("failed to restore checkpoint block forward attributes") from errors[0]


class ResidentH3Trajectory:
    """Use ``with ... as trajectory: loss(trajectory.sample(...)).backward()``.

    Caller must move all DiT parameters/buffers (including the FP32 adapter) to
    h3.dev before entering; this class never loads, moves, merges or offloads
    weights. It intentionally rejects hooks, dynamic/low-VRAM patchers and
    custom model options. Concurrent use of Comfy global math state is unsupported.
    ``_backend`` is only for isolated CPU contract tests, never an execution CLI.
    """
    def __init__(self, h3, *, checkpoint_blocks=False, checkpoint_decoder=False,
                 decoder_context=None, _backend=None):
        self.h3 = h3
        self.backend = _runtime() if _backend is None else _backend
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.checkpoint_decoder = bool(checkpoint_decoder)
        self.decoder_context = decoder_context
        self.vae_versions = None
        self.active = False
        self.generation = 0
        self.patcher = h3.patcher
        self.model = self.patcher.model
        self.dit = self.model.diffusion_model
        self.device = torch.device(h3.dev)
        self.stack = None
        schedule = torch.as_tensor(h3.sigma_schedule, dtype=torch.float32, device="cpu")
        if not torch.equal(schedule, torch.tensor(NATIVE_SIGMAS, dtype=torch.float32)):
            raise ValueError("only the declared native three-step sigma schedule is supported")
        self.sigmas = schedule.to(self.device)
        if not h3.pretrained_manifest or not getattr(self.dit, "_openh3_pretrained_record", None):
            raise ValueError("verified pretrained Open-H3 provenance is required")
        self.contract = {
            "schema": "openh3.resident_full_joint_trajectory.v1",
            "sigma_schedule_float32": schedule.tolist(), "steps": 3, "cfg": 1.0,
            "mode": "base", "refs": [], "inner_dc": False, "s_churn": 0.0,
            "target_used_as_model_input": False,
            "noise": "official Comfy CPU prepare_noise: video then audio, shared seeded generator",
            "audio_semantics": "official process_latent_in/out + MiniMaxH3.forward carried audio velocity",
            "model_output": "official BaseModel.apply_model packed joint denoised x0",
            "adapter_arithmetic": "independent FP32 LoRA branch; no weight merge or STE",
            "checkpoint_blocks": self.checkpoint_blocks,
            "checkpoint_decoder_blocks": self.checkpoint_decoder,
            "decoder_checkpoint_math": "per-forward and per-recompute context_fn; never global over DiT backward",
            "implementation_sha256": dict(self.backend.sources),
            "pretrained": self.dit._openh3_pretrained_record,
            "acceptance": {"real_zero_adapter_deployment_parity": "not_established_by_interface",
                "nonzero_unmerged_reload_parity": "separate_required_gate",
                "nonzero_BF16_merge_parity": "not_assumed; separately_measure_rounding",
                "joint_full_resolution_memory_backward": "unverified",
                "uncheckpointed_full_resolution_VAE_backward": "blocked_by_observed_forward_OOM"},
        }

    def _validate_resident(self):
        p = self.patcher
        if p.is_dynamic() or getattr(self.model, "model_lowvram", False):
            raise ValueError("dynamic/low-VRAM patchers are not admitted for a retained graph")
        if _has_content(p.model_options):
            raise ValueError("custom model/transformer options are outside this parity contract")
        for name in ("patches", "object_patches", "weight_wrapper_patches", "callbacks", "wrappers",
                     "hook_patches", "cached_hook_patches", "current_hooks", "forced_hooks",
                     "injections", "additional_models"):
            if _has_content(getattr(p, name, None)):
                raise ValueError("unsupported patcher state: " + name)
        if self.dit.training:
            raise ValueError("DiT must remain in eval mode, including during adapter training")
        if float(self.model.model_sampling.audio_scale) != 4.0:
            raise ValueError("expected the reviewed video/audio shift ratio 12/3")
        if float(self.model.latent_format.scale_factor) != 1.0:
            raise ValueError("unexpected video latent scale")
        self.base_versions = []
        adapters = []
        for name, value in self.dit.named_parameters():
            if value.device != self.device:
                raise ValueError("DiT parameter is not resident: " + name)
            if name.endswith((".lora_a", ".lora_b")):
                if value.dtype != torch.float32 or not value.requires_grad:
                    raise ValueError("adapter must be live FP32: " + name)
                adapters.append((name, value))
            else:
                if value.requires_grad or value.grad is not None:
                    raise ValueError("pretrained base is not frozen: " + name)
                self.base_versions.append((name, value, value._version))
        for name, value in self.dit.named_buffers():
            if value.device != self.device:
                raise ValueError("DiT buffer is not resident: " + name)
        if not self.base_versions:
            raise ValueError("empty frozen DiT")
        self.contract["adapter_parameter_tensors"] = len(adapters)
        self.contract["adapter_B_all_zero_at_session_entry"] = all(
            not bool(torch.count_nonzero(v)) for n, v in adapters if n.endswith(".lora_b"))

    def __enter__(self):
        if self.active:
            raise RuntimeError("trajectory session is already active")
        self._validate_resident()
        self.saved_patcher = getattr(self.model, "current_patcher", None)
        self.had_shapes = hasattr(self.model, "latent_shapes")
        self.saved_shapes = getattr(self.model, "latent_shapes", None)
        self.active = True
        self.generation += 1
        self.stack = ExitStack()
        try:
            self.stack.enter_context(self.backend.math_context())
            self.stack.enter_context(_checkpoint_modules(self.dit.blocks, self.checkpoint_blocks, lambda: self.active))
            if self.checkpoint_decoder:
                stage = self._validate_vae()
                blocks = tuple(stage.decoder.transformer_blocks)
                if len(blocks) != 36 or len({id(b) for b in blocks}) != 36:
                    raise ValueError("expected 36 distinct native VAE decoder transformer blocks")
                self.stack.enter_context(_checkpoint_modules(blocks, True, lambda: self.active,
                                                              self._decoder_context))
            self.patcher.pre_run()  # callbacks rejected; sets current_patcher only
        except BaseException:
            self.__exit__(*__import__("sys").exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.stack is not None:
                self.stack.close()
            if exc_type is None:
                self.assert_frozen()
        finally:
            self.model.current_patcher = self.saved_patcher
            if self.had_shapes:
                self.model.latent_shapes = self.saved_shapes
            elif hasattr(self.model, "latent_shapes"):
                del self.model.latent_shapes
            self.active = False
        return False

    def assert_frozen(self):
        for name, value, version in self.base_versions:
            if value._version != version or value.requires_grad or value.grad is not None:
                raise RuntimeError("pretrained base changed: " + name)
        for name, value, version in self.vae_versions or []:
            if value._version != version or value.requires_grad or value.grad is not None:
                raise RuntimeError("frozen VAE changed: " + name)
        return {"frozen_base_parameters": len(self.base_versions), "versions_unchanged": True}

    def _guard_backward(self, gradient, generation):
        if not self.active or generation != self.generation:
            raise RuntimeError("backward must remain inside the trajectory session")
        return gradient

    def sample(self, guide, raw_warm_start, *, seed=101, audio_init="zeros", callback=None):
        """Return graph-bearing video/audio on device; never accepts target pixels.

        callback receives Euler's pre-update state/x0 and must not mutate it.
        Both conditioning guide and draft are frozen normalized CPU latents;
        raw_warm_start remains the raw-input VAE encoding, not a Rphi guide.
        """
        if not self.active:
            raise RuntimeError("use the trajectory session context")
        packed, noise, shapes, inputs = _joint_inputs(
            self.backend, guide, raw_warm_start, seed, audio_init)
        ctx = self.h3.context
        extra = {"minimax_token_tags": ctx["text_token_tags"],
                 "minimax_keyframes": [{"resolved_frame_index": 0, "latent": guide}]}
        positive = [[ctx["prompt_embeds"][None], extra]]
        guider = self.backend.Guider(self.patcher)
        guider.set_cfg(1.0)
        guider.set_conds(positive, [])
        guider.inner_model = self.model
        # The corresponding copy is normally made by CFGGuider.sample before
        # outer_sample. Our accepted conditions have no hooks/control/areas.
        guider.conds = {k: [dict(c) for c in v] for k, v in guider.original_conds.items()}
        calls = []

        def euler(model, state, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
            if kwargs:
                raise ValueError("unexpected sampler options")

            def observer(event):
                calls.append(float(event["sigma"]))
                if callback is not None:
                    callback(event)
                if user_callback is not None:
                    user_callback(event)

            return differentiable_euler(model, state, sigmas, extra_args=extra_args, callback=observer)

        user_callback = callback
        sampler = self.backend.Sampler(euler)
        # All conversions below match CFGGuider.outer_sample. The actual official
        # inner_sample handles condition processing, audio scaling and x0 CFG.
        result = guider.inner_sample(
            noise.to(device=self.device, dtype=torch.float32),
            packed.to(device=self.device, dtype=torch.float32), self.device, sampler,
            self.sigmas, None, None, True, seed, latent_shapes=shapes)
        if len(calls) != 3:
            raise RuntimeError("trajectory did not evaluate exactly three joint denoisers")
        video, audio = self.backend.unpack(result, shapes)
        for value in (video, audio):
            if not torch.isfinite(value).all():
                raise FloatingPointError("nonfinite trajectory result")
            if value.requires_grad:
                value.register_hook(lambda g, generation=self.generation: self._guard_backward(g, generation))
        return {"video": video, "audio": audio, "contract": {
            **self.contract, **inputs, "seed": seed, "actual_forward_NFE": len(calls),
            "backward_checkpoint_recomputations_included_in_NFE": False,
            "output_requires_grad": video.requires_grad, "output_device": str(video.device)}}

    def _decoder_context(self):
        if self.decoder_context is not None:
            return self.decoder_context()
        from check_vae_decode_grad import decoder_math
        return decoder_math(self.backend.math_context, "ascend_native")

    def _validate_vae(self):
        vae = self.h3.vae
        stage = vae.first_stage_model
        if (stage.training or not vae.disable_offload or torch.device(vae.device) != self.device
                or torch.device(vae.offload_device) != self.device):
            raise ValueError("frozen eval VAE must be explicitly resident with offload disabled")
        for name, value in stage.named_parameters():
            if value.device != self.device or value.requires_grad or value.grad is not None:
                raise ValueError("VAE parameter is not resident/frozen: " + name)
        for name, value in stage.named_buffers():
            if value.device != self.device:
                raise ValueError("VAE buffer is not resident: " + name)
        if self.vae_versions is None:
            self.vae_versions = [(n, p, p._version) for n, p in stage.named_parameters()]
        return stage

    def decode(self, video):
        """Optional direct frozen-VAE decode, retaining internal deployed clipping.

        The default reuses the native-RoPE probe. Full 22x480x832 WITHOUT decoder
        activation checkpointing has already OOMed in isolation. Optional
        checkpoint_decoder=True wraps the 36 blocks for the whole session with
        per-recomputation native decoder math; its real NPU parity/memory is a
        separate pending gate. There is no silent fallback. Caller must keep
        both resident DiT and VAE through backward, which is not yet validated.
        """
        if not self.active:
            raise RuntimeError("use the trajectory session context")
        vae = self.h3.vae
        self._validate_vae()
        with self._decoder_context():
            pixels = vae.decode(video.to(dtype=vae.vae_dtype)).float()
        pixels = pixels[0] if pixels.ndim == 5 else pixels
        if pixels.requires_grad:
            pixels.register_hook(lambda g, generation=self.generation: self._guard_backward(g, generation))
        return pixels
