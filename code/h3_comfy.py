"""Frozen MiniMax-H3 (ComfyUI implementation) on Ascend NPU: guide encode, 49-step sampling, sampling-inner DC, decode.

Ported from the colleague's H100 scripts (experiments/h3_backbone/sample_inner_dc*.py in
RedRoman/streaming-sr-h3). Only device calls change: CUDA -> NPU. Same weights, ComfyUI
helpers, sigma grid, seed, guide preprocessing, correction indices and pixel projection.
"""
import ast
import hashlib
import json
import math
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

ROOT = Path('/cache/zihaohe/streaming_sr0')
COMFY = ROOT / 'code/ComfyUI'
ASSETS = ROOT / 'assets/comfy_h3'
DIT_PATHS = {'fl2va': ASSETS / 'minimax_h3_fl2va_pruned_bf16.safetensors', 'ref2va': ASSETS / 'minimax_h3_ref2va_pruned_bf16.safetensors'}
DIT_PATH = DIT_PATHS['fl2va']
VAE_PATH = ASSETS / 'minimax_h3_video_vae_fp16.safetensors'
CONTEXT_PATH = ASSETS / 'h3_qwen_context_neutral_restore.pt'
SHA = {'dit': 'a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4',
       'ref2va': '37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe',
       'vae': '7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522',
       'context': 'fb03abcc1eb569d2fcd360fe512f2ae0d707298f4a7ad8b6a35b609e1d97d5d2'}
SIZE = {'dit': 40225724176, 'vae': 5207808496}
POINTS = [28, 45, 48]
SEED = 101
GRID = torch.linspace(1, 0, 50)
SIGMAS = 12 * GRID / (1 + 11 * GRID)

sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(ROOT / 'code'))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(64 << 20):
            h.update(b)
    return h.hexdigest()


def sync():
    torch.npu.synchronize()


def patch_module_to():
    """torch_npu replaces nn.Module.to with a version that calls self.cast_weight(device); ComfyUI ops define an
    unrelated cast_weight(input), so the call crashes. With ND matmul format and jit_compile=False, torch_npu's
    cast_weight performs no format cast at all, so dropping the call is behaviour-preserving (asserted here)."""
    assert torch.npu.get_mm_bmm_format_nd() and torch.npu.is_jit_compile_false()

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None and not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError(f'nn.Module.to only accepts floating point or complex dtypes, but got desired dtype={dtype}')

        def convert(t):
            if convert_to_format is not None and t.dim() == 4:
                return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking, memory_format=convert_to_format)
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
        return self._apply(convert)
    torch.nn.Module.to = to
    return {'torch_npu_module_to_cast_weight_dropped': True, 'mm_bmm_format_nd': True, 'jit_compile_false': True}


def set_precision():
    """Disable HF32 (the NPU analogue of TF32); return the initial flags for the record."""
    initial = {'matmul_allow_hf32': bool(torch.npu.matmul.allow_hf32), 'conv_allow_hf32': bool(torch.npu.conv.allow_hf32)}
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    return initial


@contextmanager
def native_training_math(functional_residuals=True):
    """Same scoped eager math the colleague used for sampling on H100 (training_math.py)."""
    import comfy.model_management as mm
    import comfy.quant_ops
    import comfy.ldm.minimax.model as implementation
    from comfy_kitchen.backends.eager.rope import rms_rope_split_half
    saved = (mm.in_training, comfy.quant_ops.ck.rms_rope_split_half, implementation._mod_scale_shift, implementation._mod_gate)

    def scale_shift(h, shift, scale, segments):
        return torch.cat([h[a:b] * (1.0 + implementation._mod_row(scale, row, h.dtype)) + implementation._mod_row(shift, row, h.dtype)
                          for a, b, row in segments], dim=0)

    def gate(x, gates, other, segments):
        return torch.cat([torch.addcmul(x[a:b], other[a:b], implementation._mod_row(gates, row, x.dtype)) for a, b, row in segments], dim=0)
    try:
        mm.in_training = True
        comfy.quant_ops.ck.rms_rope_split_half = rms_rope_split_half
        if functional_residuals:
            implementation._mod_scale_shift = scale_shift
            implementation._mod_gate = gate
        yield
    finally:
        mm.in_training, comfy.quant_ops.ck.rms_rope_split_half, implementation._mod_scale_shift, implementation._mod_gate = saved


def official_resize():
    """The node file's _resize (lanczos via comfy.utils.common_upscale), extracted without importing the node module."""
    import comfy.utils
    node_src = COMFY / 'comfy_extras/nodes_minimax_h3.py'
    node = next(n for n in ast.parse(node_src.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_resize')
    ns = {'comfy': types.SimpleNamespace(utils=comfy.utils)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(node_src), 'exec'), ns)
    return ns['_resize']


def ref_resize(img_hw3, hw, resize):
    """Official MiniMaxH3ReferenceToVideo 'match' sizing: down only, keep aspect, generation pixel area, multiples of 32.
    img_hw3: [H,W,3] float in [0,1]; hw: generation (height, width). Returns [1,th,tw,3]."""
    h, w = img_hw3.shape[0], img_hw3.shape[1]
    scale = min(1.0, math.sqrt((hw[1] * hw[0]) / (w * h)))
    tw = max(32, round(w * scale / 32) * 32)
    th = max(32, round(h * scale / 32) * 32)
    return resize(img_hw3[None].float(), tw, th, 'disabled')


class H3:
    def __init__(self, load_dit=True, verify_sha=False, dit='fl2va', lora_path=None, lora_rank=32):
        self.precision_initial = set_precision()
        module_to_patch = patch_module_to()
        from comfy.cli_args import args as ca
        ca.reserve_vram = 8.
        import comfy.sd
        import comfy.utils
        self.comfy_sd, self.cu = comfy.sd, comfy.utils
        self.dev = torch.device('npu', 0)
        assert VAE_PATH.stat().st_size == SIZE['vae']
        if verify_sha:
            assert sha(VAE_PATH) == SHA['vae']
        assert sha(CONTEXT_PATH) == SHA['context']
        self.record = {'precision_initial': self.precision_initial, 'precision_used': {'matmul_hf32': False, 'conv_hf32': False},
                       'torch': torch.__version__, 'torch_npu': torch_npu.__version__, **module_to_patch,
                       'comfy_commit': (COMFY / 'COMMIT').read_text().strip()}
        t = time.perf_counter()
        sd, meta = comfy.utils.load_torch_file(str(VAE_PATH), return_metadata=True)
        self.vae = comfy.sd.VAE(sd=sd, device=self.dev, dtype=torch.float16, metadata=meta)
        del sd
        self.vae.first_stage_model.eval().requires_grad_(False)
        self.record['vae_load_seconds'] = time.perf_counter() - t
        self.context = torch.load(CONTEXT_PATH, map_location='cpu', weights_only=True)
        self.patcher = None
        if load_dit:
            dit_path = DIT_PATHS[dit]
            assert dit_path.stat().st_size == SIZE['dit']
            if verify_sha:
                assert sha(dit_path) == SHA['dit' if dit == 'fl2va' else dit]
            self.record['dit'] = dit
            t = time.perf_counter()
            self.patcher = comfy.sd.load_diffusion_model(str(dit_path), model_options={'dtype': torch.bfloat16})
            model = self.patcher.model.diffusion_model.eval().requires_grad_(False)
            assert len(model.blocks) == 50 and self.patcher.model.latent_format.scale_factor == 1
            if lora_path is not None:
                import lora as L
                from safetensors.torch import load_file
                L.inject(model, lora_rank)
                n_loaded = L.load(model, load_file(str(lora_path)))
                self.record['lora'] = {'path': str(lora_path), 'rank': lora_rank, 'tensors_loaded': n_loaded,
                                       'sha256': sha(lora_path)}
            self.frozen = [(n, v, v._version) for n, v in model.named_parameters() if not (n.endswith('.lora_a') or n.endswith('.lora_b'))]
            self.record['dit_load_seconds'] = time.perf_counter() - t
        self._resize = official_resize()
        self.record['guide_helper_sha256'] = sha(COMFY / 'comfy_extras/nodes_minimax_h3.py')

    def freeze_audit(self):
        assert all(v._version == ver and v.grad is None and not v.requires_grad for _, v, ver in self.frozen)
        return {'base_parameter_versions_unchanged': True, 'frozen_base_tensors': len(self.frozen)}

    @torch.no_grad()
    def encode_guide(self, low, hw):
        """low: [T,3,h,w] float in [0,1] (observed LR only). Official AddGuide-style preprocessing."""
        up = torch.nn.functional.interpolate(low, size=hw, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        pixels = self._resize(up.permute(0, 2, 3, 1), hw[1], hw[0], 'center')
        return self.vae.encode(pixels).cpu().contiguous()

    @torch.no_grad()
    def encode_guide_pixels(self, pixels_thw3, hw):
        """Guide from already-built HR-size pixels [T,H,W,3] in [0,1] (e.g. U(LR) plus written memory detail);
        same official _resize and VAE encode as encode_guide."""
        pixels = self._resize(pixels_thw3.float(), hw[1], hw[0], 'center')
        return self.vae.encode(pixels).cpu().contiguous()

    @torch.no_grad()
    def decode(self, latent):
        px = self.vae.decode(latent).float().cpu()
        return px[0] if px.ndim == 5 else px  # [T,H,W,3]

    @torch.no_grad()
    def encode(self, rgb_thw3):
        return self.vae.encode(rgb_thw3.contiguous().clone())

    @torch.no_grad()
    def ref_block(self, resized):
        """DiT ref block for one already-resized reference image [1,th,tw,3] (see ref_resize)."""
        th, tw = resized.shape[1], resized.shape[2]
        z = self.vae.encode(resized.float()).cpu().contiguous()
        return {'kind': 'image', 'latent_h': th // 16, 'latent_w': tw // 16, 'latent': z}

    def sample(self, guide, mode, low=None, hw=None, log=None, seed=SEED, context=None, refs=None, noise=None,
               quota=None, audio_target=None, audio_strength=1.0, audio_init='zeros',
               warm_start=None, dc_project=False):
        """mode: 'base' (unadapted 49-step with LR guide) or 'innerdc' (post-CFG z0 += E(P25(D z0)) - E(D z0) at POINTS)."""
        import comfy.nested_tensor
        import comfy.sample
        from pixel_consistency import BicubicConsistency
        assert mode in ('base', 'innerdc')
        # the step count follows the sigma schedule: a turbo LoRA runs the same sampler on 8 points, not 49
        n_steps = len(SIGMAS) - 1
        # innerDC's projection points were chosen on the 49-step grid; keep their relative position
        points = sorted({min(n_steps - 1, max(0, round(i * n_steps / 49))) for i in POINTS})
        T = guide.shape[2]
        h16, w16 = guide.shape[3], guide.shape[4]
        frames = (T - 2) // 5 * 17 + 5
        audio_t = round(frames / 24 * 40)
        ctx = context if context is not None else self.context
        extra = {'minimax_token_tags': ctx['text_token_tags'], 'minimax_keyframes': [{'resolved_frame_index': 0, 'latent': guide}]}
        if refs:
            extra['minimax_refs'] = refs
        positive = [[ctx['prompt_embeds'][None], extra]]
        # The audio stream's initialisation, and it provably does not matter. This looked like a
        # train/deploy mismatch -- train_stage1.py line 159 inits with `torch.randn` while this fed exact
        # zeros -- but the flag was measured and arms A (zeros) and B (randn) came out bit-identical on all
        # 12 clips. The reason is in the sampler: SIGMAS[0] is exactly 1.0, and flow matching scales the
        # start as `sigma*noise + (1-sigma)*latent_image` (comfy/model_sampling.py CONST.noise_scaling),
        # so this tensor's CONTENT is multiplied by zero. Only its shape reaches the model; the actual
        # init is prepare_noise's N(0,1), which is the same distribution training used. The flag is kept
        # because it costs nothing and documents the measurement, but it is a no-op by construction.
        if audio_init == 'randn':
            g = torch.Generator().manual_seed(seed)
            a0 = torch.randn(1, 32, 2, audio_t, generator=g)
        elif audio_init == 'zeros':
            a0 = torch.zeros(1, 32, 2, audio_t)
        else:
            raise SystemExit('audio_init must be zeros or randn, got %r' % (audio_init,))
        # The video latent the sampler starts from. Zeros means "generate everything from noise", and
        # that is what has always been used -- after which innerDC projects the range component away and
        # replaces it with R_phi(y). So the model spends its capacity rebuilding a component we already
        # know and then has it discarded.
        #
        # warm_start supplies that component up front instead. The sampler mixes the start as
        # sigma*noise + (1-sigma)*latent_image, so the content only survives if the schedule begins
        # below 1.0 -- at sigma=1.0 it is multiplied by exactly zero, which is why the audio_init flag
        # measured as a no-op. A schedule like [0.5, 0] gives the start half the weight and needs one
        # step instead of two, which is the point: sampling is 68% of compute and the step count is the
        # only lever on it that does not need a new module.
        v0 = torch.zeros(1, 24, T, h16, w16)
        if warm_start is not None:
            v0 = warm_start.to(dtype=v0.dtype)
            if tuple(v0.shape) != (1, 24, T, h16, w16):
                raise SystemExit('warm_start latent is %s, expected %s'
                                 % (tuple(v0.shape), (1, 24, T, h16, w16)))
        latent = comfy.nested_tensor.NestedTensor((v0, a0))
        _, shapes = self.cu.pack_latents(list(latent.unbind()))
        events, state = [], {'calls': 0, 'hooks': 0}

        def observer(module, args, kwargs, output):
            if state['calls'] in [0, *points]:
                actual = kwargs['minimax_payload']['cond_video_latents'][0].detach().cpu()
                assert float((actual - guide).abs().max()) == 0
            state['calls'] += 1
        handle = self.patcher.model.diffusion_model.register_forward_hook(observer, with_kwargs=True)
        current = self.patcher.clone()
        # Same Dykstra math as the colleague's CPU float64 run, on NPU in float32: on a 22x480x832 block the
        # max abs difference is 3.6e-7 (results/proj_precision.json), 27x faster.
        # the factor is whatever `low` actually is; it used to be hardcoded to 4 here while low carried the truth
        lo_hw = tuple(low.shape[-2:]) if low is not None else (hw[0] // 4, hw[1] // 4)
        op = BicubicConsistency(tuple(hw), lo_hw, dtype=torch.float32, device=self.dev) if mode == 'innerdc' else None
        low_dev = low.to(self.dev) if mode == 'innerdc' else None

        # Spatial detail quota applied inside sampling (quota={'gate': [T,H,W] pixel activity, 's0':..,'p':..}).
        # The deployed post-hoc version multiplies the generated band down after the block is finished, which
        # throws away work the sampler already did and lets the discarded detail steer the intermediate steps
        # it was produced in. Applying the same gate to z0 at every step instead keeps the sampler anchored on
        # the observation wherever the observation says there is nothing to reconstruct. The gate is derived
        # from U(R_phi(y)) only, so it stays deployable, and it is resampled to the latent grid once.
        gate = None
        if quota is not None:
            a = quota['gate']
            while a.ndim < 5:          # [T,H,W] or [1,T,H,W] -> [1,1,T,H,W] for trilinear
                a = a[None]
            assert a.ndim == 5, a.shape
            g = F.interpolate(a.float(), size=tuple(guide.shape[2:]), mode='trilinear', align_corners=False)
            if quota['s0'] < 0:
                gate = torch.ones_like(g)      # identity check: a fully open gate must be a no-op
            else:
                gate = (g / quota['s0']).clamp(0, 1)
                if quota.get('p', 1.0) != 1.0:
                    gate = gate ** quota['p']
        # Flatten the guide and the gate once into the packed layout the sampler uses. Doing the
        # unpack/modify/pack round trip inside the hook allocated several tensors per step, which at a 44 GB
        # peak was enough to make ComfyUI offload LoRA weights to CPU mid-run and crash the next matmul.
        g5 = m5 = video_cut = vshape = None
        if gate is not None:
            import math as _math
            vshape = tuple(guide.shape)
            video_cut = _math.prod(vshape[1:])
            g5 = guide.to(device=self.dev)                       # [1,C,T,h,w]
            m5 = gate.to(device=self.dev)                        # [1,1,T,h,w], broadcasts over C

        def hook(args):
            index = state['hooks']
            state['hooks'] += 1
            assert state['calls'] == index + 1 and abs(float(args['sigma'].flatten()[0]) - float(SIGMAS[index])) < 1e-6
            if gate is not None:
                d = args['denoised']
                out = d.clone()
                gd = g5.to(dtype=out.dtype)
                gm = m5.to(dtype=out.dtype)
                v = out[:, :, :video_cut].view(*vshape)          # view, no copy
                v.sub_(gd).mul_(gm).add_(gd)                     # in place: gd + (v - gd) * gm
                if state['hooks'] == 1:
                    state['quota_dbg'] = {'denoised_type': type(d).__name__, 'denoised': str(getattr(d, 'device', '?')),
                                          'out_type': type(out).__name__, 'out_dev': str(getattr(out, 'device', '?')),
                                          'guide': str(gd.device), 'gate_mean': float(m5.float().mean())}
                    import sys as _sys
                    _sys.stderr.write('QUOTA_DBG ' + repr(state['quota_dbg']) + '\n'); _sys.stderr.flush()
                args = dict(args)
                args['denoised'] = out
                state.setdefault('quota_steps', 0)
                state['quota_steps'] += 1
            if index not in points:
                return args['denoised']
            begin = time.perf_counter()
            z0, audio = self.cu.unpack_latents(args['denoised'], shapes)
            rgb = self.decode(z0)
            assert rgb.shape == (frames, hw[0], hw[1], 3) and torch.isfinite(rgb).all()
            projected, residual = op.project(rgb.permute(0, 3, 1, 2).to(self.dev), low_dev, iterations=25)
            zc = self.encode(projected.permute(0, 2, 3, 1).cpu())
            if dc_project:
                zbase = delta = None
            else:
                zbase = self.encode(rgb)
                delta = (zc - zbase).to(device=z0.device, dtype=z0.dtype)
                assert delta.shape == z0.shape and torch.isfinite(delta).all()
            assert zc.shape == z0.shape and torch.isfinite(zc).all()
            # innerDC for the audio stream, which until now was unpacked and repacked untouched.
            # The video side computes a correction by leaving the latent, enforcing the observation
            # constraint where the operator is exactly defined, and coming back. Audio has no pixel-space
            # projection to make, because the restored latent IS the constraint: R_phi^audio already maps
            # the degraded audio latent toward the clean one, standalone and with this DiT frozen. So the
            # audio correction is a step toward that target at the same POINTS the video projection uses.
            # audio_strength=0 reproduces the previous behaviour exactly, which is the control.
            if audio_target is not None and audio_strength != 0.0:
                # The sampler does not carry audio in the VAE's units. H3 packs the audio stream onto the
                # video sigma schedule by scaling it by audio_scale = shift/audio_shift = 12/3 = 4
                # (model_base.MiniMaxH3.process_latent_in) and divides it back out in process_latent_out.
                # A target taken straight from avae.encode() is therefore 4x too small, and the first
                # measurement said exactly that: the returned audio was 0.25 * target with cos 1.0000.
                ascale = float(self.patcher.model.audio_scale())
                tgt = audio_target.to(device=audio.device, dtype=audio.dtype) * ascale
                if tgt.shape[-1] != audio.shape[-1]:
                    # The audio VAE's stride and round(frames/24*40) disagree by one frame on a 22-frame
                    # block (36 against 37). Repeat the last frame rather than padding zeros: the caller
                    # keeps only each block's `new` span, and that span's end includes this frame, so a
                    # zero here would inject a silent latent frame at every block boundary.
                    n = audio.shape[-1]
                    tgt = (torch.cat([tgt, tgt[..., -1:].expand(*tgt.shape[:-1], n - tgt.shape[-1])], -1)
                           if tgt.shape[-1] < n else tgt[..., :n])
                audio = audio + audio_strength * (tgt - audio)
            # dc_project: hand back the manifold projection of z0 plus the consistency correction,
            # i.e. zbase + delta, instead of z0 + delta. Since delta = zc - zbase that collapses to zc
            # exactly, so this branch needs ONE encode where the default needs two -- the default keeps
            # z0 and adds only the difference, which is why it has to form zbase as a reference point.
            # Measured: encode is 2.365 s and decode 1.165 s per 22-frame block, against a mean sampling
            # time of 8.89 s with exactly one correction event, so dropping an encode is 27% of sampling.
            # Motivation: decode(encode(.)) was measured to improve every temporal metric at a near-zero
            # fidelity cost (flicker better on 59/69 clips, warp 68/69, PSNR -0.017, LPIPS +0.003 median,
            # results/deploy/judge_m0.json), which says z0 carries content off the VAE manifold that the
            # projection removes. The default path never applies that projection to z0 itself.
            z_out = zc.to(device=z0.device, dtype=z0.dtype) if dc_project else z0 + delta
            packed, _ = self.cu.pack_latents([z_out, audio])
            ev = {'index': index, 'sigma': float(args['sigma'].flatten()[0]), 'dc_project': bool(dc_project),
                  'delta_rms': (None if delta is None else float(delta.float().square().mean().sqrt())),
                  'z0_to_zc_rms': float((zc.to(z0.device, z0.dtype) - z0).float().square().mean().sqrt()),
                  'projected_lr_rmse_max': float(residual.max()), 'hook_seconds': time.perf_counter() - begin}
            events.append(ev)
            if log:
                log(ev)
            return packed
        if mode == 'innerdc' or gate is not None:
            current.set_model_sampler_post_cfg_function(hook)
        t = time.perf_counter()
        try:
            with torch.no_grad(), native_training_math():
                z = comfy.sample.prepare_noise(latent, seed) if noise is None else noise
                samples = comfy.sample.sample(current, z, n_steps, 1., 'euler', 'simple', positive, [], latent,
                                              seed=seed, disable_pbar=True, sigmas=SIGMAS)
            sync()
        finally:
            handle.remove()
        video, audio = samples.unbind()
        assert state['calls'] == n_steps and torch.isfinite(video).all() and torch.isfinite(audio).all()
        if mode == 'innerdc':
            assert state['hooks'] == n_steps and [e['index'] for e in events] == points
        return {'video': video.cpu(), 'audio': audio.cpu()}, {'mode': mode, 'seed': seed, 'steps': n_steps, 'cfg': 1.,
                'audio_init': audio_init, 'audio_strength': (audio_strength if audio_target is not None else None),
                                                              'quota_steps': state.get('quota_steps', 0), 'quota_dbg': state.get('quota_dbg'),
                                                              'noise': 'absolute' if noise is not None else 'per_block',
                                                              'seconds': time.perf_counter() - t, 'correction_events': events, 'projection_device_dtype': 'npu/float32' if mode == 'innerdc' else None,
                                                              **self.freeze_audit()}


def to_u8(px):
    return np.uint8(px.numpy().clip(0, 1) * 255 + .5)
