"""Stage-1 pilot: LoRA on frozen MiniMax-H3 FL2VA (ComfyUI implementation) for x4 restoration.
Conditioning = native keyframe guide (official LR-guide latent, same interface as the frozen inner-DC baseline).
Loss = video flow-matching velocity MSE to (eps - z_HR); audio stream gets noise and no loss (pilot: video only).
Data-parallel with manual all-reduce of LoRA grads (HCCL). Launch with torchrun."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file

import lora as L
from h3_comfy import VAE_PATH, CONTEXT_PATH, DIT_PATH, ROOT, COMFY, native_training_math, patch_module_to, set_precision

SHIFT = 12.0


def official_stream_inputs(video_x, audio_noise, sigma, time_shift_sigma):
    """Same packing as Comfy BaseModel -> MiniMaxH3.forward (colleague's audited native_io helper)."""
    st = torch.as_tensor(sigma, device=video_x.device, dtype=torch.float32).reshape(1)
    timestep = st * 1000
    s = (timestep.flatten()[0] / 1000).float().clamp(min=1e-6)
    carry = (time_shift_sigma(s, 12, 3) / s).to(torch.bfloat16)
    audio_carried = (st.reshape(1, 1, 1, 1) * audio_noise).to(torch.bfloat16)
    return video_x.to(torch.bfloat16), audio_carried * carry, timestep


def log(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT / 'data/pilot_latents')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--rank', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--save_every', type=int, default=250)
    p.add_argument('--val_every', type=int, default=250)
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--pretrained-manifest', type=Path,
                   help='verified Open-H3 Comfy recipe; excludes old Stage-1/Turbo weights')
    p.add_argument('--train-manifest', type=Path)
    p.add_argument('--val-manifest', type=Path)
    p.add_argument('--draft-key', default='draft')
    p.add_argument('--draft-source', choices=['frozen_openh3_upsampler', 'canonical_restore_latent',
                                             'pixel_upsample_guide_fallback'])
    p.add_argument('--smoke', action='store_true',
                   help='one update, two latent frames and an 8x8 latent crop; not a quality evaluation')
    p.add_argument('--inspect-only', action='store_true',
                   help='CPU manifest/header contract checks only; no distributed/NPU initialization')
    p.add_argument('--perc', type=float, default=0.0,
                   help="weight on an LPIPS term computed on decoded pixels. Velocity MSE is only weakly "
                        "related to what we are judged on: training with a clean range estimate cut it by "
                        "39%% and bought 0.0184 LPIPS, and switching the training data from generated to "
                        "real footage moved LPIPS by 0.0003. Real-ESRGAN optimises perceptual quality "
                        "directly and beats us on it, so this optimises it directly too.")
    p.add_argument('--perc-sigma', type=float, default=0.6,
                   help='only apply it below this sigma, where x0 is close enough to be worth decoding')
    p.add_argument('--perc-crop', type=int, default=16, help='latent crop side; 16 -> 256x256 pixels')
    p.add_argument('--guide-key', default='guide',
                   help="which guide the model is conditioned on. Stage-7 latents carry two: guide_raw is "
                        "the degraded observation Stage 1 used, guide_oracle is the same pipeline fed the "
                        "true D(x). Training on the oracle bounds what a range branch could ever buy, "
                        "which is cheaper to measure now than after R_phi converges.")
    a = p.parse_args()
    from openh3_assets import (NATIVE_SIGMAS, read_manifest, select_samples, validate_split_sources,
                               sha256, safetensors_header)
    from openh3_runtime import (merge_pretrained, inject_adaptation, frozen_snapshot, assert_frozen,
                                zero_output_parity, adapter_metadata, check_adapter)
    pretrained = None
    if a.pretrained_manifest:
        if not a.train_manifest or not a.val_manifest or not a.draft_source:
            p.error('Open-H3 requires explicit train/val source manifests and --draft-source')
        pretrained = read_manifest(a.pretrained_manifest, verify=not a.inspect_only)
    tr_rows = select_samples(a.train_manifest, a.data, 'train') if a.train_manifest else None
    va_rows = select_samples(a.val_manifest, a.data, 'val') if a.val_manifest else None
    if pretrained:
        validate_split_sources(tr_rows, va_rows)
    train_files = [r['path'] for r in tr_rows] if tr_rows else sorted((a.data / 'train').glob('*.safetensors'))
    val_files = [r['path'] for r in va_rows] if va_rows else sorted((a.data / 'val').glob('*.safetensors'))[:16]
    if not train_files or not val_files:
        p.error('nonempty training and validation samples are required')
    keys = ['target', a.guide_key] + ([a.draft_key] if pretrained else [])
    # All selected headers are checked before any accelerator is initialized.
    for path in train_files + val_files:
        header = safetensors_header(path)
        if any(k not in header for k in keys):
            raise ValueError('missing requested latent keys %s in %s' % (keys, path))
        shapes = [header[k]['shape'] for k in keys]
        if any(len(s) != 5 or s[:2] != [1, 24] for s in shapes) or any(s != shapes[0] for s in shapes):
            raise ValueError('incompatible H3 target/guide/draft shapes in %s' % path)
        t, h, w = shapes[0][2:]
        if t < 2 or (t - 2) % 5 or h % 2 or w % 2:
            raise ValueError('H3 requires T=2+5k and even spatial latent dimensions: %s' % path)
    if a.inspect_only:
        print(json.dumps({'status': 'headers_valid', 'full_asset_sha256_verified': False,
                          'train_samples': len(train_files), 'val_samples': len(val_files),
                          'train_sources': len({r['source_id'] for r in tr_rows}) if tr_rows else None,
                          'val_sources': len({r['source_id'] for r in va_rows}) if va_rows else None,
                          'keys': keys, 'draft_source': a.draft_source,
                          'pretrained_fingerprint': pretrained['_fingerprint'] if pretrained else None}, indent=2))
        return
    if a.smoke:
        a.steps = a.save_every = a.val_every = 1
        val_files = val_files[:1]
    training_contract = {'objective': 'warmstart_x0' if pretrained else 'legacy_flow_velocity',
                         'sigmas': list(NATIVE_SIGMAS[:-1]) if pretrained else 'shift12_random',
                         'guide_key': a.guide_key, 'draft_key': a.draft_key if pretrained else None,
                         'draft_source': a.draft_source, 'smoke': a.smoke,
                         'train_manifest_sha256': sha256(a.train_manifest) if a.train_manifest else None,
                         'val_manifest_sha256': sha256(a.val_manifest) if a.val_manifest else None}
    dist.init_process_group('hccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local)
    torch.npu.set_compile_mode(jit_compile=False)
    dev = torch.device('npu', local)
    set_precision(); patch_module_to()
    a.out.mkdir(parents=True, exist_ok=True)
    logf = a.out / 'train_log.jsonl'
    sys_path = COMFY
    import sys
    sys.path.insert(0, str(sys_path))
    from comfy.cli_args import args as ca
    ca.reserve_vram = 4.
    import comfy.sd
    from comfy.ldm.minimax.model import PackedLayout, time_shift_sigma

    if rank == 0:
        (a.out / 'config.json').write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, 'world': world,
                                                       'n_train': len(train_files), 'n_val': len(val_files),
                                                       'conditioning': 'native keyframe guide', 'guide_key': a.guide_key,
                                                       'lora_targets': list(L.TARGETS),
                                                       'sigma_sampling': training_contract['sigmas'],
                                                       'audio': 'noise input, no loss (Comfy port; not release silence)',
                                                       'text': 'cached neutral restore context',
                                                       'training_contract': training_contract}, indent=1))

    dit_path = pretrained['_paths']['base'] if pretrained else DIT_PATH
    context_path = pretrained['_paths']['context'] if pretrained else CONTEXT_PATH
    vae_path = pretrained['_paths']['vae'] if pretrained else VAE_PATH
    patcher = comfy.sd.load_diffusion_model(str(dit_path), model_options={'dtype': torch.bfloat16})
    model = patcher.model.diffusion_model.eval().requires_grad_(False)
    pre_rec = merge_pretrained(model, a.pretrained_manifest, pretrained) if pretrained else None
    model.to(dev)
    torch.manual_seed(a.seed)
    names = inject_adaptation(model, a.rank, a.seed) if pretrained else L.inject(model, a.rank)
    for blk in model.blocks:
        orig = blk.forward

        def ck(*inputs, _o=orig, **kw):
            if not torch.is_grad_enabled():
                return _o(*inputs, **kw)
            if 'transformer_options' in kw:
                kw = {**kw, 'transformer_options': dict(kw['transformer_options'])}
            return checkpoint(_o, *inputs, use_reentrant=False, **kw)
        blk.forward = ck
    params = [p_ for _, p_ in L.lora_parameters(model)]
    for p_ in params:
        p_.requires_grad_(True)
        dist.broadcast(p_.data, 0)  # identical init on all ranks
    n_params = sum(p_.numel() for p_ in params)
    frozen = frozen_snapshot(model)
    assert_frozen(model, frozen)
    if rank == 0 and pre_rec:
        (a.out / 'pretrained_provenance.json').write_text(json.dumps(pre_rec, indent=2))
    cache = torch.load(context_path, map_location='cpu', weights_only=True)
    tags = cache['text_token_tags'].to(dev)
    with torch.no_grad():
        context = model.preprocess_text_embeds(cache['prompt_embeds'][None].to(dev).to(torch.bfloat16)).detach()

    vae = None
    lpips_net = None
    if a.perc > 0:
        import lpips as _lp
        sd_vae, meta_vae = comfy.utils.load_torch_file(str(vae_path), return_metadata=True)
        vae = comfy.sd.VAE(sd=sd_vae, device=dev, dtype=torch.float16, metadata=meta_vae)
        vae.first_stage_model.eval().requires_grad_(False).to(dev)
        # Pin the VAE on the device. ComfyUI manages VAE memory itself and will offload parts of it under
        # pressure; with rank 32 that never triggered, but a larger LoRA leaves less headroom and the
        # offloaded weights come back mid-forward, so the backward pass through the decode then sees a
        # layer_norm weight on cpu against activations on npu.
        vae.device = dev
        vae.offload_device = dev
        # comfy's VAE.decode calls load_models_gpu(force_full_load=self.disable_offload); left at False it
        # partially loads under memory pressure, which a larger LoRA creates. The backward pass through the
        # decode then finds half the VAE on cpu. Forcing a full load keeps it whole.
        vae.disable_offload = True
        del sd_vae
        lpips_net = _lp.LPIPS(net='alex', version='0.1', verbose=False).eval().to(dev)
        for q in lpips_net.parameters():
            q.requires_grad_(False)

    def perceptual(x0_hat, target, gen):
        """LPIPS between decoded x0 and the decoded target, on a random latent crop.

        Two latent frames is the smallest unit the H3 VAE accepts (its temporal layout is 2+5k, so 2 gives
        5 pixel frames), and a 16x16 latent crop keeps the decode that the gradient has to pass through
        small enough to fit alongside the DiT's own activations."""
        c = a.perc_crop
        H16, W16 = x0_hat.shape[-2], x0_hat.shape[-1]
        y0 = int(torch.randint(0, max(1, H16 - c + 1), (1,), generator=gen, device=dev).cpu().item())
        x0 = int(torch.randint(0, max(1, W16 - c + 1), (1,), generator=gen, device=dev).cpu().item())
        sl = (slice(None), slice(None), slice(0, 2), slice(y0, y0 + c), slice(x0, x0 + c))
        dec_hat = vae.decode(x0_hat[sl].to(torch.float16))
        with torch.no_grad():
            dec_tgt = vae.decode(target[sl].to(torch.float16))
        f = lambda z: (z[0] if z.ndim == 5 else z).permute(0, 3, 1, 2).float().clamp(0, 1).to(dev) * 2 - 1
        return lpips_net(f(dec_hat), f(dec_tgt)).mean()

    def forward(target, guide, draft, sigma, gen, perc=True, return_prediction=False):
        eps = torch.randn(target.shape, device=dev, generator=gen)
        s = torch.tensor(sigma, device=dev, dtype=torch.float32)
        # The adaptation objective explicitly starts from the available draft.
        # It is not advertised as reproducing the release's training objective.
        xt = (1 - s) * (draft if pretrained else target) + s * eps
        tt, hh, ww = guide.shape[2:]
        audio_t = max(1, round(((tt - 2) // 5 * 17 + 5) / 24 * 40)) if pretrained else 37
        audio = torch.randn((1, 32, 2, audio_t), device=dev, generator=gen)
        kf = [{'resolved_frame_index': 0, 'latent': guide}]
        payload = {'text_token_tags': tags, 'keyframes': kf, 'cond_video_latents': [guide], 'seed': 0, 'audio_scale': 1.0,
                   'layout': PackedLayout(context.shape[1], tt, hh, ww, audio_t, keyframes=kf)}
        vio, aio, ts = official_stream_inputs(xt, audio, sigma, time_shift_sigma)
        out = model([vio, aio], ts, context, minimax_payload=payload, transformer_options={})
        v = out[0].float()
        if return_prediction:
            return v
        x0_hat = xt.float() - s * v
        mse = F.mse_loss(x0_hat, target.float()) if pretrained else F.mse_loss(v, eps - target)
        # Validation reports only the declared regression objective, excluding LPIPS.
        # Legacy validation uses velocity MSE: two of its three
        # sigmas sit below the perceptual threshold, which would otherwise fold an LPIPS term into the
        # number and make a perc=4 run look worse than a perc=0 one at step 0.
        if not perc or a.perc <= 0 or sigma > a.perc_sigma:
            return mse
        # flow matching: xt = x + s*(eps - x), so x0 = xt - s*v
        return mse + a.perc * perceptual(x0_hat, target.float(), gen)

    def load(path):
        d = load_file(str(path))
        # No silent guide fallback: an oracle/raw swap changes the experiment.
        keys = ['target', a.guide_key] + ([a.draft_key] if pretrained else [])
        if any(k not in d for k in keys):
            raise ValueError('missing requested latent key in %s: %s' % (path, keys))
        tensors = [d[k] for k in keys]
        if any(z.ndim != 5 or z.shape[0] != 1 or z.shape[1] != 24 for z in tensors):
            raise ValueError('expected normalized H3 latents [1,24,T,H16,W16]')
        if any(z.shape != tensors[0].shape for z in tensors):
            raise ValueError('target, guide and draft must share the same latent geometry')
        if a.smoke:
            tensors = [z[:, :, :2, :8, :8].contiguous() for z in tensors]
        if any(not torch.isfinite(z).all() for z in tensors):
            raise ValueError('nonfinite input latent')
        tensors = [z.to(dev) for z in tensors]
        return tensors[0], tensors[1], tensors[2] if pretrained else None

    def validate(step):
        losses = []
        with torch.no_grad(), native_training_math():
            for i, f in enumerate(val_files):
                if i % world != rank:
                    continue
                tgt, gd, draft = load(f)
                for j, sg in enumerate(NATIVE_SIGMAS[:-1] if pretrained else [0.2, 0.5, 0.8]):
                    g = torch.Generator(device=dev).manual_seed(900000 + i * 10 + j)
                    losses.append(float(forward(tgt, gd, draft, sg, g, perc=False)))
        t = torch.tensor([sum(losses), len(losses)], device=dev, dtype=torch.float32)  # HCCL has no float64
        dist.all_reduce(t)
        v = float(t[0] / t[1])
        if rank == 0:
            key = 'val_warmstart_x0_mse' if pretrained else 'val_velocity_mse'
            log(logf, {'event': 'val', 'step': step, key: v})
            print(json.dumps({'val_step': step, key: round(v, 5)}), flush=True)

    # zero-adapter parity: LoRA B is zero, so the injected model equals the base exactly
    start_step = 0
    ckpts = sorted(a.out.glob('lora_step*.safetensors'))
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    if pretrained:
        tgt0, gd0, draft0 = load(train_files[0])
        def parity_forward():
            gen = torch.Generator(device=dev).manual_seed(a.seed)
            with native_training_math():
                return forward(tgt0, gd0, draft0, NATIVE_SIGMAS[0], gen, return_prediction=True)
        parity = zero_output_parity(model, parity_forward)
        if rank == 0:
            log(logf, {'event': 'zero_adaptation_parity', **parity})
        if not a.smoke:
            del tgt0, gd0, draft0
    if ckpts:
        if pretrained:
            check_adapter(ckpts[-1], pre_rec, a.rank, training_contract)
        sd = load_file(str(ckpts[-1]))
        L.load(model, sd)
        start_step = int(ckpts[-1].stem.split('step')[1])
        st = a.out / f'opt_step{start_step:06d}_rank{rank}.pt'
        if st.exists():
            opt.load_state_dict(torch.load(st, map_location=dev))
    if rank == 0:
        log(logf, {'event': 'start', 'start_step': start_step, 'lora_params': n_params, 'lora_modules': len(names), 'world': world})
    if start_step == 0:
        validate(0)
    rng = random.Random(a.seed * 1000 + rank)
    model.train(False)
    t_last = time.time()
    for step in range(start_step + 1, a.steps + 1):
        if pretrained:
            # Data selection is a function of step/rank, so resume does not replay
            # the beginning of the random sample sequence.
            rng = random.Random(a.seed * 1000 + rank + step * 1000003)
        lr = a.lr * min(1.0, step / a.warmup)
        for g_ in opt.param_groups:
            g_['lr'] = lr
        f = train_files[rng.randrange(len(train_files))]
        tgt, gd, draft = load(f)
        u = rng.random()
        sigma = (NATIVE_SIGMAS[(step - 1) % 3] if pretrained else
                 float(min(max(SHIFT * u / (1 + (SHIFT - 1) * u), 0.02), 0.995)))
        gen = torch.Generator(device=dev).manual_seed(a.seed + step * 1009 + rank)
        with native_training_math():
            loss = forward(tgt, gd, draft, sigma, gen)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            loss.backward()
        assert_frozen(model, frozen)
        for p_ in params:
            if p_.grad is None:
                p_.grad = torch.zeros_like(p_)
            dist.all_reduce(p_.grad)
            p_.grad.div_(world)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            raise FloatingPointError('nonfinite adaptation gradient norm')
        if a.smoke and float(gn) <= 0:
            raise AssertionError('smoke produced no adaptation gradient')
        opt.step(); opt.zero_grad(set_to_none=True)
        assert_frozen(model, frozen)
        if a.smoke:
            bs = [p_ for name, p_ in L.lora_parameters(model) if name.endswith('.lora_b')]
            if not all(torch.isfinite(p_).all() for p_ in bs):
                raise FloatingPointError('smoke produced a nonfinite LoRA B')
            if not any(torch.count_nonzero(p_).item() for p_ in bs):
                raise AssertionError('smoke update left every zero-initialized LoRA B unchanged')
        lt = torch.tensor([float(loss)], device=dev); dist.all_reduce(lt); lt /= world
        if rank == 0 and (step % 10 == 0 or step == start_step + 1):
            now = time.time()
            rec = {'event': 'train', 'step': step, 'loss': float(lt), 'grad_norm': float(gn), 'lr': lr, 'sigma_rank0': sigma,
                   'sec_per_step': (now - t_last) / (10 if step % 10 == 0 else 1), 'peak_GB': torch.npu.max_memory_allocated() / 1e9}
            t_last = now
            log(logf, rec); print(json.dumps(rec), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            if rank == 0:
                metadata = adapter_metadata(pre_rec, training_contract, a.rank) if pretrained else None
                save_file(L.state_dict(model), str(a.out / f'lora_step{step:06d}.safetensors'), metadata=metadata)
            torch.save(opt.state_dict(), a.out / f'opt_step{step:06d}_rank{rank}.pt')
            old = sorted(a.out.glob(f'opt_step*_rank{rank}.pt'))[:-1]
            for o in old:
                o.unlink()
            if a.smoke and pretrained:
                # A real-model serialization gate, not only the tiny CPU fixture:
                # record prediction, destroy the live adaptation, reload from disk,
                # and require exact equality for identical input/noise.
                dist.barrier()
                path = a.out / f'lora_step{step:06d}.safetensors'
                check_adapter(path, pre_rec, a.rank, training_contract)
                with torch.no_grad():
                    expected = parity_forward().detach().clone()
                    for _, parameter in L.lora_parameters(model):
                        parameter.zero_()
                    L.load(model, load_file(str(path)), strict=True, strict_unused=True)
                    actual = parity_forward().detach()
                if not torch.isfinite(actual).all() or not torch.equal(expected, actual):
                    raise AssertionError('real checkpoint reload changed prediction: %s' %
                                         float((actual - expected).abs().max()))
                assert_frozen(model, frozen)
                if rank == 0:
                    log(logf, {'event': 'checkpoint_reload_parity', 'step': step, 'max_abs': 0.0,
                               'passed': True, 'finite_nonzero_gradient': True,
                               'zero_initialized_B_changed': True, 'frozen_base_unchanged': True})
        if step % a.val_every == 0 or step == a.steps:
            validate(step)
    if rank == 0:
        log(logf, {'event': 'done', 'step': a.steps})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
