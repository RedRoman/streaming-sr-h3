"""Bounded VAE-only input-gradient probe; no DiT, LoRA, optimizer or quality claim.

Actual execution is opt-in and must be launched under an admitted isolated NPU
queue. The synthetic normalized latent tests the decoder's input gradient only.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parameter_fingerprint(stage, torch):
    h = hashlib.sha256()
    versions = {}
    for name, value in stage.named_parameters():
        if value.requires_grad or value.grad is not None:
            raise ValueError('VAE parameter is not frozen: ' + name)
        data = value.detach().cpu().contiguous()
        h.update(json.dumps([name, str(data.dtype), list(data.shape)], separators=(',', ':')).encode())
        h.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
        versions[name] = value._version
    if not versions:
        raise ValueError('VAE has no parameters')
    return h.hexdigest(), versions


def gradient_summary(value, torch):
    """Diagnostic counts distinguish missing/zero/nonfinite without JSON NaNs."""
    if value is None:
        return {'present': False}
    value = value.detach().float().cpu()
    finite = torch.isfinite(value)
    valid = value[finite]
    nonzero = valid[valid != 0].abs()
    return {'present': True, 'shape': list(value.shape), 'elements': value.numel(),
        'dtype': str(value.dtype), 'zero_count': int((value == 0).sum()),
        'zero_fraction': float((value == 0).float().mean()),
        'finite_count': int(finite.sum()), 'nan_count': int(torch.isnan(value).sum()),
        'inf_count': int(torch.isinf(value).sum()),
        'finite_nonzero_count': int(torch.count_nonzero(valid)),
        'finite_l2': float(valid.norm()), 'finite_max_abs': float(valid.abs().max()) if valid.numel() else None,
        'finite_min_abs_nonzero': float(nonzero.min()) if nonzero.numel() else None,
        'sha256_float32': hashlib.sha256(value.numpy().tobytes()).hexdigest()}


@contextmanager
def decoder_math(native_training_math, rope_mode):
    """Choose only the grad-enabled RoPE implementation; preserve inference dispatch."""
    with native_training_math():
        import comfy.quant_ops
        prior = comfy.quant_ops.ck.rms_rope_split_half
        try:
            if rope_mode == 'ascend_native':
                # Same NPU primitives used by the deployed in-place implementation,
                # without its custom-op autograd boundary or in-place copies.
                from comfy_kitchen.backends.ascend.rope import rms_rope_split_half
                comfy.quant_ops.ck.rms_rope_split_half = rms_rope_split_half
            yield
        finally:
            comfy.quant_ops.ck.rms_rope_split_half = prior


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pretrained-manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--rope-mode', choices=['eager', 'ascend_native'], default='eager')
    ap.add_argument('--latent-t', type=int, choices=[2, 7], default=2,
                    help='2 tests five RGB frames; 7 tests 22 RGB frames and temporal chunking at 128x128')
    ap.add_argument('--geometry', choices=['small', 'full'], default='small',
                    help='small=128x128; full=480x832; neither admits joint DiT/LoRA memory')
    ap.add_argument('--checkpoint-vae-blocks', action='store_true',
                    help='nonreentrant activation checkpointing of exactly36 decoder blocks; decoder-only probe')
    ap.add_argument('--expected-gradient-sha256',
                    help='optional exact float32 input-gradient SHA from an otherwise identical completed probe')
    ap.add_argument('--diagnose-loss-scale', type=float,
                    help='explicit paired diagnostic: unscaled retained-graph backward then scaled backward, unscale FP32 input grad; no optimizer')
    ap.add_argument('--loss-scale', type=float, default=1.0,
                    help='single-backward fixed loss multiplier, followed by FP32 input-gradient unscale; no optimizer')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.loss_scale <= 65536:
        ap.error('loss scale must satisfy 1<=scale<=65536')
    if args.diagnose_loss_scale is not None:
        if not 1 < args.diagnose_loss_scale <= 65536 or args.expected_gradient_sha256 or args.loss_scale != 1:
            ap.error('loss-scale diagnostic requires 1<scale<=65536 and no expected-gradient SHA')
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(args.out)
    from openh3_assets import read_manifest
    manifest = read_manifest(args.pretrained_manifest, verify=False)
    latent_h, latent_w = (30, 52) if args.geometry == 'full' else (8, 8)
    contract = {'schema': 'openh3.vae_decode_input_gradient_probe.v2' if args.diagnose_loss_scale else ('openh3.vae_decode_input_gradient_probe.v3' if args.loss_scale != 1 else 'openh3.vae_decode_input_gradient_probe.v1'),
        'diagnostic_kind': 'loss_scale_pair' if args.diagnose_loss_scale else ('scaled_single_backward' if args.loss_scale != 1 else 'single_backward'),
        'manifest_sha256': digest(args.pretrained_manifest),
        'pretrained_fingerprint': manifest['_fingerprint'],
        'implementation_sha256': digest(__file__), 'seed': args.seed,
        'grad_rope_mode': args.rope_mode,
        'checkpoint_vae_blocks': args.checkpoint_vae_blocks,
        'expected_gradient_sha256': args.expected_gradient_sha256,
        'paired_loss_scale_diagnostic': args.diagnose_loss_scale,
        'fixed_loss_scale': args.loss_scale,
        'latent_shape': [1, 24, args.latent_t, latent_h, latent_w], 'latent_distribution': 'CPU seeded float32 N(0,1), normalized native latent units',
        'expected_rgb_shape': [(args.latent_t - 2) // 5 * 17 + 5, latent_h * 16, latent_w * 16, 3],
        'forward_comparison_atol': 0.0002, 'forward_comparison_rtol': 0.0001,
        'scope': 'synthetic decoder input-gradient only at declared geometry; no original-RGB restoration loss, no adapter gradient, no joint fullH3 memory acceptance',
        'planned_model_forwards': 2, 'planned_backward_calls': 2 if args.diagnose_loss_scale else 1, 'H3_denoiser_calls': 0,
        'optimizer_steps': 0, 'VAE_parameters_updated': False}
    if not args.execute:
        print(json.dumps({'inspection_only': True, 'actual_asset_hashes_verified': False, 'contract': contract}, indent=2))
        return
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')
    if not visible.isdecimal():
        raise ValueError('explicit single admitted physical NPU required')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed_forwards = 0
    completed_backwards = 0
    gradient_diagnostics = {}
    try:
        import torch
        from h3_comfy import native_training_math
        from run_vae_roundtrip_diagnostic import load_vae_only
        from vae_checkpoint_context import checkpoint_decoder_blocks
        h3 = load_vae_only(manifest)
        vae = h3.vae
        stage = vae.first_stage_model
        vae.device = h3.dev
        vae.offload_device = h3.dev
        vae.disable_offload = True
        stage.eval().requires_grad_(False).to(h3.dev)
        before_hash, before_versions = parameter_fingerprint(stage, torch)
        gen = torch.Generator(device='cpu').manual_seed(args.seed)
        source = torch.randn(contract['latent_shape'], generator=gen, dtype=torch.float32)
        source_hash = hashlib.sha256(source.numpy().tobytes()).hexdigest()
        torch.npu.reset_peak_memory_stats()
        with native_training_math(), torch.no_grad():
            reference = h3.decode(source.to(device=h3.dev, dtype=vae.vae_dtype))
        completed_forwards += 1
        if reference.requires_grad or list(reference.shape) != contract['expected_rgb_shape']:
            raise ValueError('unexpected inference reference')
        latent = source.to(device=h3.dev, dtype=torch.float32).requires_grad_(True)
        with torch.enable_grad(), decoder_math(native_training_math, args.rope_mode), checkpoint_decoder_blocks(stage, args.checkpoint_vae_blocks) as checkpoint_audit:
            pixels = vae.decode(latent.to(dtype=vae.vae_dtype)).float().cpu()
            completed_forwards += 1
            pixels = pixels[0] if pixels.ndim == 5 else pixels
            if list(pixels.shape) != contract['expected_rgb_shape'] or not pixels.requires_grad:
                raise ValueError('direct decoder output is not differentiable or has wrong shape')
            if not torch.isfinite(pixels).all():
                raise FloatingPointError('nonfinite decoder output')
            max_abs = float((pixels.detach() - reference).abs().max())
            if not torch.allclose(pixels.detach(), reference,
                                  atol=contract['forward_comparison_atol'], rtol=contract['forward_comparison_rtol']):
                raise ValueError('gradient-mode decoder forward differs: ' + str(max_abs))
            # Fixed synthetic probe target, never a training/dev record.
            target = torch.linspace(0.2, 0.8, pixels.shape[-2], dtype=torch.float32)[None, None, :, None]
            loss = (pixels - target).square().mean()
            first_loss = loss * args.loss_scale if args.loss_scale != 1 else loss
            first_loss.backward(retain_graph=args.diagnose_loss_scale is not None)
            completed_backwards += 1
            gradient_diagnostics['scaled_loss_backward_raw' if args.loss_scale != 1 else 'unscaled_loss_backward'] = gradient_summary(latent.grad, torch)
            if args.diagnose_loss_scale is not None:
                latent.grad = None
                (loss * args.diagnose_loss_scale).backward()
                completed_backwards += 1
                gradient_diagnostics['scaled_loss_backward_raw'] = gradient_summary(latent.grad, torch)
        torch.npu.synchronize()
        grad = latent.grad
        if grad is None or not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
            raise FloatingPointError('decoder input gradient missing, nonfinite or zero')
        grad_cpu = grad.detach().float().cpu()
        effective_loss_scale = args.diagnose_loss_scale or args.loss_scale
        if effective_loss_scale != 1:
            grad_cpu = grad_cpu / effective_loss_scale
            gradient_diagnostics['scaled_loss_backward_unscaled_float32'] = gradient_summary(grad_cpu, torch)
        gradient_hash = hashlib.sha256(grad_cpu.numpy().tobytes()).hexdigest()
        if args.expected_gradient_sha256 and gradient_hash != args.expected_gradient_sha256:
            raise ValueError('input gradient SHA differs from pinned reference: ' + gradient_hash)
        after_hash, after_versions = parameter_fingerprint(stage, torch)
        if before_hash != after_hash or before_versions != after_versions:
            raise ValueError('frozen VAE parameters changed')
        report = {'complete': True, 'contract': contract,
            'actual_completed_forwards': completed_forwards, 'actual_completed_backwards': completed_backwards,
            'latent_sha256_float32': source_hash,
            'decoded_shape': list(pixels.shape), 'decode_dtype': str(vae.vae_dtype),
            'gradient_mode_output_requires_grad': pixels.requires_grad,
            'inference_wrapper_output_requires_grad': reference.requires_grad,
            'forward_max_abs_difference': max_abs,
            'forward_bitwise_equal': bool(torch.equal(pixels.detach(), reference)),
            'synthetic_loss': float(loss.detach()),
            'gradient_diagnostics': gradient_diagnostics,
            'gradient_loss_scale': effective_loss_scale,
            'decoder_checkpoint': checkpoint_audit,
            'expected_gradient_sha256_matched': gradient_hash == args.expected_gradient_sha256 if args.expected_gradient_sha256 else None,
            'latent_gradient': {'finite': True, 'nonzero_count': int(torch.count_nonzero(grad_cpu)),
                'elements': grad_cpu.numel(), 'l2': float(grad_cpu.norm()), 'max_abs': float(grad_cpu.abs().max()),
                'sha256_float32': gradient_hash},
            'VAE_parameter_sha256_before': before_hash, 'VAE_parameter_sha256_after': after_hash,
            'VAE_parameter_count': len(before_versions), 'VAE_parameter_versions_unchanged': True,
            'VAE_parameters_have_no_grad': True,
            'npu_peak_allocated_bytes_after_load': torch.npu.max_memory_allocated(),
            'npu_peak_reserved_bytes_after_load': torch.npu.max_memory_reserved(),
            'seconds': time.monotonic() - started, 'model': h3.record,
            'limitations': ['No DiT/LoRA backward tested',
                f'Decoder-only probe at {contract["expected_rgb_shape"]}; no joint DiT/LoRA memory measurement',
                'Paired-loss-scale mode uses retain_graph and two backwards; its peak is not single-backward training memory' if args.diagnose_loss_scale else 'Single backward probe',
                'Loss-scale sensitivity alone does not isolate the RGB mean reduction versus intermediate FP16 gradient arithmetic',
                'Synthetic latent/loss is a numerical probe, not a restoration quality or training objective result']}
        with (args.out / 'report.json').open('x') as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        with (args.out / 'failure.json').open('x') as f:
            json.dump({'complete': False, 'contract': contract, 'error': repr(error),
                'actual_completed_forwards': completed_forwards, 'actual_completed_backwards': completed_backwards,
                'gradient_diagnostics': gradient_diagnostics,
                'npu_peak_allocated_bytes_after_load': torch.npu.max_memory_allocated() if 'torch' in locals() and torch.npu.is_initialized() else None,
                'npu_peak_reserved_bytes_after_load': torch.npu.max_memory_reserved() if 'torch' in locals() and torch.npu.is_initialized() else None,
                'seconds': time.monotonic() - started}, f, indent=2)
        raise


if __name__ == '__main__':
    main()
