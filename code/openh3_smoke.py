"""Inspect or execute one explicit Open-H3 Comfy warm-start block.

--inspect-only uses no torch/NPU. --cpu-self-test checks the integration contract
on a tiny CPU model. Actual inference is a separate, explicitly invoked mode and
is not an end-to-end upstream Open-H3 reproduction or a quality benchmark.
"""
import argparse
import json
from pathlib import Path

from openh3_assets import NATIVE_SIGMAS, read_manifest, safetensors_header


def cpu_self_test():
    import tempfile
    import torch
    from torch import nn
    from safetensors.torch import save_file, load_file
    import lora as L
    from openh3_assets import sha256
    from openh3_runtime import (merge_pretrained, inject_adaptation, frozen_snapshot,
                               assert_frozen, zero_output_parity, adapter_metadata, check_adapter,
                               merge_adaptation_for_inference)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Module()
            self.mlp = nn.Module()
            for parent, names in ((self.attn, ('qkv_proj', 'out_proj')), (self.mlp, ('fc1', 'fc2'))):
                for name in names:
                    setattr(parent, name, nn.Linear(4, 4))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(50)])
            self.token_refiner = nn.Module()
            self.token_refiner.blocks = nn.ModuleList([Block() for _ in range(2)])

        def forward(self, x):
            b = self.blocks[0]
            return b.mlp.fc2(torch.tanh(b.mlp.fc1(b.attn.out_proj(b.attn.qkv_proj(x)))))

    with tempfile.TemporaryDirectory(prefix='openh3-cpu-test-') as temp:
        root = Path(temp)
        dummy = root / 'dummy'
        dummy.write_bytes(b'tiny CPU fixtures, not actual H3 assets')
        rec = lambda p: {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
        torch.manual_seed(7)
        model = Tiny().eval().requires_grad_(False)
        weights = {n: p.clone() for n, p in model.state_dict().items()}
        sd = {}
        for prefix, count in (('blocks', 50), ('token_refiner.blocks', 2)):
            for i in range(count):
                for target in L.TARGETS:
                    name = f'diffusion_model.{prefix}.{i}.{target}'
                    sd[name + '.lora_A.weight'] = torch.randn(2, 4) * .1
                    sd[name + '.lora_B.weight'] = torch.randn(4, 2) * .1
        adapter = root / 'pretrained.safetensors'
        save_file(sd, str(adapter))
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'layout': 'comfyui_minimax_h3',
                                       'recipe': 'refiner-only-comfy', 'base': rec(dummy),
                                       'vae': rec(dummy), 'context': rec(dummy),
                                       'components': [{'role': 'refiner', **rec(adapter)}]}))
        pretrained = merge_pretrained(model, manifest)
        assert not torch.equal(model.blocks[0].attn.qkv_proj.weight, weights['blocks.0.attn.qkv_proj.weight'])
        try:
            merge_pretrained(model, manifest)
        except ValueError:
            duplicate_rejected = True
        else:
            raise AssertionError('duplicate pretrained merge accepted')
        inject_adaptation(model, 2, seed=13)
        frozen = frozen_snapshot(model)
        x = torch.randn(3, 4)
        zero = zero_output_parity(model, lambda: model(x))
        params = [p for _, p in L.lora_parameters(model)]
        optimizer = torch.optim.AdamW(params, lr=.01)
        loss = model(x).square().mean()
        loss.backward()
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in params)
        assert_frozen(model, frozen)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert_frozen(model, frozen)
        trained = model(x).detach()
        checkpoint = root / 'adaptation.safetensors'
        contract = {'objective': 'tiny_cpu_test'}
        save_file(L.state_dict(model), str(checkpoint), metadata=adapter_metadata(pretrained, contract, 2))
        check_adapter(checkpoint, pretrained, 2, contract)
        clone = Tiny().eval().requires_grad_(False)
        clone.load_state_dict(weights)
        merge_pretrained(clone, manifest)
        inject_adaptation(clone, 2, seed=99)
        L.load(clone, load_file(str(checkpoint)), strict=True, strict_unused=True)
        assert torch.equal(clone(x).detach(), trained)
        inference_merge = merge_adaptation_for_inference(clone)
        merged_prediction = clone(x).detach()
        if not torch.allclose(merged_prediction, trained, rtol=1e-5, atol=1e-6):
            raise AssertionError('FP32 inference merge exceeded numerical tolerance')
        merge_error = float((merged_prediction - trained).abs().max())
        assert not L.lora_parameters(clone)
        wrong = {**pretrained, 'fingerprint': '0' * 64}
        try:
            check_adapter(checkpoint, wrong, 2)
        except ValueError:
            provenance_rejected = True
        else:
            raise AssertionError('wrong checkpoint base accepted')
        return {'status': 'passed', 'scope': 'tiny CPU contract only, no actual H3/NPU',
                'zero_output_parity': zero, 'frozen_after_update': True, 'checkpoint_reload_exact': True,
                'inference_merge_fp32_max_abs': merge_error, 'inference_merge': inference_merge,
                'duplicate_pretrained_rejected': duplicate_rejected,
                'wrong_provenance_rejected': provenance_rejected}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--latent', type=Path)
    ap.add_argument('--draft-key', default='draft')
    ap.add_argument('--guide-key', default='guide')
    ap.add_argument('--draft-source', choices=['frozen_openh3_upsampler', 'canonical_restore_latent',
                                             'pixel_upsample_guide_fallback'])
    ap.add_argument('--adaptation', type=Path)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--inspect-only', action='store_true')
    ap.add_argument('--cpu-self-test', action='store_true')
    ap.add_argument('--tiny-crop', action='store_true', help='two latent frames, 8x8 spatial; interface smoke only')
    a = ap.parse_args()
    if a.cpu_self_test:
        print(json.dumps(cpu_self_test(), indent=2))
        return
    if not a.manifest or not a.latent or not a.draft_source:
        ap.error('--manifest, --latent and --draft-source are required')
    m = read_manifest(a.manifest, verify=False)  # full hashing happens once inside H3 before loading
    h = safetensors_header(a.latent)
    for key in (a.draft_key, a.guide_key):
        if key not in h:
            raise ValueError('missing explicit latent key: ' + key)
    shape = h[a.draft_key]['shape']
    if len(shape) != 5 or shape[:2] != [1, 24] or shape != h[a.guide_key]['shape']:
        raise ValueError('draft/guide must have equal [1,24,T,H16,W16] geometry')
    report = {'recipe': m['recipe'], 'fingerprint': m['_fingerprint'], 'sigmas': list(NATIVE_SIGMAS),
              'input_shape': shape, 'draft_source': a.draft_source, 'draft_key': a.draft_key,
              'guide_key': a.guide_key, 'tiny_crop': a.tiny_crop,
              'scope': 'Comfy warm-start port; upstream Diffusers/audio/geometry parity unverified'}
    if a.inspect_only:
        print(json.dumps({**report, 'status': 'headers_valid', 'full_hash_verified': False}, indent=2))
        return
    if not a.output:
        ap.error('actual inference requires --output in the isolated experiment directory')
    import torch
    from safetensors.torch import load_file
    from h3_comfy import H3
    torch.npu.set_compile_mode(jit_compile=False)
    d = load_file(str(a.latent))
    draft, guide = d[a.draft_key], d[a.guide_key]
    if a.tiny_crop:
        draft, guide = [z[:, :, :2, :8, :8].contiguous() for z in (draft, guide)]
    if not all(torch.isfinite(z).all() for z in (draft, guide)):
        raise ValueError('nonfinite latent input')
    h3 = H3(pretrained_manifest=a.manifest, adaptation_path=a.adaptation,
            lora_rank=a.rank, sigma_schedule=NATIVE_SIGMAS)
    output, sampling = h3.sample(guide, 'base', hw=[draft.shape[-2] * 16, draft.shape[-1] * 16],
                                 seed=a.seed, warm_start=draft)
    pixels = h3.decode(output['video'])
    if not torch.isfinite(pixels).all():
        raise FloatingPointError('nonfinite decoded output')
    a.output.mkdir(parents=True, exist_ok=True)
    torch.save({'video_latents': output['video'], 'pixels': pixels}, a.output / 'block.pt')
    report.update({'status': 'completed_interface_smoke', 'full_hash_verified': True,
                   'output_shape': list(pixels.shape), 'sampling': sampling, 'model': h3.record})
    if 'target' in d:
        target = d['target'][:, :, :2, :8, :8] if a.tiny_crop else d['target']
        if target.shape == output['video'].shape:
            report['latent_mse_to_target_not_a_pixel_quality_metric'] = float((output['video'].float()-target.float()).square().mean())
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
