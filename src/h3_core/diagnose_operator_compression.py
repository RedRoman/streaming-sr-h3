"""Frozen interpolation diagnostics; no new optimization or held-query fitting.

A shared channel-PCA8 basis is fitted solely to training-source responses at
the three support coordinates. Its three coefficient fields use the same
24-channel persistent state as the learned heads. Full24 interpolation uses
72 channels and is explicitly a larger-memory diagnostic, not a fair winner.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_function_data import COORDS, cameras_for
from operator_function_model import KINDS, build
from h3_interface import load_frozen_vae
from observation_quotient import ObservationQuotient


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--comfy', required=True)
    p.add_argument('--vae', required=True)
    args = p.parse_args()
    began = time.monotonic()
    root = Path(args.root)
    dest = root / 'compression_diagnostic'
    dest.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((root / 'manifest.json').read_text())
    items = [torch.load(root / s['id'] / 'family.pt', map_location='cpu', weights_only=False)
             for s in manifest['sources']]
    train = [x for x in items if x['source']['split'] == 'train']
    assert len(train) == 4
    bases = {}
    basis_audit = {}
    for key in ('response', 'VAE_response'):
        covariance = torch.zeros(24, 24, dtype=torch.float64)
        count = 0
        for item in train:
            # No held/center responses, no RGB/GT labels, no val data.
            x = item[key][:, :3].double().permute(2, 0, 1, 3, 4, 5).reshape(24, -1)
            covariance += x @ x.T
            count += x.shape[1]
        values, vectors = torch.linalg.eigh(covariance / count)
        bases[key] = vectors[:, -8:].flip(1).float()
        basis_audit[key] = dict(training_sources=[x['source']['id'] for x in train],
                                support_indices=[0, 1, 2], centered=False,
                                retained_energy=float(values[-8:].sum() / values.sum()),
                                covariance_eigenvalues=values.flip(0).tolist())
    torch.save(dict(bases=bases, audit=basis_audit), dest / 'train_support_PCA8.pt')
    sys.path.insert(0, str(Path(args.comfy).resolve()))
    vae, _ = load_frozen_vae(args.vae)
    q = ObservationQuotient()
    rows = []
    learned_fit = []
    family_audit = []
    calls = 0
    checkpoint = torch.load(root/'fixed_seed17/checkpoint.pt', map_location='cpu', weights_only=False)
    heads = {k: build(k).cuda().eval() for k in KINDS}
    for k, h in heads.items():
        h.load_state_dict(checkpoint['heads'][k])
    design_inverse = torch.linalg.inv(torch.tensor([[1., -1., -1.],
                                                    [1., -1., 1.],
                                                    [1., 1., -1.]], device='cuda'))
    def metrics(rgb, item, carrier_rgb):
        gt = item['target'].cuda()
        private = item['private_mask'].cuda()
        return dict(psnr_full_db=float(-10 * torch.log10((rgb-gt).square().mean().clamp_min(1e-12))),
                    private_HF_revision_RMS=float(q.null(rgb-carrier_rgb)[:, :, private].square().mean().sqrt()),
                    nominal_A0B_LR_RMSE=float(torch.stack([
                        (cameras_for((0., 0.), q)[k].observe(rgb)-item['observations'][k].cuda()).square().mean()
                        for k in ('A', 'B')]).mean().sqrt()))
    with torch.no_grad():
        for item in items:
            sid = item['source']['id']
            conditional = item['conditional_late_weights'].double()
            entropy = -(conditional*conditional.clamp_min(1e-300).log()).sum(1)
            family_audit.append(dict(source=sid, split=item['source']['split'],
                                     conditional_operator_weights=conditional.tolist(),
                                     conditional_entropy_nats=entropy.tolist(),
                                     conditional_effective_support=entropy.exp().tolist(),
                                     joint_operator_weights=item['joint_late_weights'].sum(0).tolist(),
                                     far_only_native_seed_pair_RMS=float((item['basez'][0]-item['basez'][1]).square().mean().sqrt()),
                                     far_only_RGB_seed_pair_RMS=float((item['basergb'][0]-item['basergb'][1]).square().mean().sqrt()),
                                     H3_native_response_RMS=float(item['response'].square().mean().sqrt()),
                                     VAE_native_response_RMS=float(item['VAE_response'].square().mean().sqrt())))
            for u in range(2):
                carrier = item['basez'][u:u+1].cuda()
                base = item['basergb'][u:u+1].cuda()
                for kind, head in heads.items():
                    memory = head.memory(item['response'][u:u+1, :3].cuda(), carrier)
                    for j, g in enumerate(COORDS):
                        pred = head.query(memory, carrier, carrier.new_tensor(g))
                        learned_fit.append(dict(source=sid, split=item['source']['split'],
                                                method=kind, carrier_seed=[101, 202][u],
                                                query=('support'+str(j) if j < 3 else ['held_composition', 'held_interior'][j-3]),
                                                native_response_MSE=float((pred-item['response'][u:u+1, j].cuda()).square().mean())))
                for family, key, teacher_key in [('H3', 'response', 'teacher_rgb'),
                                                  ('VAE', 'VAE_response', 'VAE_rgb')]:
                    support = item[key][u, :3].cuda()
                    for mode in ('same_state_PCA8', 'larger_full24'):
                        basis = bases[key].cuda()
                        code = torch.einsum('cr,jcthw->jrthw', basis, support) if mode == 'same_state_PCA8' else support
                        coefficients = torch.einsum('ij,jcthw->icthw', design_inverse, code)
                        grid = []
                        for j, (gx, gy) in enumerate(COORDS):
                            predicted = coefficients[0] + gx*coefficients[1] + gy*coefficients[2]
                            if mode == 'same_state_PCA8':
                                predicted = torch.einsum('cr,rthw->cthw', basis, predicted)
                            rgb = vae.decode((carrier+predicted[None]).half())[:, :, 21].cuda().float()
                            calls += 1
                            rgb = q.bounded(rgb, item['observations']['far'].cuda())
                            grid.append(rgb)
                            rows.append(dict(source=sid, split=item['source']['split'],
                                             method=family+'_'+mode, carrier_seed=[101, 202][u],
                                             query=('support'+str(j) if j < 3 else ['held_composition', 'held_interior'][j-3]),
                                             state_channels=24 if mode == 'same_state_PCA8' else 72,
                                             response_latent_MSE=float((predicted-item[key][u, j].cuda()).square().mean()),
                                             teacher_RGB_RMSE=float((rgb-item[teacher_key][u:u+1, j].cuda()).square().mean().sqrt()),
                                             **metrics(rgb, item, base)))
                        weights = item['conditional_late_weights'][u].cuda()
                        combined = sum(weights[j]*v for j, v in enumerate(grid))
                        rows.append(dict(source=sid, split=item['source']['split'],
                                         method=family+'_'+mode, carrier_seed=[101, 202][u],
                                         query='LR_joint_conditioned_function',
                                         state_channels=24 if mode == 'same_state_PCA8' else 72,
                                         **metrics(combined, item, base)))
            print(json.dumps(dict(event='fixed_compression_diagnostic', source=sid, actual_D=calls)), flush=True)
    report = dict(status='complete', rows=rows, basis_audit=basis_audit,
                  learned_native_response_fit=learned_fit, family_audit=family_audit,
                  actual_D_calls=calls, actual_E_calls=0, new_DiT_calls=0,
                  optimizer_steps=0, seconds=time.monotonic()-began,
                  same_state_shape=[24, 7, 16, 16], larger_state_shape=[72, 7, 16, 16],
                  support_order=[[-1, -1], [-1, 1], [1, -1]],
                  held_composition_weights=[-1, 1, 1],
                  held_composition_is='Affine extrapolation; fourth corner lies outside the support triangle.',
                  held_center_weights=[0, .5, .5],
                  held_center_is='Midpoint of support1/support2, on their triangle boundary.',
                  PCA_global_basis_bytes_per_family=24*8*4,
                  PCA_mean='No mean is fitted or stored; uncentered response covariance, origin preserved.',
                  PCA_per_carrier_coefficients_bytes=24*7*16*16*4,
                  full24_per_carrier_coefficients_bytes=72*7*16*16*4,
                  common_per_carrier_native_latent_bytes=24*7*16*16*4,
                  common_per_carrier_q_weights_bytes=5*8,
                  caution='No held or val coordinate enters fitting. Full24 is a three-times-larger-memory interpolation diagnostic. Both are affine functions inferred from only three support coordinates; neither can freely fit a fourth mixed response.')
    (dest / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(event='fixed_compression_diagnostic_complete', calls=calls,
                          seconds=report['seconds'], basis_audit=basis_audit)), flush=True)


if __name__ == '__main__':
    main()
