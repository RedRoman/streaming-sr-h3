"""Read-only CPU analysis of already generated true H3 conditional responses.

Candidate noise is genuine H3 sampler noise. This audit does not establish
semantic independent factors or linkage to the previous persistent source.
It never reads HR, runs H3/E/D, or fits a model.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def rms(value):
    return float(value.double().square().mean().sqrt())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    result = {
        'scope': 'Existing 56-sample development cache only; CPU latent audit, no new H3/VAE calls or training.',
        'caution': 'Two candidate noises measure nonlinear cue/noise interaction, not semantic factorization, calibrated posterior uncertainty, or established coupling to old X0 identity.',
        'sources': {},
    }
    for source in json.loads((root / 'manifest.json').read_text())['sources']:
        folder = root / source['id']
        ensemble = torch.load(folder / 'ensemble_control.pt', map_location='cpu', weights_only=False)
        initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=False)
        basis = initial['basis'].double()
        latents = {k: v.double() for k, v in ensemble['latents'].items()}
        guide0 = load_file(str(folder / 'response_empty_seed101/guide.safetensors'))['guide'].double()
        rows = {}
        for subset in ('A', 'B', 'AB'):
            guide = load_file(str(folder / f'response_{subset}_seed101/guide.safetensors'))['guide'].double()
            response = latents[subset] - latents['empty']
            mean = response.mean(0, keepdim=True)
            mixed = response[0:1] - response[1:2]
            projected = torch.einsum('ck,bkthw->bcthw', basis,
                                    torch.einsum('ck,bcthw->bkthw', basis, mixed))
            residual = mean - (guide - guide0)
            rows[subset] = {
                'mean_cue_response_latent_RMS': rms(mean),
                'cue_by_sampler_noise_mixed_latent_RMS': rms(mixed),
                'mixed_to_mean_norm_ratio': float(mixed.norm() / mean.norm().clamp_min(1e-20)),
                'X0_PCA8_fraction_of_mixed_energy': float(projected.square().sum() / mixed.square().sum().clamp_min(1e-30)),
                'mean_H3_response_minus_VAEguide_response_RMS': rms(residual),
                'raw_conditional_candidate_noise_difference_RMS': rms(latents[subset][0] - latents[subset][1]),
                'empty_candidate_noise_difference_RMS': rms(latents['empty'][0] - latents['empty'][1]),
            }
        result['sources'][source['id']] = rows
    result['calls'] = dict(DiT=0, VAE_encode=0, VAE_decode=0, optimizer=0)
    (root / 'mixed_response_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
