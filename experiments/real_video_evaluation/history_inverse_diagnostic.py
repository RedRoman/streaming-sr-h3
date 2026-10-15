"""Post-hoc operator error audit. HR is evaluation only, never controller input."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from history_inverse import ROOT, DATA, OUT, Observation, BicubicConsistency, load, sha


def main(device):
    protocol = json.loads((OUT / 'protocol.json').read_text())
    op = BicubicConsistency(256, 64, dtype=torch.float32, device=device)
    sources = []
    for case in protocol['cases']:
        folder = OUT / case['id']
        proof = json.loads((folder / 'inference.json').read_text())
        assert sha(folder / 'outputs.pt') == proof['output_sha256']
        outputs = torch.load(folder / 'outputs.pt', weights_only=True, map_location=device)
        lr = torch.stack([load(DATA / p) for p in case['lr_paths']]).to(device)
        # GT is opened only by this separate evaluation program.
        hr = torch.stack([load(DATA / p) for p in case['hr_paths']]).to(device)
        geo = np.load(folder / 'geometry.npz')
        pairs = []
        bins = [{'low': a, 'high': a + .2, 'count': 0, 'truth_sse': 0.,
                 'base_sse': 0., 'inverse_sse': 0.} for a in [0., .2, .4, .6, .8]]
        with torch.inference_mode():
            for index, (current, past) in enumerate(geo['pairs']):
                current, past = int(current), int(past)
                obs = Observation(geo['displacement'][index], geo['confidence'][index], op)
                errors = {key: (obs.forward(value) - lr[past]).square()
                          for key, value in [('truth', hr[current]),
                                             ('base', outputs['vae_dc'][current]),
                                             ('inverse', outputs['inverse_vae_dc'][current])]}
                denominator = 3 * obs.weight.sum().clamp_min(1e-12)
                pairs.append({'current': current, 'past': past,
                              'mean_confidence': float(obs.weight.mean()),
                              **{key + '_weighted_observation_mse':
                                 float((value * obs.weight).sum() / denominator)
                                 for key, value in errors.items()}})
                for b in bins:
                    selected = (obs.weight >= b['low']) & (obs.weight < b['high'])
                    if b['high'] == 1.: selected |= obs.weight == 1.
                    b['count'] += 3 * int(selected.sum())
                    for key, error in errors.items():
                        b[key + '_sse'] += float((error * selected).sum())
            base = outputs['vae_dc'][1:]
            revised = outputs['inverse_vae_dc'][1:]
            truth = hr[1:]
            delta = revised - base
            target = truth - base
            frame_rows = []
            for i in range(21):
                frame_rows.append({'frame': i + 1,
                    'correction_rms': float(delta[i].square().mean().sqrt()),
                    'correction_error_inner_product': float((delta[i] * target[i]).mean()),
                    'correction_energy': float(delta[i].square().mean()),
                    'base_hr_mse': float(target[i].square().mean()),
                    'inverse_hr_mse': float((revised[i] - truth[i]).square().mean())})
            current_lr_drift = float((op.down(delta).square().mean()).sqrt())
        summary = {'case': case['id'], 'source': case['sequence'], 'split': case['split'],
                   'pair_count': len(pairs),
                   'pair_mean': {key: float(np.mean([p[key] for p in pairs]))
                                 for key in pairs[0] if key not in ['current', 'past']},
                   'frame_mean': {key: float(np.mean([p[key] for p in frame_rows]))
                                  for key in frame_rows[0] if key != 'frame'},
                   'current_lr_drift_after_box_dc': current_lr_drift,
                   'confidence_bins': [{**b, **{key + '_mse': b[key + '_sse'] / max(1, b['count'])
                                               for key in ['truth', 'base', 'inverse']}} for b in bins]}
        (folder / 'diagnostic_pairs.json').write_text(json.dumps(pairs, indent=2))
        (folder / 'diagnostic_frames.json').write_text(json.dumps(frame_rows, indent=2))
        (folder / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2))
        sources.append(summary)
        print(json.dumps(summary), flush=True)
    (OUT / 'diagnostic.json').write_text(json.dumps({
        'status': 'complete', 'source_sha256': sha(Path(__file__)),
        'evaluation_only_hr': True, 'fitted_parameters': 0,
        'meaning': 'GT observation residual measures mismatch of estimated warp/downsample/brightness model, not pure flow endpoint error',
        'no_new_inference_or_training': True, 'sources': sources}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--device', default='cpu'); a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    main(a.device)
