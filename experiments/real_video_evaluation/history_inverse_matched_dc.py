"""Match finite output projection budgets; preserve all original solver outputs."""
import argparse
import json
import shutil
from pathlib import Path

import torch

import history_inverse as original

ROOT = original.ROOT
SOURCE = original.OUT
OUT = SOURCE / 'matched_final_projection'


def fix(device):
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    protocol['fairness_repair'] = ('All controls receive the same pre-projection to construct the '
        'current-LR-compatible initialization and final25 projection. The no-history control '
        'previously omitted final projection; finite box projection is not exactly idempotent. '
        'Historical inverse candidates are byte-for-byte unchanged. No parameter or model fitting.')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    op = original.BicubicConsistency(256, 64, device=device)
    records = []
    for case in protocol['cases']:
        source, dest = SOURCE / case['id'], OUT / case['id']
        dest.mkdir(exist_ok=True)
        proof = json.loads((source / 'inference.json').read_text())
        assert original.sha(source / 'outputs.pt') == proof['output_sha256']
        outputs = torch.load(source / 'outputs.pt', map_location='cpu', weights_only=True)
        lr = torch.stack([original.load(original.DATA / p) for p in case['lr_paths']]).to(device)
        for method in ['bicubic_dc', 'vae_dc']:
            outputs[method] = torch.stack([op.project(outputs[method][t].to(device), lr[t], iterations=25)[0].cpu()
                                          for t in range(22)])
        no_history = {method: float((outputs[method][0] - outputs['inverse_' + method][0]).abs().max())
                      for method in ['bicubic_dc', 'vae_dc']}
        assert max(no_history.values()) == 0, no_history
        torch.save(outputs, dest / 'outputs.pt')
        proof.update({'original_output_sha256': proof.pop('output_sha256'),
                      'output_sha256': original.sha(dest / 'outputs.pt'),
                      'repair_source_sha256': original.sha(Path(__file__)),
                      'inverse_candidates_unchanged': True, 'zero_history_final_output_max': no_history,
                      'repair_no_hr_input': True, 'legacy_directory': str(source.relative_to(ROOT))})
        (dest / 'inference.json').write_text(json.dumps(proof, indent=2))
        for filename in ['geometry.npz', 'geometry_provenance.json']:
            shutil.copy2(source / filename, dest / filename)
        records.append({'case': case['id'], 'zero_history_final_output_max': no_history,
                        'output_sha256': proof['output_sha256']})
        print(json.dumps(records[-1]), flush=True)
    (OUT / 'fairness_repair.json').write_text(json.dumps({'passed': True, 'records': records}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--device', default='cpu'); a = p.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fix(a.device)
    original.OUT = OUT
    original.evaluate(a.device)
    import history_inverse_diagnostic as diagnostic
    diagnostic.OUT = OUT
    diagnostic.main(a.device)
