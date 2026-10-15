"""GT correctness and carrier preservation are separate readout audits.

CPU tensor arithmetic only: no VAE/DiT calls, optimization, or model selection.
The private query is exactly the four predeclared corner patches, not every
unobserved pixel. GT is used only to evaluate already-frozen readouts.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from observation_quotient import ObservationQuotient


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', nargs='+', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    q = ObservationQuotient(device='cpu')
    for name in args.roots:
        root = Path(name)
        manifest = json.loads((root/'manifest.json').read_text())
        rows = []
        for source in manifest['sources']:
            item = torch.load(root/'fixed_seed17'/f"{source['id']}_readouts.pt", map_location='cpu', weights_only=False)
            gt = item['target_evaluation_only'].float()
            private = item['private_mask']
            for key, value in item['outputs'].items():
                u = int(key.rsplit('_u', 1)[1])
                base = item['outputs'][f'source_carrier_u{u}'].float()
                value = value.float()
                error = value-gt
                rows.append(dict(source=source['id'], split=source['split'], output=key,
                                 method=key.rsplit('_u', 1)[0], carrier_seed=[101, 202][u],
                                 private_GT_RMSE=float(error[:, :, private].square().mean().sqrt()),
                                 private_GT_HF_RMSE=float(q.null(error)[:, :, private].square().mean().sqrt()),
                                 private_carrier_HF_revision_RMS=float(q.null(value-base)[:, :, private].square().mean().sqrt()),
                                 full_GT_RMSE=float(error.square().mean().sqrt())))
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['source'], row['split'], row['method']].append(row)
        sourcewise = []
        metrics = ('private_GT_RMSE', 'private_GT_HF_RMSE', 'private_carrier_HF_revision_RMS', 'full_GT_RMSE')
        for (source, split, method), group in grouped.items():
            sourcewise.append(dict(source=source, split=split, method=method,
                                   **{k: mean(r[k] for r in group) for k in metrics}))
        groups = defaultdict(list)
        for row in sourcewise:
            groups[row['split'], row['method']].append(row)
        averages = [dict(split=split, method=method, source_count=len(group),
                         **{k: mean(r[k] for r in group) for k in metrics})
                    for (split, method), group in groups.items()]
        report = dict(status='complete', rows=rows, sourcewise=sourcewise, means=averages,
                      actual_D_calls=0, actual_E_calls=0, optimizer_steps=0,
                      scope='Read-only evaluation of frozen outputs. Four-corner private query only; high carrier fidelity is not GT accuracy or verified semantic identity preservation.')
        (root/'readout_truth_audit.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(root=name, status='complete', output_count=len(rows))), flush=True)


if __name__ == '__main__':
    main()
