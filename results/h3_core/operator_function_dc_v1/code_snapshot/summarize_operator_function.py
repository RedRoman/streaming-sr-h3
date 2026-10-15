"""Source-paired summary of the fixed raw/DC conditional-function pilots.

Reports every source and preserves the distinction between finite-prior Bayes,
cue-conditioned response pushforwards, compression, and identity error.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


HEADS = ('mixed', 'additive', 'joint_set_cnp')
METRICS = ('psnr_full_db', 'private_HF_revision_RMS', 'HF_RMS',
           'hypothesis_LR_RMSE', 'nominal_A0B_LR_RMSE', 'far_LR_RMSE',
           'response_latent_MSE', 'teacher_RGB_RMSE')


def summarize(root):
    root = Path(root)
    result = json.loads((root / 'fixed_seed17/results.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    groups = defaultdict(list)
    for row in result['rows']:
        groups[row['source'], row['split'], row['method'], row['query']].append(row)
    paired = []
    for (source, split, method, query), rows in groups.items():
        paired.append(dict(source=source, split=split, method=method, query=query,
                           **{m: mean(r[m] for r in rows if m in r)
                              for m in METRICS if any(m in r for r in rows)}))
    aggregated = defaultdict(list)
    for row in paired:
        aggregated[row['split'], row['method'], row['query']].append(row)
    means = []
    for (split, method, query), rows in aggregated.items():
        means.append(dict(split=split, method=method, query=query,
                          source_count=len(rows),
                          **{m: mean(r[m] for r in rows if m in r)
                             for m in METRICS if any(m in r for r in rows)}))
    lookup = {(r['source'], r['method'], r['query']): r for r in paired}
    differences = []
    for source in manifest['sources']:
        sid = source['id']
        for query in ('LR_joint_conditioned_function', 'held_composition', 'held_interior'):
            first = lookup[sid, 'mixed', query]
            second = lookup[sid, 'joint_set_cnp', query]
            differences.append(dict(source=sid, split=source['split'], query=query,
                                    **{m + '_mixed_minus_CNP': first[m] - second[m]
                                       for m in METRICS if m in first and m in second}))
    cost = dict(optimizer_steps=result['optimizer_steps'],
                formal_train_and_evaluation_seconds=result['seconds'],
                formal_actual_D=result['calls'],
                new_actual_H3_samples=manifest['actual_new_H3_samples'],
                new_full50_DiT_forwards=manifest['full50_DiT_forwards'],
                inherited_conditional_trajectories=24 if 'raw_version' not in manifest else 0,
                memory_bytes_per_carrier_FP32=344064,
                carrier_count_per_source=2, parameters=result['parameters'])
    # Exact actual D/E calls from completed sampler metadata, not estimated
    # wall-time or the number of repeated RGB frames.
    sampler_results = [json.loads(p.read_text()) for p in root.glob('*/**/sample_result.json')]
    cost['inside_sampler_D_calls'] = sum(r.get('D_calls_inside_sampler', 0) for r in sampler_results)
    cost['inside_sampler_E_calls'] = sum(r.get('E_calls', 0) for r in sampler_results)
    cost['actual_completed_sampler_trajectories_in_this_folder'] = len(sampler_results)
    cost['identity_guard_forwards_separate'] = sum(len(r.get('forward_events', []))
                                                for r in sampler_results if r.get('mode') == 'identity')
    for name in ('assembly', 'rebase'):
        p = root / (name + '.json')
        if p.exists():
            cost[name] = json.loads(p.read_text())
    cost['preparation_E_calls'] = manifest.get('nativeE_calls', 0)
    cost['preparation_D_calls'] = manifest.get('nativeD_calls', 0)
    cost['smoke_runs_excluded_from_formal_initialization'] = []
    for p in sorted(root.glob('smoke*/results.json')):
        smoke = json.loads(p.read_text())
        cost['smoke_runs_excluded_from_formal_initialization'].append(dict(
            directory=str(p.parent), optimizer_steps=3*smoke['config']['steps'],
            actual_D=sum(smoke['calls'].values()),
            scope='Interface only; legacy pre-v2 smoke is not a held-coordinate training result.'))
    diagnostic_path = root/'compression_diagnostic/results.json'
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    if diagnostic:
        cost['fixed_compression_diagnostic'] = dict(actual_D_calls=diagnostic['actual_D_calls'],
                                                   optimizer_steps=0, new_DiT_calls=0,
                                                   seconds=diagnostic['seconds'])
    cost['total_top_level_actual_D_calls_in_this_stage'] = (
        sum(result['calls'].values()) + cost['inside_sampler_D_calls'] +
        cost['preparation_D_calls'] +
        sum(cost.get(n, {}).get('nativeD_calls', 0) for n in ('assembly', 'rebase')) +
        sum(v['actual_D'] for v in cost['smoke_runs_excluded_from_formal_initialization']) +
        (diagnostic['actual_D_calls'] if diagnostic else 0))
    cost['total_top_level_actual_E_calls_in_this_stage'] = cost['inside_sampler_E_calls'] + cost['preparation_E_calls']
    cost['scope'] = ('Top-level native E/D calls. Reused raw center/carrier samples and earlier '
                     'guide construction are reported separately; repeated native22 RGB context '
                     'is one physical far measurement, not 22 independent observations.')
    summary = dict(version=str(root), status=result['status'],
                   aggregation='Average seeds101/202 within source, then equal source weights; no pooling pixels or seeds as independent sources.',
                   fixed_development_sources=manifest['sources'],
                   means=means, sourcewise=paired, mixed_minus_CNP=differences, cost=cost,
                   fixed_compression_diagnostic=diagnostic,
                   cautions=[
                       'Four train/two val development sources; all BSDS val/test were previously evaluated elsewhere in this project.',
                       'No held composition/interior GT, responses or inferred-coordinate losses enter the fixed 300-step training.',
                       'Private HF revision is displacement from each own carrier, not correctness; a larger value is not a benefit.',
                       'Dense cue-conditioned H3/VAE pushforwards are not exact Bayes posteriors. Exact finite-prior joint mean only uses fixed far-only atoms.',
                       'Function coordinates and two sampler seeds do not establish semantic factorization or meaningful natural H3 identity diversity.',
                       'This is post-arrival static conditional-family compression, not demonstrated long streaming reconstruction.'])
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    for filename, rows in [('sourcewise.csv', paired), ('mixed_minus_CNP.csv', differences)]:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (root / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    vals = [r for r in means if r['split'] == 'val' and
            (r['query'] == ('LR_joint_conditioned_function' if r['method'] in HEADS else 'LR_inferred'))]
    text = ['# Fixed conditional-function pilot', '',
            'All three heads completed 300 actual-decoder training steps from the same initialization schedule. '
            'The table averages both carrier seeds within each of the two validation sources, then weights sources equally.', '',
            '| Method | PSNR dB | Private HF revision RMS | Nominal near LR RMSE |',
            '|---|---:|---:|---:|']
    for r in vals:
        text.append(f"| {r['method']} | {r['psnr_full_db']:.6f} | {r['private_HF_revision_RMS']:.8f} | {r['nominal_A0B_LR_RMSE']:.8f} |")
    text += ['', 'Every source and held-query error is retained in `sourcewise.csv`; mixed-minus-CNP paired results are in `mixed_minus_CNP.csv`. '
             'The larger private revision of a method describes a tradeoff and must be read alongside GT error, not called identity preservation.', '',
             'Raw and sampling-inner DC teachers are separate experiments. The training only queries the three fixed support coordinates. '
             'The unseen (+,+) composition and center are evaluation labels; no held finite-difference supervision is used.', '',
             'The exact finite-prior control represents its declared two far-only atoms and five operator coordinates. '
             'The H3/VAE conditional-family pushforwards are predictive readouts; scoring a cue-conditioned response is not exact posterior inference.', '',
             'This development pilot cannot establish a new method, generalization, long memory, or useful H3 identity diversity. '
             'All computation and family/learning controls are recorded in `summary.json`.', '']
    (root / 'FINDINGS.md').write_text('\n'.join(text))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', nargs='+', required=True)
    p.add_argument('--comparison')
    args = p.parse_args()
    summaries = [summarize(root) for root in args.roots]
    if args.comparison and len(summaries) == 2:
        first, second = summaries
        before = {(r['source'], r['method'], r['query']): r for r in first['sourcewise']}
        after = {(r['source'], r['method'], r['query']): r for r in second['sourcewise']}
        rows = []
        for key, r in before.items():
            s = after[key]
            rows.append(dict(source=key[0], split=r['split'], method=key[1], query=key[2],
                             **{m + '_DC_minus_raw': s[m] - r[m] for m in METRICS if m in r and m in s}))
        dest = Path(args.comparison)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'paired_comparison.json').write_text(json.dumps(dict(
            before=first['version'], after=second['version'], rows=rows,
            caveat='Same source/seed/schedule/heads; carriers and conditional response families are regenerated under the declared far-only sampling correction. Neither version selected by val.'), indent=2))
    print(json.dumps([dict(root=s['version'], status=s['status'], cost=s['cost']) for s in summaries]), flush=True)


if __name__ == '__main__':
    main()
