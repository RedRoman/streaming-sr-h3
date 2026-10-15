"""Join the fixed development pilot and its executed structural controls.

This summary never selects checkpoints, examples, gains, or training budgets.
Decode counts are top-level VAE.decode calls reconstructed from executed loops.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    load = lambda name: json.loads((root / name).read_text())
    trained = load('fixed_seed17/results.json')
    controls = load('strong_controls.json')
    direct = load('direct_readout_controls.json')
    manifest = load('manifest.json')
    splits = {v['id']: v['split'] for v in manifest['sources']}
    rows = []

    def add(source, method, category, metrics, prefix='AB', **extra):
        rows.append(dict(source=source, split=splits[source], method=method,
                         category=category, prefix=prefix, **metrics, **extra))

    for source, methods in trained['evaluation'].items():
        for method, result in methods.items():
            for prefix, metrics in result['prefixes'].items():
                add(source, method, 'trained', metrics, prefix,
                    RGB_path_mse=result['RGB_path_mse'],
                    latent_path_mse=result['latent_path_mse'])
            endpoints = [result['prefixes'][p] for p in ('AB', 'BA')]
            mean = {k: sum(v[k] for v in endpoints) / 2 for k in endpoints[0]}
            add(source, method, 'trained', mean, 'AB_BA_mean',
                RGB_path_mse=result['RGB_path_mse'],
                latent_path_mse=result['latent_path_mse'])
    for source, result in controls['sources'].items():
        add(source, 'initial_X0', 'reference', result['initial'])
        for method, metrics in result['subsets']['AB'].items():
            if isinstance(metrics, dict) and 'psnr_full_db' in metrics:
                add(source, method, 'strong_control', metrics)
        for path, result_ot in result['conditional_OT'].items():
            add(source, 'conditional_heuristic_OT', 'strong_control',
                result_ot['metrics'], path)
        if 'exact_fixed_natural_far_prior' in result:
            prior = result['exact_fixed_natural_far_prior']
            for method in ('exact_mean', 'fixedsource'):
                add(source, 'exact_finite_prior_' + method, 'finite_prior',
                    prior['subsets']['AB'][method])
            for path, item in prior['OT'].items():
                add(source, 'exact_finite_prior_OT', 'finite_prior',
                    item['metrics'], path)
    for source, methods in direct['sources'].items():
        for method, metrics in methods.items():
            add(source, method, 'direct_native', metrics)

    primary = [v for v in rows if
               (v['category'] == 'trained' and v['prefix'] == 'AB_BA_mean') or
               (v['category'] != 'trained' and v['prefix'] == 'AB')]
    groups = defaultdict(list)
    for row in primary:
        groups[(row['split'], row['method'])].append(row)
    nonmetrics = {'source', 'split', 'method', 'category', 'prefix'}
    means = {}
    for (split, method), values in sorted(groups.items()):
        metrics = {k for k in values[0] if k not in nonmetrics}
        means.setdefault(split, {})[method] = {
            k: sum(row[k] for row in values) / len(values) for k in sorted(metrics)}
        means[split][method]['source_count'] = len(values)
    lookup = {(v['source'], v['method']): v for v in primary}
    paired = []
    for source in splits:
        for comparator in ('basic_sequential', 'gru', 'set', 'classical_joint_LR'):
            candidate = lookup[(source, 'connection_loop')]
            baseline = lookup[(source, comparator)]
            paired.append(dict(source=source, split=splits[source],
                               comparator=comparator,
                               loop_minus_comparator={k: candidate[k] - baseline[k]
                                   for k in candidate if k not in nonmetrics and
                                   k in baseline}))

    # Includes completed diagnostics/repeated readouts, not just the fair
    # training comparison. No claim to count internal decoder layer calls.
    decode_calls = {
        'formal_training_4x300x4_endpoints_forward_backward': 4800,
        'formal_evaluation_6sources_x4heads_x4endpoints_forward': 96,
        'smoke_4x2_training_plus_6source_evaluation': 128,
        'initial_native_output_decodes': 8,
        'response_audit_6x4subsets_x2candidates': 48,
        'first_strong_control_readouts': 26,
        'repeated_controls_plus_full24_diagnosis': 74,
        'independent_RGB_export_2val_x4heads_x4endpoints': 32,
        'direct_native_readout_diagnosis_6x4': 24,
    }
    summary = {
        'scope': 'Completed fixed 4-train / 2-val development pilot, one seed17; no confirmation claim.',
        'outcome': 'No independent joint correction/private-preservation benefit. Do not scale or tune this loop head.',
        'means': means,
        'per_source_paired_differences': paired,
        'protocol_checks': trained['protocol_checks'],
        'exact_set_path_mse_all6sources': [
            trained['evaluation'][source]['set']['RGB_path_mse'] for source in splits],
        'cost': {
            'native_H3_samples': 56,
            'official_Euler_calls_per_sample': 49,
            'full_50block_DiT_forward_calls': 2744,
            'DiT_backward_calls': 0,
            'native_VAE_encode_calls': 30,
            'native_VAE_decode_calls_by_stage': decode_calls,
            'native_VAE_decode_calls_total': sum(decode_calls.values()),
            'count_definition': 'Top-level VAE.encode/decode invocations from executed loops, not internal-layer hook counters.',
            'formal_optimizer_steps': 1200,
            'smoke_optimizer_steps_excluded_from_formal_budget': 8,
            'parameters_each_of_four_heads': trained['head_parameters'],
            'formal_training_and_evaluation_seconds': trained['seconds'],
            'formal_peak_allocated_bytes': trained['peak_allocated_bytes'],
            'formal_peak_reserved_bytes': trained['peak_reserved_bytes'],
            'initial8_sampling_batch_seconds': 51.3406,
            'conditional48_sampling_batch_seconds': 236.8976,
            'extra_diagnostic_cost_is_not_part_of_matched_training_budget': True,
        },
        'interpretation': [
            'DeepSets has exactly zero order discrepancy by construction, on all six sources. Small loop reduction versus a path-dependent endpoint head is insufficient.',
            'All trained heads and matched conditional controls share the same X0 channel-PCA8 and latent private complement. These coordinates are computational, not semantically identified.',
            'Removing the arbitrary PCA/private split improves correction and can also reduce decoded private-query change; a frozen latent complement is not guaranteed visual identity protection.',
            'VAE-guide response is at least as strong as actual H3 response here. Direct native output remains weaker than the direct VAE guide and classical inverse.',
            'Exact finite-prior Bayes/MHT/OT claims apply only to two fixed far-only natural atoms with the specified likelihood. Cue-conditioned candidate rescoring is heuristic, not a calibrated posterior.',
            'Private queries are four predeclared 32px corners, 6.25% of pixels. Global query is 50% outside nominal near rectangles and private queries. These are operational regions, not semantic conditional-independence labels or exact camera-support nulls.',
            'Earlier project evaluation already used all BSDS val/test sources. These six sources are development data. Sealed DIV2K confirmation data was not read or evaluated.',
            'All conditional caches use official BF16 sampling. Earlier five-step native guide gradient feasibility had an FP32-vs-BF16 IO discrepancy and is not evidence for this pilot.',
        ],
    }
    perceptual_path = root.parent / 'connection_perceptual/evaluation.json'
    if perceptual_path.exists():
        perceptual = json.loads(perceptual_path.read_text())
        assert perceptual['status'] == 'complete'
        summary['independent_perceptual_evaluation'] = {
            'path': str(perceptual_path),
            'protocol': perceptual['protocol'],
            'LPIPS': perceptual['lpips'],
            'source_means': perceptual['source_means'],
            'by_source': perceptual['by_source'],
            'conclusion': 'No stable perceptual contribution: loop/basic source differences change sign; classical control is substantially stronger.',
        }
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (root / 'paired_differences.csv').open('w', newline='') as stream:
        flat = [dict(source=v['source'], split=v['split'], comparator=v['comparator'],
                     **v['loop_minus_comparator']) for v in paired]
        fields = list(dict.fromkeys(k for row in flat for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({'val': means['val'], 'decode_calls_total': sum(decode_calls.values())}, indent=2))


if __name__ == '__main__':
    main()
