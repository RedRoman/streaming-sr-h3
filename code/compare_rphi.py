"""Compare completed matched standalone R_phi development experiments."""
import argparse
import json
from pathlib import Path

from eval_rphi import TASKS, source_summary, sha_file


def assert_matched(a, b):
    for r in (a, b):
        if r.get('schema') != 'rphi.rgb_eval.v1' or not r.get('complete'):
            raise ValueError('both evaluations must complete under rphi.rgb_eval.v1')
        if r.get('scope') != 'full_development_set':
            raise ValueError('engineering subsets cannot serve as full pilot comparisons')
        if not r.get('initial_backbone_sha256') or r['initial_backbone_sha256'] != r['training_contract'].get('initial_backbone_sha256'):
            raise ValueError('missing or inconsistent initial backbone fingerprint')
        if r.get('checkpoint_step') != r['training_contract'].get('steps'):
            raise ValueError('comparison requires the final checkpoint at the declared training budget')
        expected_mode = 'known' if r['model_config']['variant'] == 'known' else 'observations_only'
        if r.get('condition_mode') != expected_mode:
            raise ValueError('condition interference controls are not the main matched arms')
    for key in ('dev_manifest_sha256', 'source_ids', 'frame_stride', 'metric_contract', 'training_contract', 'initial_backbone_sha256'):
        if a.get(key) != b.get(key):
            raise ValueError('comparison contract mismatch: ' + key)
    for key in ('k', 'ch', 'blocks', 'condition_dim'):
        if a['model_config'][key] != b['model_config'][key]:
            raise ValueError('backbone budget mismatch: ' + key)
    for key in ('available', 'alexnet_sha256'):
        if a['lpips'].get(key) != b['lpips'].get(key):
            raise ValueError('LPIPS availability/weights mismatch')
    aa = {row['sample_id']: row for row in a['per_sample']}
    bb = {row['sample_id']: row for row in b['per_sample']}
    if len(aa) != len(a['per_sample']) or len(bb) != len(b['per_sample']) or aa.keys() != bb.keys():
        raise ValueError('sample sets differ or contain duplicates')
    fields = ('source_id', 'source_sha256', 'target_key', 'task', 'target_pixel_sha256',
              'observation_pixel_sha256', 'record_sha256', 'frame_indices')
    for key in aa:
        if any(aa[key][f] != bb[key][f] for f in fields):
            raise ValueError('input/target sample mismatch: ' + key)
        if aa[key]['controls']['near_clean']['seed'] != bb[key]['controls']['near_clean']['seed']:
            raise ValueError('control input seed mismatch: ' + key)
    return aa, bb


def paired(a, b):
    aa, bb = assert_matched(a, b)
    result = {}
    for control in ('task', 'identity', 'near_clean'):
        by_task = {}
        for task in TASKS:
            rows = []
            for key in sorted(aa):
                if aa[key]['task'] != task:
                    continue
                ar = aa[key] if control == 'task' else aa[key]['controls'][control]
                br = bb[key] if control == 'task' else bb[key]['controls'][control]
                rows.append({'source_id': aa[key]['source_id'], 'model': ar['model'], 'baseline': br['model'],
                             'temporal_model': ar['temporal_model'], 'temporal_baseline': br['temporal_model'],
                             'attribution': {'mse_change': ar['model']['mse'] - br['model']['mse']}})
            by_task[task] = source_summary(rows)
        result[control] = by_task
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--known', type=Path, required=True)
    p.add_argument('--blind', type=Path, required=True)
    p.add_argument('--confidence', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    loaded = {name: json.loads(getattr(a, name).read_text()) for name in ('known', 'blind', 'confidence')}
    expected = {'known': 'known', 'blind': 'blind', 'confidence': 'blind_confidence'}
    if any(loaded[k]['model_config']['variant'] != v for k, v in expected.items()):
        raise ValueError('arm labels do not match checkpoint variants')
    result = {'schema': 'rphi.matched_comparison.v1', 'complete': True,
              'scope': '32 training / 8 development source engineering pilot; not frozen-test omni evidence',
              'inputs': {k: {'path': str(getattr(a, k)), 'sha256': sha_file(getattr(a, k))} for k in loaded},
              'blind_minus_known': paired(loaded['blind'], loaded['known']),
              'confidence_minus_blind': paired(loaded['confidence'], loaded['blind']),
              'fidelity_weights_identical_blind_and_confidence':
                  loaded['blind'].get('fidelity_state_sha256') is not None and
                  loaded['blind']['fidelity_state_sha256'] == loaded['confidence'].get('fidelity_state_sha256'),
              'interpretation': 'confidence head is calibration-only and is not expected to improve RGB without a separately tested downstream rule',
              'identity_controls': {k: v['controls']['identity'] for k, v in loaded.items()},
              'confidence_calibration': loaded['confidence']['calibration_per_task']}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'written': str(a.out), 'complete': True,
                      'fidelity_weights_identical': result['fidelity_weights_identical_blind_and_confidence']}))


if __name__ == '__main__':
    main()
