"""Predeclared P1 teacher-headroom reduction, with no model or data execution.

The producer must separately authenticate the final1000 teacher, exact dev rows,
and the all/current-only/LOO RGB predictions used to make these records. Passing
this reducer permits a development experiment; it is not quality acceptance.
"""
from collections import defaultdict
import numpy as np

SCHEMA = 'rphi.history_utility_headroom_rule.v1'
RULE = {'neutral_margin': .05 / 255, 'minimum_mean_degraded_mae_gain': .05 / 255,
        'minimum_sources_with_positive_degraded_gain': 4,
        'minimum_positive_tile_fraction': .01, 'minimum_negative_tile_fraction': .01,
        'minimum_sources_per_non_neutral_sign': 3, 'minimum_raw_utility_std': .05 / 255}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def assess_records(records, protocol):
    """Use source-balanced errors and available tiles; reject partial coverage."""
    require(protocol['schema'] == SCHEMA and protocol['rule'] == RULE, 'headroom rule changed')
    source_ids, profiles = protocol['dev_source_ids'], protocol['profile_ids']
    require(len(source_ids) == 6 and len(set(source_ids)) == 6, 'exact six dev sources required')
    require(len(profiles) == 5 and len(set(profiles)) == 5
            and set(profiles[-2:]) == {'identity', 'near_clean'}, 'exact three profiles plus controls required')
    expected = {(s, p, t) for s in source_ids for p in profiles for t in range(96)}
    require(len(records) == len(expected), 'missing/extra dev frame crops')
    seen, errors, values = set(), defaultdict(list), defaultdict(list)
    margin = RULE['neutral_margin']
    for row in records:
        key = (row['source_id'], row['profile_id'], row['frame_index'])
        require(key in expected and key not in seen, 'duplicate/foreign dev row')
        seen.add(key)
        t = key[2]
        require(row['crop_tlhw'] == [192, 368, 96, 96]
                and row['availability'] == [int(t >= 2), int(t >= 1)], 'crop/causal availability changed')
        a, k = float(row['all_history_mae']), float(row['current_only_mae'])
        require(np.isfinite([a, k]).all() and min(a, k) >= 0, 'invalid unclamped RGB MAE')
        raw = np.asarray(row['raw_utility'], dtype=np.float64)
        require(raw.shape == (2, 6, 6) and np.isfinite(raw).all(), 'invalid LOO tile utility')
        valid = np.asarray(row['availability'], dtype=bool)
        require(np.all(raw[~valid] == 0), 'unavailable history has nonzero utility')
        errors[key[:2]].append((a, k))
        values[key[:2]].append(raw[valid].reshape(-1))
    require(seen == expected, 'incomplete dev evaluation')
    source_rows, pooled = [], []
    signs = {'positive': [], 'negative': []}
    strata = []
    for sid in source_ids:
        profile_gains = []
        source_values = []
        for pid in profiles:
            e = np.asarray(errors[sid, pid], dtype=np.float64)
            u = np.concatenate(values[sid, pid])
            gain = float(np.mean(e[:, 1] - e[:, 0]))
            if pid not in ('identity', 'near_clean'):
                profile_gains.append(gain)
            source_values.append(u)
            strata.append({'source_id': sid, 'profile_id': pid,
                           'all_history_mae': float(e[:, 0].mean()),
                           'current_only_mae': float(e[:, 1].mean()),
                           'history_mae_gain': gain, 'valid_tiles': int(u.size),
                           'positive_fraction': float(np.mean(u > margin)),
                           'negative_fraction': float(np.mean(u < -margin))})
        u = np.concatenate(source_values)
        pooled.append(u)
        if np.any(u > margin): signs['positive'].append(sid)
        if np.any(u < -margin): signs['negative'].append(sid)
        source_rows.append({'source_id': sid, 'degraded_mae_gain': float(np.mean(profile_gains))})
    u = np.concatenate(pooled)
    gains = np.asarray([x['degraded_mae_gain'] for x in source_rows])
    mean_gain, positive_sources = float(gains.mean()), int((gains > 0).sum())
    positive, negative, std = float((u > margin).mean()), float((u < -margin).mean()), float(u.std())
    gates = {
        'source_balanced_degraded_gain': mean_gain > RULE['minimum_mean_degraded_mae_gain'],
        'source_support': positive_sources >= RULE['minimum_sources_with_positive_degraded_gain'],
        'beneficial_tile_support': positive >= RULE['minimum_positive_tile_fraction'],
        'harmful_tile_support': negative >= RULE['minimum_negative_tile_fraction'],
        'beneficial_source_support': len(signs['positive']) >= RULE['minimum_sources_per_non_neutral_sign'],
        'harmful_source_support': len(signs['negative']) >= RULE['minimum_sources_per_non_neutral_sign'],
        'nondegenerate_raw_utility': std > RULE['minimum_raw_utility_std'],
    }
    return {'schema': 'rphi.history_utility_headroom_assessment.v1', 'coverage_rows': len(seen),
            'valid_tiles': int(u.size), 'source_balanced_degraded_mae_gain': mean_gain,
            'sources_with_positive_degraded_gain': positive_sources,
            'positive_tile_fraction': positive, 'negative_tile_fraction': negative,
            'neutral_tile_fraction': float((np.abs(u) <= margin).mean()),
            'raw_utility_std': std, 'non_neutral_source_ids': signs,
            'per_source': source_rows, 'per_source_profile': strata, 'gates': gates,
            'headroom_rule_passed': all(gates.values()),
            'student_training_admitted': False,
            'interpretation': 'Operational development headroom only; producer/source/freeze/cache/shuffle review remains required.'}
