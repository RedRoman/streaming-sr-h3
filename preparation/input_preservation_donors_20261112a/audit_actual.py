"""One bounded independent stdlib audit; never imports the donor producer.

Reads only the authenticated student schedule and the new map/manifest.
No pixel/label/model files, schedule regeneration, training, or retries.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path('/cache/zihaohe/omni_openh3_20261111')
SCHEDULE = 'runs/history_utility_schedules_20261112a/output/student_schedule.json'
RUN = 'runs/input_preservation_donors_20261112a'
SCHEDULE_SHA = 'b5e6d4c6031d79d5b9457c98fe215d49e393e335994559597498176f4e666f08'
OUTER_SHA = 'adda5689d88417b4141ca8f632a7b03515a829b18266057312569dea3298a100'
NUMERIC_SHA = '17c7ca57110c4006b2a0f41986dac58daa9945bff7e113265bb8b66ffb3be5cc'
MAP_FILE_SHA = 'fdb872a4e1b20c07b67af2c7b40a322f6b968ea7efcac9b2a96239c8898a2d6a'
MAP_VALUE_SHA = '93351eb1d63955431c8e3546f42c89f5d3933da53e6636ca89765f872b15a2bf'
COHORT_SHA = 'c4c3a1c044fa13e6af93c03259ca0e52ea18531ddb6a2c55bbd19abff81d44e1'
DATASET_SHA = '3e075085a2e846e10bb9edf5775b2df00fa32fad3f04d4135e49713b2d6f3e87'
PRODUCER_SHA = 'a860e7a519275f9ed650c4ca988e1cd478717b9d56d2378319f4b85a0451c2bd'
PAYLOAD_SHA = '59ec0fbebdc91eba294e176164755e9689feabba166dac042ee43fc3cbe33995'
PROFILES = ('native_static_blur_noise_h264', 'native_smooth_blur_noise_h264',
            'native_abrupt_blur_noise_h264')
RESOURCES = {'CPU_cores': 1, 'address_space_bytes': 2 << 30, 'wall_seconds': 120}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def semantic(value, field, expected=None):
    actual = digest({k: v for k, v in value.items() if k != field})
    require(value[field] == actual, field + ' semantic SHA mismatch')
    if expected is not None:
        require(actual == expected, field + ' differs from fixed identity')
    return actual


def read_bound(path, limit, expected_sha=None, expected_bytes=None):
    size = path.stat().st_size
    require(0 < size <= limit, 'file size out of bounds: ' + str(path))
    if expected_bytes is not None:
        require(size == expected_bytes, 'fixed file byte size changed')
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if expected_sha is not None:
        require(actual == expected_sha, 'fixed file SHA changed: ' + str(path))
    return json.loads(data), {'path': str(path), 'bytes': len(data), 'sha256': actual}


def expected_phase(profile, frame):
    if profile in ('identity', 'near_clean'):
        return profile
    require(profile in PROFILES, 'unrecognized profile')
    if profile == PROFILES[0]:
        return 'static'
    if profile == PROFILES[2]:
        return 'abrupt-pre' if frame < 48 else 'abrupt-post'
    return ('low', 'ramp-up', 'high', 'ramp-down')[frame // 24]


def audit_values(schedule, manifest, mapping):
    require(schedule['schema'] == 'openh3.history_utility_revised_schedule.v1', 'outer schema')
    for obj in (schedule, schedule['numeric_schedule']):
        require(obj['role'] == 'student' and obj['seed'] == 20261111 and
                obj['steps'] == 1000 and obj['batch_size'] == 16, 'schedule recipe')
    require(schedule['numeric_schedule']['schema'] == 'openh3.history_utility_schedule.v1', 'numeric schema')
    semantic(schedule, 'schedule_sha256', OUTER_SHA)
    numeric = schedule['numeric_schedule']
    semantic(numeric, 'schedule_sha256', NUMERIC_SHA)
    require(schedule['cohort_sha256'] == COHORT_SHA and schedule['dataset_sha256'] == DATASET_SHA,
            'cohort/dataset binding')
    require(schedule['legacy_research_ready'] is False and schedule['revised_cohort_ready'] is True and
            schedule['training_admitted'] is False, 'schedule readiness changed')
    require(manifest['schema'] == 'rphi.input_preservation_donor_manifest.v1' and
            manifest['complete'] is True and manifest['donor_ready'] is True and
            manifest['both_head_training_arms_blocked'] is False, 'producer is not successful')
    require(manifest['source_sha256'] == PRODUCER_SHA and manifest['execution_payload_sha256'] == PAYLOAD_SHA,
            'producer/source identity')
    require(manifest['original_history_headroom_remains_failed'] is True and
            manifest['student_training_admitted'] is False, 'scientific/admission scope changed')
    for key in ('label_values_read', 'GT_values_read', 'schedule_regenerated'):
        require(manifest[key] is False, 'producer scope flag: ' + key)
    require(manifest['model_calls'] == manifest['source_NPZ_reads'] == 0, 'unexpected model/pixel activity')
    require(manifest['rows'] == 16000 and manifest['train_sources'] == 19 and manifest['strata'] == 36 and
            manifest['impossible_strata'] == [], 'producer totals')
    require(manifest['cohort_sha256'] == COHORT_SHA and manifest['dataset_sha256'] == DATASET_SHA and
            manifest['student_schedule_sha256'] == OUTER_SHA and manifest['numeric_schedule_sha256'] == NUMERIC_SHA,
            'manifest schedule binding')
    require(manifest['input_files']['student_schedule'] == {'path': str(ROOT / SCHEDULE), 'sha256': SCHEDULE_SHA},
            'manifest input path/SHA binding')
    require(mapping['schema'] == 'rphi.input_preservation_whole_benefit_map_donors.v1' and
            mapping['seed'] == 20261112, 'map schema/seed')
    semantic(mapping, 'donor_map_sha256', MAP_VALUE_SHA)
    require(mapping['student_schedule_sha256'] == OUTER_SHA and mapping['student_schedule_file_sha256'] == SCHEDULE_SHA,
            'map schedule binding')
    require(mapping['donor_ready'] is True and mapping['both_head_training_arms_blocked'] is False and
            mapping['impossible_strata'] == [] and mapping['no_resampling_or_coarsening'] is True,
            'partial/infeasible map cannot pass')
    require(mapping['stratum_fields'] == ['profile_id', 'severity_phase', 'effective_availability[2]'], 'wrong strata')
    require(mapping['label_values_read'] is False and mapping['GT_values_read'] is False and mapping['model_calls'] == 0,
            'mapping scope')
    rows = numeric['rows']
    require(len(rows) == 16000 and mapping['total_rows'] == mapping['mapped_rows'] == len(rows), 'row coverage')
    units, strata, source_groups, source_hashes = [], defaultdict(list), {}, {}
    for index, row in enumerate(rows):
        require(type(row['row_index']) is int and row['row_index'] == index, 'noncanonical row index')
        semantic(row, 'row_sha256')
        sid, gid = row['source_id'], row['source_group_id']
        require(row['split'] == 'train' and isinstance(sid, str) and sid and isinstance(gid, str) and gid, 'train/source/group')
        require(sid not in source_groups or source_groups[sid] == gid, 'multiple groups for source')
        source_groups[sid] = gid
        require(sid not in source_hashes or source_hashes[sid] == row['source_sha256'], 'source SHA inconsistency')
        require(len(row['source_sha256']) == 64 and all(c in '0123456789abcdef' for c in row['source_sha256']), 'invalid source SHA')
        source_hashes[sid] = row['source_sha256']
        frame = row['frame_index']
        require(type(frame) is int and 0 <= frame < 96, 'frame bounds')
        available = [int(frame >= 2), int(frame >= 1)]
        require(row['true_availability'] == available and row['history_indices'] == [max(0, frame-2), max(0, frame-1), frame],
                'causal availability/history')
        external = row['external_mask']
        require(type(external) is list and len(external) == 2 and
                all(type(x) is int and x in (0, 1) for x in external), 'external availability')
        require(row['teacher_mask'] is None and row['severity_phase'] == expected_phase(row['profile_id'], frame),
                'student role/phase')
        descriptor = [row['profile_id'], row['severity_phase'], [available[j]*external[j] for j in (0, 1)]]
        unit = {'row_index': index, 'row_sha256': row['row_sha256'], 'source_id': sid,
                'source_group_id': gid, 'stratum': descriptor}
        units.append(unit)
        strata[canonical(descriptor)].append(index)
    require(len(source_groups) == 19 and len(strata) == 36, 'actual source/stratum counts')
    expected_batch = {**{p: 4 for p in PROFILES}, 'identity': 2, 'near_clean': 2}
    for start in range(0, 16000, 16):
        require(Counter(row['profile_id'] for row in rows[start:start+16]) == expected_batch, 'batch12+2+2 changed')
    entries = mapping['entries']
    require(len(entries) == 16000, 'entry coverage')
    donors, recipients, pair = [], [], {}
    for entry in entries:
        ri, di = entry['recipient']['row_index'], entry['donor']['row_index']
        require(type(ri) is int and type(di) is int and 0 <= ri < 16000 and 0 <= di < 16000, 'mapping index')
        r, d = units[ri], units[di]
        require(entry['recipient'] == r and entry['donor'] == d, 'mapping metadata not exact authenticated row')
        require(ri != di and r['source_id'] != d['source_id'] and r['source_group_id'] != d['source_group_id'],
                'self/same-source/same-group donor')
        require(r['stratum'] == d['stratum'], 'cross-stratum donor')
        recipients.append(ri); donors.append(di); pair[ri] = di
    require(recipients == list(range(16000)) and sorted(donors) == list(range(16000)), 'not a complete ordered bijection')
    summaries, audited_strata = [], []
    for key, indices in sorted(strata.items()):
        group_counts = Counter(units[i]['source_group_id'] for i in indices)
        largest = max(group_counts.values())
        require(largest * 2 <= len(indices), 'actual infeasible stratum')
        summaries.append({'stratum': json.loads(key), 'rows': len(indices), 'source_groups': len(group_counts),
                          'maximum_group_rows': largest, 'group_row_counts': dict(sorted(group_counts.items())), 'feasible': True})
        chosen = [pair[i] for i in indices]
        require(sorted(chosen) == sorted(indices), 'stratum donor frequencies differ')
        rc = Counter(units[i]['source_id'] for i in indices)
        dc = Counter(units[i]['source_id'] for i in chosen)
        dg = Counter(units[i]['source_group_id'] for i in chosen)
        require(rc == dc and group_counts == dg, 'source/group label distribution not preserved')
        # Independently reconstruct the predeclared seed-keyed group rotation.
        ordered = []
        for group in sorted(group_counts, key=lambda g: (digest([20261112, key, 'group', g]), g)):
            ordered.extend(sorted((i for i in indices if units[i]['source_group_id'] == group),
                                  key=lambda i: (digest([20261112, key, 'unit', i]), i)))
        require(all(pair[ri] == ordered[(j+largest) % len(ordered)] for j, ri in enumerate(ordered)),
                'declared seeded rotation differs')
        audited_strata.append({'stratum': json.loads(key), 'rows': len(indices),
                              'source_counts': dict(sorted(rc.items())), 'group_counts': dict(sorted(group_counts.items())),
                              'donor_frequency_equal': True, 'source_frequency_equal': True,
                              'group_frequency_equal': True, 'seeded_rotation_equal': True})
    require(mapping['strata'] == summaries, 'reported strata differ from actual rows')
    absent = {i for i, unit in enumerate(units) if unit['stratum'][2] == [0, 0]}
    require(len(absent) == mapping['both_absent_rows'] == manifest['both_absent_rows'] == 1815 and
            mapping['includes_both_absent_rows'] is True, 'both-absent rows missing')
    require({pair[i] for i in absent} == absent, 'both-absent donor coverage')
    return {'actual_rows': 16000, 'actual_source_count': 19, 'actual_strata': 36,
            'actual_both_absent_rows': len(absent), 'row_semantic_SHA_verified': 16000,
            'all_recipients_and_donors_exactly_once': True, 'nonself_cross_source_AND_group': True,
            'profile_phase_effective_availability_exact': True, 'all_strata_frequency_preserved': True,
            'declared_seeded_rotation_verified': True, 'source_groups': dict(sorted(source_groups.items())),
            'source_sha256': dict(sorted(source_hashes.items())), 'strata': audited_strata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    require(args.root == ROOT, 'this audit is bound to the admitted actual namespace')
    require(sys.platform.startswith('linux'), 'Linux resource enforcement required')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    started = time.monotonic()
    def deadline(*_):
        raise TimeoutError('independent metadata audit120seconds exceeded')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    output = ROOT / RUN / 'independent_audit.json'
    require(not output.exists() and not output.is_symlink(), 'fresh audit output only; no retry/overwrite')
    attempt = ROOT / RUN / 'independent_audit_attempt.json'
    with attempt.open('x') as handle:
        json.dump({'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'pid': os.getpid(), 'source_sha256': file_sha(Path(__file__)),
                   'CPU_affinity': sorted(os.sched_getaffinity(0)), 'resources': RESOURCES}, handle, indent=2)
    report = {'schema': 'rphi.input_preservation_donor_independent_actual_audit.v1', 'complete': False,
              'donor_ready': False, 'student_training_admitted': False, 'original_history_headroom_remains_failed': True,
              'generator_imported': False, 'model_calls': 0, 'source_NPZ_reads': 0, 'GT_values_read': False,
              'label_values_read': False, 'schedule_regenerated': False, 'unique_data_JSON_files_read': 3,
              'resources': RESOURCES, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'inputs': {}}
    try:
        schedule, sb = read_bound(ROOT / SCHEDULE, 16 << 20, SCHEDULE_SHA, 14924552)
        manifest, mb = read_bound(ROOT / RUN / 'manifest.json', 1 << 20)
        mapping, db = read_bound(ROOT / RUN / 'input_preservation_donor_map.json', 16 << 20, MAP_FILE_SHA, 7745074)
        report['inputs'] = {'schedule': sb, 'manifest': mb, 'mapping': db}
        require(manifest['donor_map'] == {'path': db['path'], 'bytes': db['bytes'],
                'file_sha256': db['sha256'], 'donor_map_sha256': MAP_VALUE_SHA}, 'manifest/map binding')
        report['validation'] = audit_values(schedule, manifest, mapping)
        require(all(file_sha(Path(b['path'])) == b['sha256'] for b in report['inputs'].values()),
                'input changed during one audit')
        require(not {'torch', 'numpy', 'torch_npu', 'prepare_input_preservation_donors'}.intersection(sys.modules),
                'forbidden import')
        report.update(complete=True, donor_ready=True, status='independently_verified_metadata_bijection_no_training_admission',
                      input_bytes_unchanged=True, terminal_integrity_rehashes=3)
    except BaseException as exc:
        report.update(complete=False, donor_ready=False, status='failed_preserve_evidence_no_retry',
                      error={'type': type(exc).__name__, 'message': str(exc)})
    report.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  audit_source_sha256=file_sha(Path(__file__)))
    with output.open('x') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')
    signal.alarm(0)
    print(canonical({'path': str(output), 'sha256': file_sha(output), 'complete': report['complete'],
                     'elapsed_seconds': report['elapsed_seconds'], 'error': report.get('error')}), flush=True)
    raise SystemExit(0 if report['complete'] else 1)


if __name__ == '__main__':
    main()
