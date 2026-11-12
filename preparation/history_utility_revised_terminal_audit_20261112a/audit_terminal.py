"""One CPU terminal dataset validation; never import models or decode video."""
import argparse
from collections import Counter
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import time

OWN = Path(__file__).resolve().parent
FORBIDDEN = {'torch', 'torch_npu', 'comfy', 'comfy_kitchen', 'h3_comfy', 'rphi_model'}


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError('model/backend forbidden in terminal data audit: ' + fullname)


sys.meta_path.insert(0, NoModels())
sys.path.insert(0, str(OWN / 'code'))
import history_utility_data as D
import history_utility_revised_data as R

WRAPPER_SHA = '3f48214e3df626f988a92e78349a8929dea01414b831566fda0dd793ac1937d1'
COHORT_SHA = 'c4c3a1c044fa13e6af93c03259ca0e52ea18531ddb6a2c55bbd19abff81d44e1'
REUSE = {'4EKWTiWjPTQ', '6UZVovkR3IU', '6rJrrfRO3cA'}
SCHEMA = 'openh3.history_utility_revised_actual_terminal_audit.v1'


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': D.file_sha(path)}


def checked_json(ref):
    D.require(Path(ref['path']).is_absolute(), 'absolute pinned input required')
    D.require(D.file_sha(ref['path']) == ref['sha256'], 'input SHA differs: ' + ref['path'])
    return D._json(ref['path'])


def validate_terminal(status, report, contract, config, actual_report_sha):
    D.require(status['schema'] == 'openh3.history_utility_cpu_supervision.v1'
              and status['state'] == 'completed' and status['complete'] is True
              and status['returncode'] == 0 and status['report_sha256'] == actual_report_sha,
              'producer supervisor did not complete these exact report bytes')
    D.require(status['freeze_sha256'] == config['producer_freeze']['sha256']
              and status['admission_sha256'] == config['producer_admission']['sha256']
              and status['owned_group_only'] is True and status['wall_seconds'] == 3600
              and status['grace_seconds'] == 30 and status['memory_mib'] == 768,
              'producer supervision identity/resource contract changed')
    D.require(0 <= status['elapsed_seconds'] <= 3600
              and 0 <= status['peak_owned_group_rss_bytes'] <= 768 << 20
              and len(status['cpu_affinity']) == 1 and type(status['cpu_affinity'][0]) is int,
              'actual producer elapsed/RSS/CPU measurements exceed declared budget')
    D.require(report['schema'] == 'openh3.history_utility_revised_materialization.v1'
              and report['complete'] is True
              and report['status'] == 'materialized_revised_primary_pending_dataset_validation',
              'producer incomplete; no dataset promotion permitted')
    D.require(report['execution_contract'] == config['producer_contract']
              and report['cohort'] == contract['files']['cohort'], 'producer identity changed')
    D.require(report['new_target_decodes'] == 0 and report['new_codec_passes_completed'] == 132
              and report['model_calls'] == report['NPU_calls'] == 0
              and report['legacy_research_ready'] is False
              and report['revised_dataset_validation_pending'] is True
              and report['training_admitted'] is False, 'producer operation/readiness contract changed')
    D.require(report['import_guard'] == {'forbidden_modules': [], 'forbidden_import_attempts': []},
              'producer model/backend guard did not pass')
    selected = contract['selected_source_ids']
    D.require({s: len(selected[s]) for s in ('train', 'dev')} == {'train': 19, 'dev': 6},
              'this frozen audit is exactly19train6dev')
    rows = report['source_results']
    all_ids = set(selected['train'] + selected['dev'])
    D.require(len(rows) == 25 and {r['source_id'] for r in rows} == all_ids,
              'missing/duplicate/extra source result')
    D.require(set(report['receipt_indices']) == {'train', 'dev'}, 'index split changed')
    by_id = {}
    for row in rows:
        sid = row['source_id']; reused = sid in REUSE
        D.require(row['mode'] == ('reused_complete' if reused else 'new_materialization')
                  and row['new_target_decodes'] == 0
                  and row['new_codec_passes'] == (0 if reused else 6), 'source accounting differs')
        D.require(('revised_authority' in row['entry']) == reused, 'reuse/inline authority namespace differs')
        D.require(row['verification'] == {'old_loader_actual_files_pixels_metadata_verified': 4,
                                         'legacy_research_ready': False}, 'producer source validation missing')
        by_id[sid] = row
    for split in ('train', 'dev'):
        D.require(report['receipt_indices'][split] == [by_id[s]['entry'] for s in selected[split]],
                  'split entries do not bind ordered exact source results')
    return by_id


def schedule_summary(schedule, data):
    """Independent row-contract checks; no sample() or model forward is called."""
    role = schedule['role']; numeric = schedule['numeric_schedule']
    D.require(role in D.SEEDS and schedule['schema'] == 'openh3.history_utility_revised_schedule.v1'
              and schedule['steps'] == 1000 and schedule['batch_size'] == 16
              and schedule['seed'] == D.SEEDS[role]
              and schedule['dataset_sha256'] == data.dataset_sha256
              and schedule['cohort_sha256'] == data.cohort_sha256
              and schedule['legacy_research_ready'] is False
              and schedule['revised_cohort_ready'] is True and schedule['training_admitted'] is False,
              'revised schedule header changed')
    D.require(schedule['schedule_sha256'] == D.value_sha({k: v for k, v in schedule.items() if k != 'schedule_sha256'}),
              'revised schedule digest differs')
    D.require(numeric['schema'] == 'openh3.history_utility_schedule.v1'
              and numeric['role'] == role and numeric['seed'] == D.SEEDS[role]
              and numeric['steps'] == 1000 and numeric['batch_size'] == 16
              and numeric['scope'] == 'engineering' and numeric['research_ready'] is False
              and numeric['dataset_sha256'] == data._numeric.dataset_sha256,
              'underlying frozen numeric schedule changed')
    D.require(numeric['schedule_sha256'] == D.value_sha({k: v for k, v in numeric.items() if k != 'schedule_sha256'}),
              'numeric schedule digest differs')
    rows = numeric['rows']; D.require(len(rows) == 16000, 'not1000x16rows')
    family = {**{p: 4 for p in D.PROFILES}, 'identity': 2, 'near_clean': 2}
    source_counts, masks = Counter(), Counter()
    for start in range(0, len(rows), 16):
        D.require(Counter(r['profile_id'] for r in rows[start:start + 16]) == family,
                  'per-step profile/control balance changed')
    for i, row in enumerate(rows):
        sid, pid, frame = row['source_id'], row['profile_id'], row['frame_index']
        D.require(sid in data.source_ids and row['split'] == 'train', 'schedule crosses selected train boundary')
        D.require(all(row[k] == data.registry[sid][k] for k in ('source_id', 'source_sha256', 'source_group_id', 'split')),
                  'schedule source binding differs')
        receipt = data._numeric.sources[sid]['receipt']; control = pid if pid in ('identity', 'near_clean') else 'original'
        D.require(row['row_index'] == i and row['control'] == control
                  and row['dataset_sha256'] == data._numeric.dataset_sha256
                  and row['window_key'] == receipt['window_key']
                  and row['target_pixel_sha256'] == receipt['target']['pixel_sha256'], 'schedule target/control differs')
        D.require(row['history_indices'] == D.history_indices(frame)
                  and row['true_availability'] == D.true_availability(frame), 'noncausal history/availability')
        top, left, h, w = row['crop_tlhw']
        D.require(all(type(x) is int for x in (top, left, h, w)) and (h, w) == (96, 96)
                  and top % 4 == left % 4 == 0 and 0 <= top <= 384 and 0 <= left <= 736,
                  'crop geometry/alignment changed')
        ext = row['external_mask']
        D.require(ext in ([0, 0], [0, 1], [1, 0], [1, 1]), 'bad availability mask')
        if role == 'student': D.require(row['teacher_mask'] is None, 'teacher metadata enters student schedule')
        else:
            D.require(ext == [1, 1] and row['teacher_mask']['kind'] in
                      ('all_available', 'oldest_missing', 'recent_missing', 'both_missing', 'local_binary', 'local_soft'),
                      'teacher mask family changed')
        expected_obs = (receipt['observations'][pid]['artifact']['pixel_sha256'] if control == 'original'
                        else receipt['target']['pixel_sha256'] if control == 'identity'
                        else data._numeric._control_sha[sid])
        D.require(row['observation_pixel_sha256'] == expected_obs, 'observation/control hash differs')
        expected_recipe = ({'recipe': D.NEAR_RECIPE, 'frame_seeds': [D.near_clean_seed(row['source_sha256'], row['window_key'], t)
                            for t in row['history_indices']]} if control == 'near_clean' else {'recipe': control, 'frame_seeds': []})
        D.require(row['control_recipe_and_seed'] == expected_recipe, 'control RNG changed')
        D.require(row['row_sha256'] == D.value_sha({k: v for k, v in row.items() if k != 'row_sha256'}), 'row hash differs')
        source_counts[sid] += 1; masks[str(ext)] += 1
    D.require(set(source_counts) == set(data.source_ids), 'schedule misses selected train source')
    return {'role': role, 'seed': D.SEEDS[role], 'steps': 1000, 'batch_size': 16, 'rows': len(rows),
            'schedule_sha256': schedule['schedule_sha256'], 'numeric_schedule_sha256': numeric['schedule_sha256'],
            'source_counts': dict(source_counts), 'external_mask_counts': dict(masks),
            'geometry_contract': {'observed': [16, 3, 3, 96, 96], 'availability': [16, 2],
                'target_supervision_only': [16, 3, 96, 96], 'teacher_weights_if_teacher': [16, 2, 96, 96]},
            'geometry_scope': 'frozen loader and verified row geometry; no actual batch tensor or model constructed',
            'legacy_research_ready': False, 'training_admitted': False}


def dataset_summary(data, expected_contract):
    D.require(data.research_ready is False and data._numeric.research_ready is False
              and data.revised_cohort_ready is True and data.binding['training_admitted'] is False,
              'incorrect readiness namespace')
    rows = []
    for sid in data.source_ids:
        source = data._numeric.sources[sid]; auth = data.authorities[sid]
        D.require(auth['execution_authority']['execution_contract'] == expected_contract,
                  'source authority uses another materialization execution')
        if auth['kind'] == 'new_materialization':
            receipt = source['receipt']; execution = D._json(expected_contract['path'])
            D.require(receipt['producer_sha256'] == execution['files']['entry_source']['sha256']
                      and receipt['materializer_sha256'] == execution['files']['numeric_source']['sha256']
                      and receipt['plan_file_sha256'] == execution['files']['plan']['sha256']
                      and receipt['software_sha256'] == execution['software_sha256'],
                      'new receipt source/software implementation changed')
            D.require(receipt['new_codec_passes_started'] == receipt['new_codec_passes_completed'] == 6
                      and receipt['new_target_decodes'] == receipt['model_calls'] == receipt['npu_calls'] == 0
                      and receipt['metadata_bridge']['target_pixels_unchanged'] is True,
                      'new receipt measured operations/target bridge differ')
        artifacts = {}
        for name, artifact in source['artifacts'].items():
            path = source['root'] / (name + '.npz') if source['root'] else Path(artifact['path'])
            artifacts[name] = {**artifact, 'actual_path': str(path),
                               'actual_file_pixel_metadata_verified_by_frozen_loader': True}
        rows.append({'source': data.registry[sid], 'receipt_sha256': auth['receipt_sha256'],
                     'mode': auth['kind'], 'window_key': source['receipt']['window_key'],
                     'source_authority': auth, 'artifacts': artifacts})
    return {'split': data.split, 'dataset_sha256': data.dataset_sha256, 'binding': data.binding,
            'source_count': len(rows), 'actual_npz_count': 4 * len(rows), 'sources': rows}


def audit(config, status_sha, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    report = {'schema': SCHEMA, 'complete': False, 'status': 'started', 'revised_pair_ready': False,
              'training_admitted': False, 'legacy_research_ready': False, 'model_calls': 0,
              'training_steps': 0, 'NPU_calls': 0, 'source_MP4_reads': 0, 'decode_calls': 0,
              'actual_batch_constructions': 0, 'quality_claim': 'none', 'phases': []}
    started = time.monotonic()
    def phase(name):
        report['phases'].append({'name': name, 'elapsed_seconds': time.monotonic() - started})
        write(out / 'audit.json', report); print(name, flush=True)
    try:
        phase('verify_frozen_metadata')
        D.require(D.file_sha(R.__file__) == WRAPPER_SHA and D.file_sha(D.__file__) == R.LOADER_SHA,
                  'frozen wrapper/loader changed')
        status_path = Path(config['producer_status']); status = checked_json({'path': str(status_path), 'sha256': status_sha})
        complete_path = Path(config['producer_run']) / 'complete.json'
        complete_ref = binding(complete_path); complete = checked_json(complete_ref)
        contract = checked_json(config['producer_contract']); freeze = checked_json(config['producer_freeze'])
        admission = checked_json(config['producer_admission'])
        D.require(admission['approved'] is True
                  and admission['contract_sha256'] == config['producer_contract']['sha256']
                  and admission['freeze_sha256'] == config['producer_freeze']['sha256']
                  and admission['entry_sha256'] == contract['files']['entry_source']['sha256'], 'producer admission differs')
        D.require(freeze['run'] == config['producer_run'] and freeze['supervisor'] + '/status.json' == str(status_path)
                  and freeze['execution_contract'] == config['producer_contract']['path'], 'producer freeze paths differ')
        D.require(contract['files']['cohort']['sha256'] == COHORT_SHA, 'cohort changed')
        for key in ('cohort', 'plan', 'train_manifest', 'dev_manifest', 'loader_source', 'revised_loader_source',
                    'numerical_window_contract', 'entry_source', 'numeric_source'):
            ref = contract['files'][key]
            D.require(D.file_sha(ref['path']) == ref['sha256'], 'producer used source/metadata pin changed: ' + key)
        validate_terminal(status, complete, contract, config, complete_ref['sha256'])
        report['producer'] = {'status': {'path': str(status_path), 'sha256': status_sha},
                              'complete': complete_ref, 'contract': config['producer_contract'], 'freeze': config['producer_freeze']}
        datasets = {}
        for split in ('train', 'dev'):
            phase('validate_' + split + '_actual_NPZ_once')
            index_path = Path(config['producer_run']) / (split + '_receipt_index.json')
            index = D._json(index_path)
            D.require(index['schema'] == 'openh3.history_utility_revised_receipt_index.v1'
                      and index['cohort'] == contract['files']['cohort']
                      and index['receipts'] == complete['receipt_indices'][split], 'index file differs from terminal report')
            data = R.RevisedHistoryUtilityDataset(index['receipts'], cohort=contract['files']['cohort'],
                plan=contract['files']['plan']['path'], train_manifest=contract['files']['train_manifest']['path'],
                dev_manifest=contract['files']['dev_manifest']['path'], split=split,
                evidence_root=contract['evidence_root'], cache_bytes=0)
            datasets[split] = data
            report.setdefault('datasets', {})[split] = dataset_summary(data, config['producer_contract'])
            report['datasets'][split]['receipt_index'] = binding(index_path)
        phase('validate_revised_pair')
        pair = R.assert_revised_pair(datasets['train'], datasets['dev'])
        D.require(pair['train_sources'] == 19 and pair['dev_sources'] == 6, 'selected counts changed')
        modes = Counter(a['kind'] for data in datasets.values() for a in data.authorities.values())
        D.require(modes == {'new_materialization': 22, 'reused_complete': 3}, '22new3reuse changed')
        report['pair'] = pair; report['authority_counts'] = dict(modes)
        report['actual_npz_files_pixel_metadata_verified'] = 100
        report['read_accounting'] = {'unique_NPZ_files_verified': 100,
            'target_NPZ_additional_reads_for_near_clean_hash': 19,
            'near_clean_full_frames_generated_and_hashed': 19 * 96,
            'near_clean_numerical_algorithm': D.NEAR_RECIPE,
            'scope': 'frozen loader constructor verifies100 unique NPZ; teacher schedule computes19 full target control hashes; student reuses the control-SHA cache; new metadata-only NPZ opens are additional, not pixel decodes'}
        report['schedules'] = {}
        for role in ('teacher', 'student'):
            phase('generate_and_validate_' + role + '_1000x16_schedule')
            schedule = datasets['train'].schedule(role, 1000)
            report['schedules'][role] = schedule_summary(schedule, datasets['train'])
            del schedule
        D.require(set(datasets['train']._numeric._control_sha) == set(datasets['train'].source_ids),
                  'near-clean hash cache incomplete')
        phase('final_metadata_stability')
        D.require(D.file_sha(status_path) == status_sha and D.file_sha(complete_path) == complete_ref['sha256'],
                  'terminal producer metadata changed during audit')
        for key in ('producer_contract', 'producer_freeze', 'producer_admission'): checked_json(config[key])
        for key in ('cohort', 'plan', 'train_manifest', 'dev_manifest', 'loader_source', 'revised_loader_source',
                    'numerical_window_contract', 'entry_source', 'numeric_source'):
            ref = contract['files'][key]
            D.require(D.file_sha(ref['path']) == ref['sha256'], 'small input/source pin changed during audit: ' + key)
        for entries in complete['receipt_indices'].values():
            for entry in entries:
                D.require(D.file_sha(entry['path']) == entry['sha256'], 'source receipt changed during audit')
                if 'revised_authority' in entry:
                    side = entry['revised_authority']
                    D.require(D.file_sha(side['path']) == side['sha256'], 'reuse sidecar changed during audit')
        D.require(D.file_sha(R.__file__) == WRAPPER_SHA and D.file_sha(D.__file__) == R.LOADER_SHA,
                  'actual dataset implementation changed during audit')
        D.require(not any(n.split('.')[0] in FORBIDDEN for n in sys.modules), 'forbidden model/backend loaded')
        report.update(complete=True, status='passed_revised_dataset_and_schedule_only', revised_pair_ready=True,
                      no_research_training_or_quality_acceptance=True)
    except BaseException as exc:
        report.update(complete=False, status='failed_not_ready', revised_pair_ready=False,
                      error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        report['numpy_version'] = D.np.__version__
        report['runtime_imports'] = {'wrapper': binding(R.__file__), 'loader': binding(D.__file__),
                                    'forbidden_loaded': [n for n in sys.modules if n.split('.')[0] in FORBIDDEN]}
        write(out / 'audit.json', report)
    return report


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True); p.add_argument('--status-sha256', required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    config = checked_json({'path': str(a.config.resolve()), 'sha256': a.config_sha256})
    result = audit(config, a.status_sha256, a.out)
    return 0 if result['complete'] else 1


if __name__ == '__main__': raise SystemExit(main())
