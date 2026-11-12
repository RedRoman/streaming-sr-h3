"""Separate train final-K1 export and dev cached-K3 preparation.

Inspection checks immutable metadata/bytes and imports no torch. Execution
requires a role-specific external admission. Train uses the original 16000
student rows once; dev reconstructs histories only from authenticated existing
current crops. No optimizer, backward, fallback, resume, or training admission.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import train_history_utility as T

SCHEMA = 'rphi.input_preservation_cache_execution.v1'
REPORT_SCHEMA = 'rphi.input_preservation_cache.v1'
MAX_BYTES = 3 << 30
RESERVE = 16 << 20
BATCH = 16
PROPOSAL_SHA = 'd1dd2afd1b9daddea8340fe1edff7c71ed6215fbf6f3ef038adaa4b01aa262de'
SCREEN_SHA = '144120baa552ae8371a73a03dcd3c48da9204e86a9f7e87186f3697ee0178577'
TEACHER_SHA = '79995694df154084af6e76020dfa05c0f64010600bba839521fee7a01d326e34'
HEADROOM_SHA = '6760dccb1e71cefae86ee23931b4b5b1491c88ba03f5c3a331e113183ed71578'
HEADROOM_AUDIT_SHA = '38d783deb2f41fc31da4657ba0329383bc192fd41ff943962d3041e5d4607807'
MANIFEST_SHA = '5c001aaa3335e335a16311f9d42e400f29274d65049f967859b7969c9d75ee0b'
STUDENT_FILE_SHA = 'b5e6d4c6031d79d5b9457c98fe215d49e393e335994559597498176f4e666f08'
STUDENT_SHA = 'adda5689d88417b4141ca8f632a7b03515a829b18266057312569dea3298a100'
DEPENDENCIES = (*T.DEPENDENCIES, 'eval_history_utility_teacher.py', 'history_utility_headroom.py',
                'rphi_input_preservation.py', 'prepare_input_preservation_donors.py',
                'export_input_preservation_cache.py')
SOURCE_ARRAYS = ('current', 'target', 'all_history', 'current_only', 'remove_0', 'remove_1')
HEADROOM_EXTRA = {'observed_crop_sha256_float32', 'target_crop_sha256_float32', 'availability',
    'all_history_mae', 'current_only_mae', 'raw_utility', 'metrics',
    'all_history_mae_harm_vs_input', 'current_only_mae_harm_vs_input'}


def asha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False)+'\n').encode()


def payload_budget(rows=16000):
    values = {'k1_rgb': rows*3*96*96*4, 'raw_features': rows*2*11*6*6*4,
              'three_FP64_maps': rows*3*6*6*8, 'normalized_FP32_map': rows*6*6*4,
              'availability': rows*2*4}
    return {'arrays': values, 'total_array_payload_bytes': sum(values.values()),
            'limit_bytes': MAX_BYTES, 'excludes': 'NPZ headers, row/index JSON and final receipts; all actual writes are also bounded'}


class Output:
    """Fresh owned directory; exact bytes tracked without repeated tree scans."""
    def __init__(self, path, limit=MAX_BYTES):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.limit, self.bytes, self.sizes = limit, 0, {}

    def put(self, name, data, *, replace=False, control=False):
        p = self.path/name
        T.require(p.parent == self.path or p.parent == self.path/'arrays', 'unsupported output path')
        p.parent.mkdir(exist_ok=True)
        old = self.sizes.get(name, 0)
        T.require((replace and name in self.sizes) or not p.exists(), 'cache output overwrite forbidden')
        projected = self.bytes-old+len(data)
        T.require(projected+(0 if control else RESERVE) <= self.limit, 'actual output byte budget exceeded')
        temporary = p.with_suffix(p.suffix+'.tmp')
        with temporary.open('xb') as f:
            f.write(data)
        temporary.replace(p)
        self.sizes[name] = len(data); self.bytes = projected
        return {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

    def append(self, name, rows):
        data = b''.join(encoded(r) for r in rows)
        T.require(self.bytes+len(data)+RESERVE <= self.limit, 'record output byte budget exceeded')
        p = self.path/name
        T.require(p.parent == self.path and (not p.exists() or name in self.sizes), 'unowned record output')
        with p.open('ab') as f:
            f.write(data)
        self.sizes[name] = self.sizes.get(name, 0)+len(data); self.bytes += len(data)

    def partial(self, report):
        # Compact progress, not a growing dump of every completed batch.
        keys = ('schema', 'role', 'complete', 'status', 'counts', 'completed_rows', 'failure',
                'last_completed_batch', 'frozen_before_sha256', 'frozen_after_sha256', 'frozen_checks')
        value = {k: report[k] for k in keys if k in report}
        value['output_bytes_before_partial'] = self.bytes
        self.put('partial.json', encoded(value), replace='partial.json' in self.sizes, control=True)

    def array_batch(self, index, arrays, rows):
        for value in arrays.values():
            T.require(isinstance(value, np.ndarray) and value.dtype.kind in 'fub'
                      and np.isfinite(value).all(), 'invalid cache array')
        arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
        buffer = io.BytesIO(); np.savez(buffer, **arrays)
        artifact = self.put(f'arrays/batch_{index:04d}.npz', buffer.getvalue())
        artifact.update(batch_index=index, arrays={k: {'dtype': str(v.dtype), 'shape': list(v.shape),
            'sha256': asha(v)} for k, v in arrays.items()}, row_indices=[r['row_index'] for r in rows],
            row_sha256=[r['row_sha256'] for r in rows])
        self.append('batches.jsonl', [artifact])
        return artifact

    def finish(self, report):
        for name in ('records.jsonl', 'batches.jsonl'):
            if name in self.sizes:
                report[name.split('.')[0]] = {'path': str(self.path/name), 'sha256': T.file_sha(self.path/name),
                                             'bytes': self.sizes[name]}
        report['actual_output_bytes'] = self.bytes
        # Include this final JSON's own size without a self-referential hash.
        while True:
            data = encoded(report); total = self.bytes+len(data)
            if report['actual_output_bytes'] == total:
                break
            report['actual_output_bytes'] = total
        self.put('manifest.json' if report['complete'] else 'failure.json', data, control=True)


def verify_contract(path, role, *, execute=False, out=None):
    c = T.read(path); root = c['evidence_root']
    T.require(c['root'] == root and Path(root).is_absolute() and os.path.abspath(root) == root,
              'root and evidence_root must name the same absolute workspace')
    T.require(c['schema'] == SCHEMA and c['role'] == role and role in ('train', 'dev')
              and c['batch_size'] == BATCH and c['max_output_bytes'] == MAX_BYTES,
              'cache role/schema/budget differs')
    expected_resources = {'wall_seconds': 900, 'CPU_cores': 4 if role == 'train' else 2,
                          'RSS_limit_bytes': (32 if role == 'train' else 4) << 30}
    T.require(c['resources'] == expected_resources
              and c['data_cache_bytes'] == (12 << 30 if role == 'train' else 0), 'cache resource contract differs')
    fixed = {'proposal': PROPOSAL_SHA, 'screen': SCREEN_SHA, 'teacher_checkpoint': TEACHER_SHA,
             'headroom_report': HEADROOM_SHA, 'headroom_audit': HEADROOM_AUDIT_SHA,
             'schedule_manifest': MANIFEST_SHA, 'student_schedule': STUDENT_FILE_SHA}
    names = set(fixed) | {'teacher_contract', 'teacher_report', 'dataset_audit', 'headroom_contract', 'headroom_records'}
    T.require(set(c['files']) == names and all(c['files'][k]['sha256'] == v for k, v in fixed.items()),
              'fixed evidence identity differs')
    paths = {k: T.bound_path(v, root) for k, v in c['files'].items()}
    docs = {k: T.read(p) for k, p in paths.items() if k not in ('teacher_checkpoint', 'student_schedule', 'headroom_records')}
    h, ha, teacher, terminal, audit = (docs[k] for k in
                                     ('headroom_report', 'headroom_audit', 'teacher_contract', 'teacher_report', 'dataset_audit'))
    T.require(h['complete'] is True and h['status'] == 'complete_headroom_fail'
              and h['assessment']['headroom_rule_passed'] is False and ha['complete'] is True
              and ha['report_sha256'] == HEADROOM_SHA and ha['actual_cache_files'] == 180
              and ha['actual_arrays'] == 1080 and ha['actual_rows'] == 2880,
              'prior failed headroom/actual cache audit differs')
    for name in ('teacher_contract', 'teacher_report', 'teacher_checkpoint', 'dataset_audit', 'headroom_contract'):
        T.require(c['files'][name] == h['execution_contract'][name], 'headroom teacher/data binding differs: '+name)
    T.require(c['files']['headroom_records'] == {k: h['records'][k] for k in ('path', 'sha256')}
              and h['records']['rows'] == 2880 and len(h['actual_rgb_caches']) == 180, 'headroom cache/records incomplete')
    T.require(terminal['complete'] is True and terminal['completed_step'] == 1000 and terminal['final_frozen'] is True
              and terminal['execution_contract'] == teacher
              and terminal['execution_contract_file_sha256'] == c['files']['teacher_contract']['sha256']
              and terminal['final_model_sha256'] == h['frozen_before_sha256'] == h['frozen_after_sha256']
              and teacher['model_config'] == T.CONFIG and teacher['optimizer'] == T.OPTIMIZER
              and teacher['loss'] == T.LOSS and teacher['steps'] == 1000 and teacher['phase'] == 'teacher',
              'not the accepted frozen final teacher')
    T.require(audit['complete'] is True and audit['revised_pair_ready'] is True
              and audit['pair']['train_sources'] == 19 and audit['pair']['dev_sources'] == 6
              and audit['pair'] == terminal['data']['pair'], 'train/dev data isolation proof differs')
    for name in ('schedule_manifest', 'student_schedule', 'dataset_audit'):
        T.require(c['files'][name] == teacher['files'][name], 'original teacher data file binding differs')
    manifest = docs['schedule_manifest']; schedule = manifest['student_schedule']
    T.require(manifest['complete'] is True and schedule['file_sha256'] == STUDENT_FILE_SHA
              and schedule['schedule_sha256'] == STUDENT_SHA and manifest['cohort_sha256'] == T.COHORT_SHA,
              'fixed student schedule identity differs')
    rt = c['runtime']
    T.require(rt['device'] == ('npu:0' if role == 'train' else 'cpu') and rt['cpu_threads'] == 2
              and rt['dtype'] == 'float32' and rt['autocast'] is False
              and rt['determinism'] == 'inherit' and rt['version_match'] == 'exact'
              and set(rt['expected_versions']) == {'python', 'numpy', 'torch', 'torch_npu'}, 'runtime/CPU features policy differs')
    expected_versions = dict(terminal['runtime']['versions'])
    if role == 'dev':
        expected_versions['torch_npu'] = None
    T.require(rt['expected_versions'] == expected_versions, 'actual teacher/CPU runtime declaration differs')
    pins = {str((Path(p) if Path(p).is_absolute() else Path(root)/p).resolve()): v for p, v in c['implementation_sha256'].items()}
    T.require(all(T.file_sha(p) == v for p, v in pins.items()), 'source pin mismatch')
    for name in DEPENDENCIES:
        p = Path(__file__).resolve().parent/name
        T.require(pins.get(str(p)) == T.file_sha(p), 'loaded source not pinned: '+name)
    old = {Path(p).name: v for p, v in teacher['implementation_sha256'].items()}
    for name in T.DEPENDENCIES:
        T.require(old.get(name) == T.file_sha(Path(__file__).parent/name), 'teacher/data implementation changed')
    old_evaluation = {Path(p).name: v for p, v in h['execution_contract']['implementation_sha256'].items()}
    for name in ('eval_history_utility_teacher.py', 'history_utility_headroom.py'):
        T.require(old_evaluation.get(name) == T.file_sha(Path(__file__).parent/name), 'accepted evaluator implementation changed')
    T.require(T.file_sha(Path(__file__).parent/'rphi_input_preservation.py') ==
              '7a52656a3119dcaad515dee0ec591bccbb21ee7756ba150b7f59827ad3b8100d', 'feature/label module changed')
    if execute:
        a = T.read(T.bound_path(c['admission'], root))
        T.require(a['schema'] == 'rphi.input_preservation_cache_admission.v1' and a['approved'] is True
                  and a['role'] == role and a['execution_sha256'] == T.execution_sha(c)
                  and a['entry_sha256'] == T.file_sha(__file__) and a['out'] == str(Path(out).resolve())
                  and a['device'] == rt['device'], 'separate exact cache admission required')
    return c, paths, docs


def checked_array(value, shape, name, *, unit=False, dtype=np.float32):
    T.require(isinstance(value, np.ndarray) and value.dtype == dtype and value.shape == tuple(shape)
              and np.isfinite(value).all(), 'invalid '+name)
    if unit:
        T.require(((value >= 0) & (value <= 1)).all(), 'non-unit '+name)
    return np.ascontiguousarray(value)


def make_train_batch(batch, rows, predict, feature_fn, label_fn):
    """Injected numeric callbacks permit CPU fixtures; production pins originals."""
    T.require(set(batch['model_inputs']) == {'observed', 'availability'} and 'teacher_weights' not in batch,
              'not raw-only original student inputs')
    b = len(rows); x = checked_array(batch['model_inputs']['observed'], (b,3,3,96,96), 'observed', unit=True)
    availability = checked_array(batch['model_inputs']['availability'], (b,2), 'availability', unit=True)
    T.require(np.isin(availability, [0,1]).all() and len(batch['metadata']) == b, 'binary availability/row count differs')
    for i, (r, m) in enumerate(zip(rows, batch['metadata'])):
        T.require(r['split'] == 'train' and r['teacher_mask'] is None
                  and all(m.get(k) == v for k, v in r.items())
                  and r['row_sha256'] == T.value_sha({k:v for k,v in r.items() if k != 'row_sha256'})
                  and asha(x[i]) == m['observed_crop_sha256_float32'], 'train row/input integrity differs')
        T.require(np.array_equal(availability[i], np.asarray(r['true_availability'],np.float32)*r['external_mask']),
                  'effective availability differs')
    # Neither callback receives target or source/task metadata.
    features = checked_array(feature_fn(x, availability), (b,2,11,6,6), 'raw features')
    k1 = checked_array(predict(x), (b,3,96,96), 'K1 output')
    target = checked_array(batch['target'], (b,3,96,96), 'target', unit=True)
    for i, m in enumerate(batch['metadata']):
        T.require(asha(target[i]) == m['target_crop_sha256_float32'], 'target recompute hash differs')
    labels = label_fn(x[:,-1], k1, target)
    T.require(set(labels) == {'input_mae','k1_mae','raw_benefit','normalized_benefit'}, 'label keys differ')
    for name, value in labels.items():
        checked_array(value, (b,1,6,6), name, dtype=np.float32 if name == 'normalized_benefit' else np.float64)
    arrays = {'k1_rgb': k1, 'raw_features': features, 'availability': availability, **labels}
    records = []
    for i, m in enumerate(batch['metadata']):
        record = {**m, 'cache_role': 'train', 'usable_for_training': True,
            'current_sha256_float32': asha(x[i,-1]), 'K1_sha256_float32': asha(k1[i]),
            'feature_sha256_float32': asha(features[i]), 'availability': availability[i].tolist(),
            'array_row_sha256': {k: asha(v[i]) for k,v in arrays.items()},
            'teacher_checkpoint_sha256': TEACHER_SHA,
            'label_saturation_fraction': float(np.mean(np.abs(labels['normalized_benefit'][i]) == 4)),
            'K1_clip_fraction': float(np.mean((k1[i] < 0) | (k1[i] > 1)))}
        record['cache_record_sha256'] = T.value_sha(record); records.append(record)
    return arrays, records


def source_recompute_references(data):
    """Keep actual source-file/receipt bindings, not merely pixel hash claims."""
    refs = {}
    for sid, source in data._numeric.sources.items():
        artifacts = {}
        for name, a in source['artifacts'].items():
            p = source['root']/(name+'.npz') if source['root'] else Path(a['path'])
            artifacts[name] = {**a, 'path': str(p.resolve())}
        refs[sid] = {'source': source['receipt']['source'], 'window_key': source['receipt']['window_key'],
                     'artifacts': artifacts, 'authority': data.authorities[sid]}
    return {'dataset_binding': data.binding, 'sources': refs,
        'recompute_rule': 'Use pinned RevisedHistoryUtilityDataset.batch on the unchanged student rows; verify observed/target/current row SHA and CPU features. No newly sampled row or profile.'}


def run_train(c, paths, docs, sink, report):
    T.require(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0', 'backend autoload must be disabled before torch import')
    import torch
    import eval_history_utility_teacher as E
    from rphi_input_preservation import InputPreservationHead, input_benefit_labels
    import prepare_input_preservation_donors as D
    teacher = docs['teacher_contract']
    data, schedules, proof = T.load_inputs({**teacher, 'data_cache_bytes': c['data_cache_bytes']})
    rows = D.validate_schedule(schedules['student'], docs['proposal'])
    T.require(schedules['student']['schedule_sha256'] == STUDENT_SHA and len(rows) == 16000, 'not exact frozen train schedule')
    report.update(data_proof=proof, source_recompute_references=source_recompute_references(data),
                  teacher_checkpoint_sha256=TEACHER_SHA, schedule_sha256=STUDENT_SHA)
    model, device, runtime = E.load_teacher(paths['teacher_checkpoint'], docs['teacher_report'], c['runtime'])
    report['runtime'] = runtime
    head = InputPreservationHead().eval().requires_grad_(False)
    before = T.state_sha(model.state_dict()); versions = {n:p._version for n,p in model.named_parameters()}
    report['frozen_before_sha256'] = before
    counts = report['counts']
    def observe(_module, _args):
        counts['restorer_invocations_observed'] += 1
        T.require(counts['restorer_invocations_observed'] <= 1000, 'extra K1 forward')
    hook = model.register_forward_pre_hook(observe)
    def features(x, a):
        with torch.inference_mode():
            value = head.features(torch.from_numpy(x), torch.from_numpy(a)).numpy().copy()
        counts['raw_feature_batches'] += 1; counts['raw_feature_rows'] += len(x)
        return value
    def predict(x):
        counts['restorer_forward_attempts'] += 1; counts['crop_forward_attempts'] += len(x)
        T.require(counts['restorer_forward_attempts'] <= 1000 and counts['crop_forward_attempts'] <= 16000, 'K1 budget exceeded')
        with torch.inference_mode():
            obs = torch.from_numpy(x).to(device)
            weights = torch.zeros((len(x),2,96,96),dtype=torch.float32,device=device)
            result = model(obs, weights)['rgb'].detach().cpu().numpy().copy()
        counts['restorer_forwards_completed'] += 1; counts['crop_forwards_completed'] += len(x)
        return result
    try:
        for start in range(0,16000,BATCH):
            selected = rows[start:start+BATCH]
            arrays, records = make_train_batch(data.batch(selected), selected, predict, features, input_benefit_labels)
            artifact = sink.array_batch(start//BATCH, arrays, selected)
            for i, r in enumerate(records):
                r.update(batch_index=start//BATCH, batch_row=i, cache_file_sha256=artifact['sha256'])
                r['cache_record_sha256'] = T.value_sha({k:v for k,v in r.items() if k != 'cache_record_sha256'})
            sink.append('records.jsonl', records)
            report.update(completed_rows=start+BATCH, last_completed_batch=artifact)
            sink.partial(report)
        T.require(counts['restorer_forward_attempts'] == counts['restorer_forwards_completed'] ==
                  counts['restorer_invocations_observed'] == 1000 and
                  counts['crop_forward_attempts'] == counts['crop_forwards_completed'] == 16000, 'incomplete train export')
    finally:
        hook.remove()
        report['frozen_after_sha256'] = T.state_sha(model.state_dict())
        report['frozen_checks'] = {'state_dict_exact': before == report['frozen_after_sha256'],
            'versions_exact': versions == {n:p._version for n,p in model.named_parameters()},
            'eval_frozen_no_grads': not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())}
        if device.type == 'npu':
            report['peak_npu_allocated_bytes'] = torch.npu.max_memory_allocated(device)
        sink.partial(report)
    T.require(all(report['frozen_checks'].values()), 'frozen teacher changed')
    for source in report['source_recompute_references']['sources'].values():
        for a in source['artifacts'].values():
            T.require(Path(a['path']).stat().st_size == a['bytes_on_disk'] and T.file_sha(a['path']) == a['file_sha256'],
                      'source artifact changed during export')


def load_old_cache(entry, output_root):
    path = Path(entry['path']); output_root = Path(output_root).resolve()
    T.require(not path.is_symlink() and path.resolve().parent == output_root/'arrays'
              and path.stat().st_size == entry['bytes'] and T.file_sha(path) == entry['sha256'], 'old cache file identity differs')
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        T.require(len(members) == 6 and {m.filename for m in members} == {n+'.npy' for n in SOURCE_ARRAYS}
                  and all(m.compress_type == zipfile.ZIP_STORED and m.file_size < 2 << 20 for m in members),
                  'old NPZ members/compression/size differ')
    retained = {}
    with np.load(path, allow_pickle=False) as z:
        for name in SOURCE_ARRAYS:
            a = checked_array(z[name], (16,3,96,96), name, unit=name in ('current','target'))
            T.require(entry['arrays'][name] == {'dtype':'float32','shape':[16,3,96,96],'sha256':asha(a)}, 'old array SHA differs')
            if name in ('current','target','current_only'):
                retained[name] = a
    return retained


def reconstruct_group(records, current):
    """One complete same-source/profile96-frame center crop, with no fallback."""
    T.require(len(records) == 96, 'missing history group rows')
    checked_array(current, (96,3,96,96), 'cached current group', unit=True)
    first = records[0]
    for t, r in enumerate(records):
        T.require(r['split'] == 'dev' and r['frame_index'] == t
                  and all(r[k] == first[k] for k in ('source_id','source_sha256','source_group_id','window_key','profile_id','crop_tlhw'))
                  and r['crop_tlhw'] == [192,368,96,96] and r['history_indices'] == [max(t-2,0),max(t-1,0),t],
                  'noncausal/missing/mixed cached history')
    # Complete96 currents are needed for reconstruction; no source NPZ fallback.
    for t, r in enumerate(records):
        x = np.ascontiguousarray(current[r['history_indices']])
        T.require(asha(x) == r['observed_crop_sha256_float32'], 'reconstructed rawK3 bytes differ')
        T.require(asha(current[t]) == r['metrics']['input']['prediction_float32_sha256'], 'cached current row differs')
    return np.ascontiguousarray(np.stack([current[r['history_indices']] for r in records]))


def rgb_metrics(prediction, target):
    x, y = prediction.astype(np.float64), target.astype(np.float64)
    e = x-y; clipped = np.clip(x,0,1)-y; mse = float(np.mean(clipped**2))
    return {'unclamped_mae':float(np.mean(np.abs(e))), 'unclamped_mse':float(np.mean(e**2)),
        'clipped_mae':float(np.mean(np.abs(clipped))), 'clipped_mse':mse,
        'clipped_psnr':None if mse == 0 else float(-10*np.log10(mse)), 'clipped_exact_match':mse == 0}


def two_candidate_diagnostic(current, k1, target, label_fn):
    labels = label_fn(current,k1,target)
    clipped_labels = label_fn(current,np.clip(k1,0,1),target)
    raw_choice = labels['raw_benefit'] > 0
    clip_choice = clipped_labels['raw_benefit'] > 0
    def compose(choice):
        mask = np.repeat(np.repeat(choice,16,axis=2),16,axis=3)
        return np.where(mask,k1,current)
    raw_oracle, clip_oracle = compose(raw_choice), compose(clip_choice)
    rows = []
    for i in range(len(current)):
        rows.append({'input':rgb_metrics(current[i],target[i]), 'K1':rgb_metrics(k1[i],target[i]),
            'unclamped_tile_oracle':rgb_metrics(raw_oracle[i],target[i]),
            'clipped_tile_oracle':rgb_metrics(clip_oracle[i],target[i]),
            'raw_oracle_K1_tile_fraction':float(raw_choice[i].mean()),
            'clipped_oracle_K1_tile_fraction':float(clip_choice[i].mean())})
    return {'diagnostic_input_mae':labels['input_mae'], 'diagnostic_K1_mae':labels['k1_mae'],
            'diagnostic_clipped_input_mae':clipped_labels['input_mae'],
            'diagnostic_clipped_K1_mae':clipped_labels['k1_mae'],
            'oracle_unclamped_index':raw_choice.astype(np.uint8),
            'oracle_clipped_index':clip_choice.astype(np.uint8)}, rows


def summarize_diagnostic(rows):
    groups = defaultdict(list)
    for row in rows:
        for phase in {row['severity_phase'],'all_frames'}:
            groups[row['source_id'],row['profile_id'],phase].append(row['two_candidate_diagnostic'])
    detailed = []
    modes = ('input','K1','unclamped_tile_oracle','clipped_tile_oracle')
    for (sid,pid,phase), values in sorted(groups.items()):
        m = {}
        for name in modes:
            item = {k:float(np.mean([r[name][k] for r in values])) for k in
                    ('unclamped_mae','unclamped_mse','clipped_mae','clipped_mse')}
            item['clipped_psnr_from_mean_mse'] = None if item['clipped_mse'] == 0 else float(-10*np.log10(item['clipped_mse']))
            m[name] = item
        detailed.append({'source_id':sid,'profile_id':pid,'severity_phase':phase,'frames':len(values),'metrics':m,
            **{k:float(np.mean([r[k] for r in values])) for k in ('raw_oracle_K1_tile_fraction','clipped_oracle_K1_tile_fraction')}})
    source_macro = []
    for sid in sorted({r['source_id'] for r in detailed}):
        native = [r for r in detailed if r['source_id']==sid and r['severity_phase']=='all_frames'
                  and r['profile_id'] not in ('identity','near_clean')]
        source_macro.append({'source_id':sid,'native_equal_profile':{mode:{k:float(np.mean([r['metrics'][mode][k] for r in native]))
            for k in ('unclamped_mae','unclamped_mse','clipped_mae','clipped_mse')} for mode in modes}})
    macro = {mode:{k:float(np.mean([r['native_equal_profile'][mode][k] for r in source_macro]))
                  for k in ('unclamped_mae','unclamped_mse','clipped_mae','clipped_mse')} for mode in modes}
    return {'per_source_profile_phase':detailed,'per_source_native':source_macro,'native_equal_source_equal_profile':macro,
        'scope':'GT-dependent hard tile oracle of input and K1 only, not a deployable head, not the prior five-candidate bound or a screen pass.'}


def run_dev(c, paths, docs, sink, report):
    T.require(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0', 'backend autoload must be disabled before torch import')
    import torch
    import eval_history_utility_teacher as E
    from rphi_input_preservation import InputPreservationHead, input_benefit_labels
    T.require('torch_npu' not in sys.modules, 'dev preparation must not import NPU backend')
    torch.set_num_threads(2)
    actual = {'python':'.'.join(map(str,sys.version_info[:3])),'numpy':np.__version__,
              'torch':str(torch.__version__),'torch_npu':None}
    T.require(actual == c['runtime']['expected_versions'], 'actual CPU feature runtime differs')
    report['runtime'] = {'versions':actual,'device':'cpu','threads':2}
    head = InputPreservationHead().eval().requires_grad_(False)
    records = [json.loads(line) for line in paths['headroom_records'].read_text().splitlines()]
    base = [{k:v for k,v in r.items() if k not in HEADROOM_EXTRA} for r in records]
    E.validate_rows(base, docs['headroom_contract'])
    old = docs['headroom_report']
    T.require(T.value_sha(base) == old['evaluation_rows_sha256'], 'full dev row table differs')
    root = Path(old['records']['path']).resolve().parent
    diagnostics = []; total_bytes = 0
    for start in range(0,2880,96):
        group = records[start:start+96]; cached = {k:[] for k in ('current','target','current_only')}; refs = []
        for begin in range(start,start+96,BATCH):
            entry = old['actual_rgb_caches'][begin//BATCH]; chosen = records[begin:begin+BATCH]
            T.require(entry['row_indices'] == list(range(begin,begin+BATCH)) and
                      entry['row_sha256'] == [r['row_sha256'] for r in chosen], 'old cache row ordering differs')
            report['counts']['old_cache_file_verification_attempts'] += 1
            arrays = load_old_cache(entry,root); total_bytes += entry['bytes']
            report['counts']['old_cache_files_verified'] += 1; report['counts']['old_arrays_verified'] += 6
            refs.append(entry)
            for k in cached:
                cached[k].append(arrays[k])
        cached = {k:np.concatenate(v) for k,v in cached.items()}
        x = reconstruct_group(group,cached['current'])
        report['counts']['raw_K3_exact_rows'] += len(group)
        for i,r in enumerate(group):
            T.require(asha(cached['target'][i]) == r['target_crop_sha256_float32'] and
                      asha(cached['current_only'][i]) == r['metrics']['current_only']['prediction_float32_sha256'],
                      'cached target/K1 row identity differs')
        for offset in range(0,96,BATCH):
            selected = group[offset:offset+BATCH]; a = np.asarray([r['availability'] for r in selected],np.float32)
            expected_a = np.asarray([r['true_availability'] for r in selected],np.float32)
            T.require(np.array_equal(a,expected_a), 'dev availability differs')
            with torch.inference_mode():
                features = head.features(torch.from_numpy(x[offset:offset+BATCH]),torch.from_numpy(a)).numpy().copy()
            checked_array(features,(BATCH,2,11,6,6),'dev raw features')
            report['counts']['raw_feature_batches'] += 1; report['counts']['raw_feature_rows'] += BATCH
            cur,y,k1 = (cached[k][offset:offset+BATCH] for k in ('current','target','current_only'))
            diag_arrays, values = two_candidate_diagnostic(cur,k1,y,input_benefit_labels)
            arrays = {'raw_features':features,'availability':a,**diag_arrays}
            artifact = sink.array_batch((start+offset)//BATCH,arrays,selected); out_rows=[]
            for i,(row,diag) in enumerate(zip(selected,values)):
                r = {**{k:v for k,v in row.items() if k not in HEADROOM_EXTRA}, 'cache_role':'dev',
                     'usable_for_training':False,'observed_crop_sha256_float32':row['observed_crop_sha256_float32'],
                     'target_crop_sha256_float32':row['target_crop_sha256_float32'],
                     'raw_K3_reconstructed_exact':True,'feature_sha256_float32':asha(features[i]),
                     'teacher_checkpoint_sha256':TEACHER_SHA,'availability':a[i].tolist(),
                     'input_RGB_reference':{'cache':refs[offset//BATCH],'batch_row':i},
                     'array_row_sha256':{k:asha(v[i]) for k,v in arrays.items()},
                     'two_candidate_diagnostic':diag,'cache_file_sha256':artifact['sha256']}
                r['cache_record_sha256'] = T.value_sha(r); out_rows.append(r); diagnostics.append(r)
            sink.append('records.jsonl',out_rows)
            report.update(completed_rows=start+offset+BATCH,last_completed_batch=artifact)
            sink.partial(report)
    T.require(total_bytes == old['array_bytes_on_disk'] == docs['headroom_audit']['actual_cache_bytes'], 'old cache byte coverage differs')
    report.update(original_cache_bytes_verified=total_bytes,two_candidate_diagnostic=summarize_diagnostic(diagnostics),
                  frozen_teacher_constructed=False,original_headroom_report_sha256=HEADROOM_SHA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=('train','dev'),required=True)
    p.add_argument('--execution-contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    c,paths,docs=verify_contract(a.execution_contract,a.role,execute=a.execute,out=a.out)
    sink=Output(a.out);started=time.monotonic()
    report={'schema':REPORT_SCHEMA,'role':a.role,'complete':False,'status':'started',
        'execution_contract':c,'execution_contract_file_sha256':T.file_sha(a.execution_contract),
        'original_history_headroom_remains_failed':True,'head_training_admitted':False,
        'usable_for_training':a.role=='train','cache_scope':'96x96 crops only; no full-frame or quality acceptance',
        'payload_budget':payload_budget(),'counts':{'restorer_forward_attempts':0,'restorer_forwards_completed':0,
            'restorer_invocations_observed':0,'crop_forward_attempts':0,'crop_forwards_completed':0,
            'backwards':0,'optimizer_updates':0,'raw_feature_batches':0,'raw_feature_rows':0,
            'old_cache_file_verification_attempts':0,'old_cache_files_verified':0,'old_arrays_verified':0,'raw_K3_exact_rows':0},
        'completed_rows':0,'input_files':c['files'],'array_semantics':'FP32 K1/raw-only CPU features; FP64 tile errors; train supervision and dev diagnostic are separate roles'}
    sink.partial(report)
    try:
        if a.execute:
            (run_train if a.role=='train' else run_dev)(c,paths,docs,sink,report)
            T.require(report['completed_rows'] == (16000 if a.role=='train' else 2880), 'incomplete cache rows')
            T.require(report['counts']['raw_feature_rows'] == report['completed_rows'] and
                      report['counts']['raw_feature_batches']*BATCH == report['completed_rows'], 'incomplete raw-feature coverage')
            if a.role == 'dev':
                T.require(report['counts']['raw_K3_exact_rows'] == 2880 and
                          report['counts']['old_cache_file_verification_attempts'] == report['counts']['old_cache_files_verified'] == 180
                          and report['counts']['old_arrays_verified'] == 1080, 'incomplete cached-dev verification')
            report['status']='cache_complete_pending_independent_audit_no_training_admission'
        else:
            report.update(status='metadata_inspected_no_model_no_cache',usable_for_training=False,
                          inspection_only=True,actual_source_pixels_verified_this_run=False)
        verify_contract(a.execution_contract,a.role,execute=a.execute,out=a.out)
        T.require(T.file_sha(a.execution_contract)==report['execution_contract_file_sha256'],'contract changed')
        report['complete']=True
    except BaseException as error:
        report.update(status='failed_no_retry',failure={'type':type(error).__name__,'message':str(error)})
    report['elapsed_seconds']=time.monotonic()-started
    sink.partial(report);sink.finish(report)
    print(json.dumps({k:report[k] for k in ('complete','role','status','completed_rows','counts','actual_output_bytes')}))
    return 0 if report['complete'] else 1


if __name__ == '__main__':raise SystemExit(main())
