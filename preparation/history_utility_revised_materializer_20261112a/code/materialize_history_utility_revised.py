"""Prepare the explicitly revised P1 cohort from immutable target caches.

Default inspection reads metadata and SHA-bound evidence, never a codec. Actual
execution is separately admitted and supervised. It references three verified
complete old sources and materializes only the remaining primary sources. The
old13b6 contract remains a numerical window/noise namespace, not the authority
for this run. Original MP4s may be hashed but are never decoded or probed here.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import argparse
import gc
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

SCHEMA = 'openh3.history_utility_revised_materialization.v1'
CONTRACT_SCHEMA = 'openh3.history_utility_revised_materialization_execution.v1'
AUTHORITY_SCHEMA = 'openh3.history_utility_revised_materialization_authority.v1'
REUSE_SCHEMA = 'openh3.history_utility_revised_reuse_admission.v1'
COHORT_SHA = 'c4c3a1c044fa13e6af93c03259ca0e52ea18531ddb6a2c55bbd19abff81d44e1'
NUMERIC_SHA = '80fa35dd84e941fb009b276cc704afb07b41088ca2b80ba72604afedc40d1c37'
REUSE_IDS = ('4EKWTiWjPTQ', '6UZVovkR3IU', '6rJrrfRO3cA')
BUDGET = {'wall_seconds': 3600, 'termination_grace_seconds': 30, 'memory_mib': 768,
          'worker_processes': 1, 'threads': 1, 'minimum_free_bytes': 6 << 30,
          'maximum_new_output_bytes': 30 << 30, 'source_MP4_decodes': 0,
          'model_calls': 0, 'NPU_calls': 0}
FORBIDDEN = {'torch', 'torch_npu', 'comfy', 'comfy_kitchen', 'h3_comfy', 'rphi_model'}


class NoModels(importlib.abc.MetaPathFinder):
    def __init__(self): self.attempts = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            self.attempts.append(fullname)
            raise ImportError('model/backend forbidden: ' + fullname)


GUARD = NoModels()
sys.meta_path.insert(0, GUARD)


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def bound(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def check_models():
    loaded = [name for name in sys.modules if name.split('.')[0] in FORBIDDEN]
    require(not loaded and not GUARD.attempts, 'model/backend import occurred')
    return {'forbidden_modules': loaded, 'forbidden_import_attempts': list(GUARD.attempts)}


def import_bound(name, binding):
    path = Path(binding['path'])
    require(path.is_absolute() and sha(path) == binding['sha256'], 'implementation SHA differs')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_context(contract):
    """Pure inspection: no pixel-array, video, codec or model execution."""
    require(contract['schema'] == CONTRACT_SCHEMA and contract['budget'] == BUDGET,
            'execution schema/budget differs')
    files = contract['files']
    expected = {'entry_source', 'loader_source', 'revised_loader_source', 'numeric_source',
                'numerical_window_contract', 'cohort', 'plan', 'train_manifest', 'dev_manifest',
                'source_inventory'}
    require(set(files) == expected, 'execution file bindings incomplete')
    for name, ref in files.items():
        require(Path(ref['path']).is_absolute() and sha(ref['path']) == ref['sha256'],
                'bound file changed: ' + name)
    require(Path(files['entry_source']['path']).resolve() == Path(__file__).resolve(), 'unbound entry')
    require(files['cohort']['sha256'] == COHORT_SHA and files['numeric_source']['sha256'] == NUMERIC_SHA,
            'frozen cohort/numerical implementation differs')
    D = import_bound('history_utility_data', files['loader_source'])
    R = import_bound('_p1_revised_loader', files['revised_loader_source'])
    require(files['loader_source']['sha256'] == R.LOADER_SHA
            and files['numerical_window_contract']['sha256'] == R.NUMERIC_WINDOW_SHA,
            'old loader/numerical namespace changed')
    plan = read(files['plan']['path']); R.validate_numeric_plan(plan)
    registry = D.load_source_registry(plan, files['train_manifest']['path'], files['dev_manifest']['path'])
    cohort = read(files['cohort']['path']); selected = R.validate_cohort(cohort, registry)
    require({s: len(v) for s, v in selected.items()} == {'train': 19, 'dev': 6}, 'revised cohort count differs')
    require(contract['selected_source_ids'] == selected and set(contract['reuse_receipts']) == set(REUSE_IDS),
            'selection/reuse set differs')
    require(set(REUSE_IDS) <= set(selected['train'] + selected['dev']), 'non-primary reuse forbidden')
    require(contract['new_sources'] == sorted(set(selected['train'] + selected['dev']) - set(REUSE_IDS))
            and len(contract['new_sources']) == 22, 'new source set differs')
    require(contract['source_MP4_decodes'] == 0 and contract['codec_passes'] == 132
            and contract['profiles'] == list(D.PROFILES) and contract['replays'] == 2,
            'fixed operation budget differs')
    original = read(files['numerical_window_contract']['path'])
    require(original['operator_contract_sha256'] == R.OPERATOR_SHA
            and original['profile_recipes_sha256'] == R.PROFILE_SHA,
            'original numerical namespace recipe differs')
    root = Path(contract['evidence_root']); require(root.is_absolute(), 'absolute evidence root required')
    R.validate_terminal_audit(cohort, registry, root)
    audit = R.bound_json(cohort['target_terminal_audit'], root)
    require(audit['schema'] == 'openh3.history_utility_target_screen_actual_terminal_audit.v1'
            and audit['complete'] is True and audit['source_count'] == 40
            and audit['target_files_actual_SHA_and_numpy_pixel_SHA_verified'] == 38,
            'actual terminal target audit incomplete')
    audited = {x['source_id']: x for x in audit['source_rows']}
    require(set(audited) == set(registry), 'terminal audit source accounting differs')
    inventory = read(files['source_inventory']['path'])
    source_files = {x['source_id']: x for x in inventory['records']}
    require(set(source_files) == set(registry), 'original source inventory incomplete')
    evidence = {}
    for sid in sorted(selected['train'] + selected['dev']):
        decision = cohort['decisions'][sid]
        evidence[sid] = R.validate_primary_evidence(sid, decision, registry, root)
        numerical = R.bound_json(decision['target_receipt'], root)
        require(audited[sid]['numerical_receipt'] == decision['target_receipt']
                and audited[sid]['target_file_sha256'] == numerical['target']['file_sha256']
                and audited[sid]['target_pixel_sha256'] == decision['target_pixel_sha256'],
                'primary differs from independently audited target')
        require(numerical['target_specification']['execution_contract_sha256'] == D.value_sha(original),
                'window/noise namespace changed')
        source = source_files[sid]
        require(source['source_sha256'] == registry[sid]['source_sha256']
                and source['split'] == registry[sid]['split'], 'original source role differs')
        evidence[sid]['numerical'] = numerical
    profiles = [next(p for p in plan['profiles'] if p['profile_id'] == pid) for pid in D.PROFILES]
    require(len(contract['software_sha256']) == 64, 'exact software fingerprint required')
    for sid, entry in contract['reuse_receipts'].items():
        old_receipt = R.bound_json(entry, root)
        D._validate_receipt(old_receipt, registry, {p['profile_id']: p for p in profiles}, R.OPERATOR_SHA)
        require(old_receipt['source']['source_id'] == sid
                and old_receipt['window_key'] == cohort['decisions'][sid]['window_key']
                and old_receipt['target_specification'] == evidence[sid]['target_specification']
                and old_receipt['target'] == evidence[sid]['target']
                and old_receipt['software_sha256'] == contract['software_sha256'],
                'complete profile reuse differs from selected target/software')
    check_models()
    return {'D': D, 'R': R, 'cohort': cohort, 'registry': registry, 'evidence': evidence,
            'source_files': source_files, 'plan': plan, 'profiles': profiles, 'original': original}


def inspect(contract):
    context = load_context(contract)
    return {'schema': SCHEMA, 'status': 'inspected_not_materialized', 'complete': False,
            'cohort': contract['files']['cohort'], 'selected_source_ids': contract['selected_source_ids'],
            'complete_profile_reuse_sources': list(REUSE_IDS), 'new_sources': contract['new_sources'],
            'new_target_decodes': 0, 'planned_codec_passes': 132, 'codec_passes_executed': 0,
            'actual_target_pixel_verification': False, 'actual_profile_pixel_verification': False,
            'target_evidence_count': len(context['evidence']), 'budget': BUDGET,
            'original_32_8_completion_claim': False, 'training_admitted': False,
            'import_guard': check_models()}


def authority(contract_path, contract, context, sid):
    return {'schema': AUTHORITY_SCHEMA, 'execution_contract': bound(contract_path),
            'execution_contract_value_sha256': context['D'].value_sha(contract),
            'cohort': contract['files']['cohort'],
            'input_target_receipt': context['cohort']['decisions'][sid]['target_receipt'],
            'input_target': context['evidence'][sid]['target']}


def verify_source_file(context, sid):
    row = context['source_files'][sid]; path = Path(row['path'])
    require(path.name == sid + '.mp4' and path.stat().st_size == row['bytes']
            and sha(path) == context['registry'][sid]['source_sha256'], 'actual source MP4 SHA changed')
    return path


def read_target(context, sid):
    import numpy as np
    D = context['D']; numerical = context['evidence'][sid]['numerical']; target = numerical['target']
    path = Path(target['path'])
    require(path.stat().st_size == target['bytes_on_disk'] and sha(path) == target['file_sha256'],
            'input target file changed')
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {'frames', 'metadata'}, 'input target NPZ keys differ')
        clean = data['frames']; metadata = json.loads(str(data['metadata'].item()))
    require(clean.dtype == np.uint8 and clean.shape == D.SHAPE and D.array_sha(clean) == target['pixel_sha256'],
            'input target pixels/geometry differ')
    expected = {**context['registry'][sid], 'role': 'target', 'window_key': numerical['window_key'],
                'target_specification': numerical['target_specification']}
    require(all(metadata.get(k) == v for k, v in expected.items()), 'input target metadata identity differs')
    if metadata['schema'] == 'openh3.history_utility_target_cache.v1':
        require(metadata['numeric_source_sha256'] == NUMERIC_SHA
                and metadata['screen_execution_contract_sha256'] == numerical['execution_contract_sha256'],
                'target-only producer namespace differs')
    else:
        require(metadata['schema'] == D.MATERIALIZER_SCHEMA
                and metadata['target_pixel_sha256'] == target['pixel_sha256'], 'unknown target metadata schema')
    clean.setflags(write=False)
    return clean


def verify_complete(context, contract, entry, split):
    D = context['D']
    data = D.HistoryUtilityDataset([entry], plan=contract['files']['plan']['path'],
        train_manifest=contract['files']['train_manifest']['path'],
        dev_manifest=contract['files']['dev_manifest']['path'], split=split,
        scope='engineering', verify_pixels=True, cache_bytes=0)
    require(not data.research_ready, 'old research gate unexpectedly true')
    return {'old_loader_actual_files_pixels_metadata_verified': 4, 'legacy_research_ready': False}


def reuse_source(contract_path, contract, context, sid, out):
    entry = contract['reuse_receipts'][sid]; receipt = read(entry['path'])
    source = verify_source_file(context, sid)
    original_bytes_sha = sha(entry['path'])
    require(original_bytes_sha == entry['sha256'], 'reused receipt changed')
    check = verify_complete(context, contract, entry, context['registry'][sid]['split'])
    require(receipt['target'] == context['evidence'][sid]['target'], 'reused target differs')
    require(sha(entry['path']) == original_bytes_sha and sha(source) == context['registry'][sid]['source_sha256'],
            'original source/receipt mutated')
    out.mkdir(parents=True, exist_ok=False)
    sidecar = {'schema': REUSE_SCHEMA, 'reused_receipt': entry,
        'execution_authority': authority(contract_path, contract, context, sid),
        'source': context['registry'][sid], 'window_key': receipt['window_key'],
        'target_pixel_sha256': receipt['target']['pixel_sha256'], 'original_receipt_unchanged': True,
        'target_pixels_unchanged': True, 'new_target_decodes': 0, 'new_codec_passes': 0,
        'verification': check, 'import_guard': check_models()}
    write(out / 'reuse_admission.json', sidecar)
    return {'mode': 'reused_complete', 'source_id': sid,
            'entry': {**entry, 'revised_authority': bound(out / 'reuse_admission.json')},
            'new_target_decodes': 0, 'new_codec_passes': 0, 'verification': check}


def materialize_source(contract_path, contract, context, old, sid, out, budget):
    D = context['D']; source = verify_source_file(context, sid); binding = context['registry'][sid]
    numerical = context['evidence'][sid]['numerical']; spec = numerical['target_specification']
    software = old.software_receipt(context['original']['ffmpeg'], context['original']['ffprobe'])
    require(D.value_sha(software) == contract['software_sha256'], 'actual codec/software build differs')
    clean = read_target(context, sid); target_sha = D.array_sha(clean); window = numerical['window_key']
    out.mkdir(parents=True, exist_ok=False)
    auth = authority(contract_path, contract, context, sid)
    registry = {'schema': 'openh3.history_utility_revised_registry.v1',
        'sources': [context['registry'][k] for k in sorted(context['registry'])],
        'selected_sources': contract['selected_source_ids'], 'cohort_sha256': COHORT_SHA,
        'manifest_bindings': {s: {'sha256': contract['files'][s + '_manifest']['sha256']} for s in ('train', 'dev')}}
    receipt = {'schema': old.VERSION, 'status': 'started_not_complete', 'producer_schema': SCHEMA,
        'producer_sha256': sha(__file__), 'source': binding, 'source_path': str(source),
        'source_stat': {'bytes': source.stat().st_size, 'mtime_ns': source.stat().st_mtime_ns},
        'source_probe': context['source_files'][sid]['probe'], 'source_probe_origin': contract['files']['source_inventory'],
        'target_specification': spec, 'target_specification_sha256': window, 'window_key': window,
        'plan_file_sha256': contract['files']['plan']['sha256'],
        'execution_contract': context['original'], 'execution_contract_scope': 'frozen numerical window/noise namespace only',
        'numerical_window_contract_file_sha256': contract['files']['numerical_window_contract']['sha256'],
        'execution_authority': auth, 'revised_cohort_sha256': COHORT_SHA,
        'input_target': numerical['target'], 'input_target_receipt': auth['input_target_receipt'],
        'registry': registry, 'registry_sha256': D.value_sha(registry),
        'software': software, 'software_sha256': D.value_sha(software), 'materializer_sha256': NUMERIC_SHA,
        'scientific_continuity_approved': False, 'source_natural_cut_status': 'external_revised_cohort_only',
        'new_split_assignment': False, 'model_calls': 0, 'npu_calls': 0, 'new_target_decodes': 0,
        'decode': numerical['decode'], 'decode_origin': auth['input_target_receipt'], 'observations': {},
        'selected_profiles': list(D.PROFILES), 'new_codec_passes_started': 0, 'new_codec_passes_completed': 0,
        'window_noise_semantics': 'unchanged frozen13b6 spec/window/keyed noise; current execution separately authorized'}
    write(out / 'receipt.partial.json', receipt)
    common = {'schema': old.VERSION, **binding, 'window_key': window, 'target_pixel_sha256': target_sha,
        **{k: receipt[k] for k in ('plan_file_sha256', 'registry_sha256', 'materializer_sha256', 'software_sha256')},
        'revised_cohort_sha256': COHORT_SHA, 'execution_authority': auth}
    try:
        receipt['target'] = old.save_npz(out / 'target.npz', clean,
            {**common, 'role': 'target', 'target_specification': spec})
        require(receipt['target']['pixel_sha256'] == numerical['target']['pixel_sha256'], 'metadata bridge changed pixels')
        receipt['metadata_bridge'] = {'target_pixels_unchanged': True, 'NPZ_bytes_expected_to_differ': True,
                                     'input': numerical['target'], 'output': receipt['target']}
        with tempfile.TemporaryDirectory(prefix='scratch_', dir=out) as directory:
            scratch = Path(directory)
            for profile in context['profiles']:
                pid = profile['profile_id']; replays = []
                for replay in range(2):
                    receipt['new_codec_passes_started'] += 1; write(out / 'receipt.partial.json', receipt)
                    observed, raw, detail = old.observation(clean, profile, binding, window, scratch, replay, budget)
                    replays.append(detail); receipt['new_codec_passes_completed'] += 1
                    if replay == 0:
                        artifact = old.save_npz(out / (pid + '.npz'), observed,
                            {**common, 'role': 'observation', 'profile': profile,
                             'parameter_schedule_sha256': D.value_sha(profile['frame_parameters']),
                             'noise_stream': profile['noise_stream'], 'blind_model_condition': 'not_supplied_new_loader_required'})
                    del observed; raw.unlink(); budget.check()
                require(replays[0]['observation_pixel_sha256'] == replays[1]['observation_pixel_sha256']
                        and replays[0]['noise_schedule_sha256'] == replays[1]['noise_schedule_sha256'],
                        'actual observation/noise double replay mismatch')
                receipt['observations'][pid] = {'artifact': artifact, 'task': profile['task'], 'scale': 1,
                    'parameter_schedule_sha256': D.value_sha(profile['frame_parameters']),
                    'replay_count': 2, 'pixel_replay_equal': True, 'replays': replays}
                require(D.array_sha(clean) == target_sha, 'immutable target pixels were changed')
                write(out / 'receipt.partial.json', receipt)
        require(receipt['observations'][D.PROFILES[1]]['replays'][0]['noise_frames'] ==
                receipt['observations'][D.PROFILES[2]]['replays'][0]['noise_frames'], 'smooth/abrupt noise differs')
        require(sha(source) == binding['source_sha256'] and sha(numerical['target']['path']) == numerical['target']['file_sha256'],
                'input source/target file changed')
        del clean; gc.collect(); budget.check()
        receipt.update(status=D.COMPLETE_STATUS, source_unchanged=True, paired_noise_equal=True,
                       paired_noise_scope=[D.PROFILES[1], D.PROFILES[2]], import_guard=check_models())
        # A candidate v1 receipt is validated before the only final complete.json.
        candidate = out / 'candidate_receipt.json'; write(candidate, receipt)
        check = verify_complete(context, contract, bound(candidate), binding['split'])
        receipt['verification'] = check; write(out / 'complete.json', receipt)
        candidate.unlink(); (out / 'receipt.partial.json').unlink()
        return {'mode': 'new_materialization', 'source_id': sid, 'entry': bound(out / 'complete.json'),
                'new_target_decodes': 0, 'new_codec_passes': 6, 'verification': check}
    except BaseException as exc:
        receipt.update(status='failed_not_complete', error=repr(exc))
        write(out / 'failure.json', receipt)
        raise


def disk_check(out):
    require(shutil.disk_usage(out).free >= BUDGET['minimum_free_bytes'], 'less than6GiB disk free')
    total = sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())
    require(total <= BUDGET['maximum_new_output_bytes'], 'new output exceeds30GiB accounting bound')
    return total


def worker(contract_path, contract, out, sid):
    context = load_context(contract)
    require(sid in contract['selected_source_ids']['train'] + contract['selected_source_ids']['dev'],
            'non-primary source forbidden')
    dest = out / 'sources' / sid
    if sid in contract['reuse_receipts']:
        result = reuse_source(contract_path, contract, context, sid, dest)
    else:
        old = import_bound('_p1_frozen_numeric', contract['files']['numeric_source'])
        require(old.SOURCE == '2BdYaGuVAKo' and len(old.PROFILE_IDS) == 6, 'frozen source constants changed')
        result = materialize_source(contract_path, contract, context, old, sid, dest, old.Budget(3600, 768))
    write(dest / 'result.json', result)


def execute(contract_path, contract, out, admission_path):
    # This coordinator calls no codec; each source executes in a fresh child
    # inheriting the supervisor-owned PG. No retry/resume/overwrite is implicit.
    require(not out.exists(), 'output already exists; explicit new run required')
    out.mkdir(parents=True); started = time.monotonic()
    result = {'schema': SCHEMA, 'complete': False, 'status': 'running', 'source_results': [],
        'cohort': contract['files']['cohort'], 'execution_contract': bound(contract_path),
        'receipt_indices': {'train': [], 'dev': []}, 'new_target_decodes': 0,
        'new_codec_passes_completed': 0, 'model_calls': 0, 'NPU_calls': 0,
        'codec_count_scope': 'completed sources; an incomplete source has separate failure/partial receipt',
        'legacy_research_ready': False, 'revised_dataset_validation_pending': True, 'training_admitted': False}
    write(out / 'partial.json', result)
    try:
        for split in ('train', 'dev'):
            for sid in contract['selected_source_ids'][split]:
                disk_check(out)
                remaining = BUDGET['wall_seconds'] - (time.monotonic() - started)
                require(remaining > 0, 'batch wall budget exhausted')
                result['current_source_id'] = sid; write(out / 'partial.json', result)
                with (out / (sid + '.log')).open('xb') as log:
                    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                        '--contract', str(contract_path), '--out', str(out), '--worker-source', sid,
                        '--admission', str(admission_path)],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
                    try: code = proc.wait(timeout=remaining)
                    except BaseException:
                        if proc.poll() is None:
                            proc.terminate()
                            try: proc.wait(timeout=5)
                            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                        raise
                require(code == 0, 'source worker failed: ' + sid)
                source_result = read(out / 'sources' / sid / 'result.json')
                require(source_result['source_id'] == sid and source_result['new_target_decodes'] == 0,
                        'source result identity/decode differs')
                result['source_results'].append(source_result)
                result['receipt_indices'][split].append(source_result['entry'])
                result['new_codec_passes_completed'] += source_result['new_codec_passes']
                result['new_output_bytes_after_source'] = disk_check(out)
                result['current_source_id'] = None
                write(out / 'partial.json', result)
        require(len(result['source_results']) == 25 and result['new_codec_passes_completed'] == 132,
                'incomplete cohort/codec accounting')
        for split, entries in result['receipt_indices'].items():
            write(out / (split + '_receipt_index.json'), {
                'schema': 'openh3.history_utility_revised_receipt_index.v1',
                'cohort': contract['files']['cohort'], 'receipts': entries})
        result.update(complete=True, status='materialized_revised_primary_pending_dataset_validation',
                      elapsed_seconds=time.monotonic() - started, import_guard=check_models())
        write(out / 'complete.json', result); (out / 'partial.json').unlink()
    except BaseException as exc:
        if result.get('current_source_id'):
            source_dir = out / 'sources' / result['current_source_id']
            for name in ('failure.json', 'receipt.partial.json'):
                if (source_dir / name).exists():
                    result['incomplete_source_evidence'] = bound(source_dir / name)
                    break
        result.update(complete=False, status='failed', error=repr(exc), elapsed_seconds=time.monotonic()-started)
        write(out / 'failure.json', result)
        raise


def validate_admission(path, contract_path):
    require(path is not None, 'separate reviewed execution admission required')
    admission = read(path)
    require(admission['schema'] == 'openh3.history_utility_revised_materialization_admission.v1'
            and admission['approved'] is True and admission['contract_sha256'] == sha(contract_path)
            and admission['entry_sha256'] == sha(__file__), 'execution admission differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--worker-source')
    args = parser.parse_args(); contract = read(args.contract)
    if args.worker_source:
        validate_admission(args.admission, args.contract)
        require(not args.execute and args.out.is_dir(), 'worker must belong to an existing owned run')
        worker(args.contract.resolve(), contract, args.out.resolve(), args.worker_source)
    elif args.execute:
        validate_admission(args.admission, args.contract)
        inspect(contract)
        execute(args.contract.resolve(), contract, args.out.resolve(), args.admission.resolve())
    else:
        require(not args.out.exists(), 'inspection output already exists')
        report = inspect(contract)
        args.out.mkdir(parents=True); write(args.out / 'inspection.json', report)
        print(json.dumps({'status': report['status'], 'target_evidence_count': report['target_evidence_count']}))


if __name__ == '__main__': main()
