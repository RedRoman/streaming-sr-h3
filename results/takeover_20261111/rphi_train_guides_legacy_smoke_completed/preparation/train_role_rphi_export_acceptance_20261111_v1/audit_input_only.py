from pathlib import Path
import sys, json, time, datetime, tarfile
import numpy as np

root = Path('/cache/zihaohe/omni_openh3_20261111')
prep = root / 'preparation/train_role_rphi_export_20261111_v1'
out = root / 'preparation/train_role_rphi_export_acceptance_20261111_v1'
run = root / 'runs/rphi_train_guides_legacy_smoke_20261111a'
sys.path.insert(0, str(prep / 'code'))
from export_rphi_train_guides import verify_cache, validate_train_contract, CHECKPOINT_SHA256, MODEL_CONFIG
from eval_rphi import sha_file, canonical_sha, causal_indices, sample_id, TASKS
from rphi_data import RphiDataset, prepare_observed

start = time.monotonic()
queue = json.loads((run / 'queue_status.json').read_text())
m = json.loads((run / 'guides/manifest.json').read_text())
prior = json.loads((prep / 'inspection.json').read_text())
assert queue['state'] == 'completed' and queue['card'] == 6
assert m['complete'] and m['rgb_export_complete'] and m['samples'] == 4
assert m['source_ids'] == ['2BdYaGuVAKo'] and m['role'] == m['split'] == 'train'
assert m['scope'] == 'engineering_training_subset' and m['condition_mode'] == 'observations_only'
assert m['model_config'] == MODEL_CONFIG and m['checkpoint_step'] == 3000
assert m['checkpoint_sha256'] == CHECKPOINT_SHA256 == sha_file(m['checkpoint'])
assert m['frozen_state_sha256_before'] == m['frozen_state_sha256_after'] and m['all_parameters_frozen']
assert m['model_forward_calls'] == 88 and m['frame_batch'] == 1
assert not m['vae_encoded'] and m['latent_path'] is None and m['vae_roundtrip_path'] is None
assert m['controls_included'] == [] and m['h3_sampling_calls'] == m['guide_training_steps'] == 0
for key in ['checkpoint_sha256', 'train_manifest_sha256', 'dev_manifest_sha256', 'source_isolation', 'implementation_sha256']:
    assert m[key] == prior[key], key
for name, digest in m['implementation_sha256'].items():
    assert sha_file(prep / name) == digest
train = RphiDataset(m['train_manifest'], verify_hashes=True, cache_bytes=128 * 1024**2)
dev = RphiDataset(m['dev_manifest'], verify_hashes=True, cache_bytes=0)
isolation = validate_train_contract({'schema': 'rphi.checkpoint.v1', 'step': 3000,
    'model_config': MODEL_CONFIG, 'training_contract': m['training_contract']},
    train.manifest_sha256, train.records, dev.manifest_sha256, dev.records)
assert isolation == m['source_isolation']
selected = {sample_id(r): (i, r) for i, r in enumerate(train.records) if r['source_id'] == '2BdYaGuVAKo'}
assert set(selected) == {r['sample_id'] for r in m['per_sample']}
assert {r['task'] for r in m['per_sample']} == set(TASKS)
rows = []
for row in m['per_sample']:
    index, record = selected[row['sample_id']]
    for key in ['source_id', 'source_sha256', 'target_key', 'task', 'split', 'target_pixel_sha256', 'observation_pixel_sha256']:
        assert row[key] == record[key], key
    assert row['role'] == 'train' and row['record_sha256'] == canonical_sha(record)
    assert row['checkpoint_sha256'] == CHECKPOINT_SHA256 and row['checkpoint_step'] == 3000
    assert row['frame_indices'] == list(range(22)) and row['history_indices'] == causal_indices(range(22), 3, 22).tolist()
    assert row['geometry'] == {'frames': 22, 'height': 480, 'width': 832, 'channels': 3, 'layout': 'THWC', 'dtype': 'float32',
        'observation_shape': ([22, 120, 208, 3] if row['task'] == 'sr' else [22, 480, 832, 3])}
    assert sha_file(run / 'guides' / row['provenance_json']) == row['provenance_file_sha256']
    sidecar = json.loads((run / 'guides' / row['provenance_json']).read_text())
    assert sidecar == {k: v for k, v in row.items() if k not in ['provenance_json', 'provenance_file_sha256']}
    cache = run / 'guides' / row['guide_npz']
    assert verify_cache(cache, row)
    loaded = train.load(index)
    assert sha_file(train.resolve_path(record['observation_npz'])) == row['observation_file_sha256']
    assert sha_file(train.resolve_path(record['target_npz'])) == row['target_file_sha256']
    expected = prepare_observed(loaded['observed'], loaded['target'].shape[1:3])
    assert expected.device.type == 'cpu'
    expected = expected.permute(0, 2, 3, 1).contiguous().numpy()
    with np.load(cache, allow_pickle=False) as archive:
        raw, draft, clipped = (archive[key] for key in ['rphi_raw_rgb', 'raw_draft', 'rphi_rgb'])
        assert raw.shape == draft.shape == clipped.shape == (22, 480, 832, 3)
        assert np.array_equal(draft, expected), 'raw draft differs from CPU prepare_observed'
        outside = (raw < 0) | (raw > 1)
        clipping = {'component_fraction': float(outside.mean()), 'component_count': int(outside.sum()),
                    'raw_min': float(raw.min()), 'raw_max': float(raw.max())}
        assert clipping == row['clipping']
        rows.append({key: row[key] for key in ['sample_id', 'task', 'source_id', 'source_sha256', 'target_key',
            'target_pixel_sha256', 'observation_pixel_sha256', 'guide_file_sha256', 'guide_file_bytes', 'arrays', 'geometry']})
        rows[-1].update(actual_npz_file_and_array_hash_verified=True,
            raw_draft_bitwise_equal_cpu_prepare_observed=True, clamp_bitwise_verified=True,
            clipping=clipping, target_in_npz=False, latent_in_npz=False)
    del raw, draft, clipped, expected, loaded
    print(json.dumps({'verified_task': row['task'], 'clipping_fraction': clipping['component_fraction']}), flush=True)

report = {'schema': 'rphi.train_guide_export.engineering_review.v1', 'complete': True, 'engineering_pass': True,
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'queue_completed_utc': queue['updated_at_utc'],
    'seconds': time.monotonic() - start, 'run': str(run), 'guide_manifest_sha256': sha_file(run / 'guides/manifest.json'),
    'queue_status_sha256': sha_file(run / 'queue_status.json'), 'checkpoint_sha256': CHECKPOINT_SHA256,
    'role': 'train', 'sources': ['2BdYaGuVAKo'], 'samples': 4, 'frames_per_sample': 22,
    'scope': 'engineering_training_subset', 'all_actual_npz_verified': True,
    'all_raw_drafts_match_observations': True, 'all_clamps_match_raw_prediction': True,
    'all_22_frame_K3_histories_verified': True, 'frozen_state_sha256': m['frozen_state_sha256_before'],
    'frozen_before_after_equal': True, 'all_parameters_frozen_export_assertion': m['all_parameters_frozen'],
    'source_isolation': isolation, 'prior_CPU_inspection_sha256': sha_file(prep / 'inspection.json'),
    'prior_CPU_contract_matches': True, 'controls_included': [], 'VAE_encoded': False, 'latent_cache_complete': False,
    'VAE_calls': 0, 'H3_calls': 0, 'training_steps': 0, 'audit_model_forwards': 0, 'audit_NPU_kernels': 0,
    'torch_npu_imported_in_audit': 'torch_npu' in sys.modules, 'per_sample': rows,
    'limitations': ['Engineering export acceptance only, not quality or held-out evidence.',
        'Original video SHA remains materialization attestation. Current source NPZ/receipt/file/pixel bindings reverified.',
        'Frozen hash assertion recorded during export, no model reload or repeat forward in audit.',
        'Only one training source exported. Other31 sources and all VAE guide/roundtrip encoding remain unexecuted.']}
with (out / 'engineering_review.json').open('x') as f:
    json.dump(report, f, indent=2)
files = [run / name for name in ['queue_status.json', 'launch_admission.json', 'logs/export.log',
    'guides/manifest.json', 'guides/inspection.json', 'guides/partial_manifest.json']]
files += sorted((run / 'guides/samples').glob('*.json')) + [out / 'engineering_review.json', Path(__file__)]
files = [p for p in files if p.is_file()]
checks = {str(p.relative_to(root)): sha_file(p) for p in files}
with (out / 'evidence_files.json').open('x') as f:
    json.dump(checks, f, indent=2)
archive = out / 'small_evidence.tar.gz'
with tarfile.open(archive, 'x:gz') as bundle:
    for path in files + [out / 'evidence_files.json']:
        bundle.add(path, arcname=str(path.relative_to(root)))
receipt = {'complete': True, 'archive': str(archive), 'archive_sha256': sha_file(archive),
    'archive_bytes': archive.stat().st_size, 'guide_manifest_sha256': report['guide_manifest_sha256'],
    'engineering_review_sha256': sha_file(out / 'engineering_review.json'), 'files': len(files)}
with (out / 'archive_receipt.json').open('x') as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt), flush=True)
