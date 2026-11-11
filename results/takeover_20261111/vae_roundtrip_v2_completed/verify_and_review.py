"""Read-only JSON evidence checks; no torch import, pixels, remote calls, or inference."""
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
def read(path):
    return json.loads(path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

archive = HERE / 'evidence.tar.gz'
assert sha(archive) == '10dba2752a61a7b88fea28b2fd826bb0e0763fb95d8ede721c341af51584d96a'
receipt = read(HERE / 'archive_receipt.json')
for rel, entry in receipt['files'].items():
    p = HERE / rel
    assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], rel
vae_path = HERE / 'runs/vae_roundtrip_legacy_smoke_20261111b/eval/metrics.json'
vae = read(vae_path)
queue = read(HERE / 'runs/vae_roundtrip_legacy_smoke_20261111b/queue_status.json')
assert vae['complete'] and queue['state'] == 'completed'
assert vae['contract']['samples'] == len(vae['per_sample']) == 12
assert vae['contract']['H3_sampler_calls'] == vae['contract']['NFE'] == 0
assert vae['contract']['roundtrip_budget'] == 24
assert vae['contract']['frame_stride'] == 1 and vae['contract']['geometry'] == [22,480,832,3]
assert vae['freeze_audit'] == {'VAE_parameter_versions_unchanged': True, 'tensors': 559}
assert vae['model_record']['vae_only']['dit_loaded'] is False
assert vae['model_record']['vae_only']['component_binding_restored_after_load']
runtime = read(HERE / 'preparation/vae_roundtrip_runtime_20261111_v2/runtime_manifest.json')
for name, digest in vae['implementation_sha256'].items():
    assert runtime['files']['code/' + name] == digest

prior = {}
manifests = {}
prior_paths = {}
for scope, directory, run in [
    ('ordinary','fixed_h3_legacy_smoke_completed','fixed_h3_rphi_legacy_smoke_20261111a'),
    ('controls','fixed_h3_controls_completed','fixed_h3_rphi_legacy_controls_smoke_20261111a'),
]:
    p = ROOT / directory / 'runs' / run
    metrics_path, manifest_path = p/'eval/metrics.json', p/'guides/manifest.json'
    m, g = read(metrics_path), read(manifest_path)
    assert m['complete'] and g['complete'] and g['checkpoint_step'] == 3000
    assert sha(metrics_path) == vae['contract']['input_hashes'][scope+'_metrics']
    assert sha(manifest_path) == vae['contract']['input_hashes'][scope+'_guides']
    assert m['contract']['dev_manifest_sha256'] == vae['contract']['input_hashes']['dev_manifest']
    assert m['contract']['frame_stride'] == 1 and m['contract']['alpha'] == .25
    for key in ('available','network','device','alexnet_sha256','downloads'):
        assert m['lpips'][key] == vae['lpips'][key]
    assert vae['lpips']['available'] and vae['lpips']['downloads'] is False
    prior[scope] = {s['sample_id']: s for s in m['per_sample']}
    manifests[scope] = {s['sample_id']: s for s in g['per_sample']}
    prior_paths[scope] = {'metrics':str(metrics_path.relative_to(ROOT)), 'sha256':sha(metrics_path)}

rows = []
quality_keys = ('psnr','psnr_from_mean_mse','mse','mae','lpips')
temporal_keys = ('unwarped_rgb_error_change_mae','flow_aligned_rgb_error_change_mae')
def compact(arm):
    return {**{k:arm['quality'][k] for k in quality_keys},
            **{k:arm['temporal'][k] for k in temporal_keys}}
for s in vae['per_sample']:
    scope, sample_id = s['scope'], s['sample_id']
    p, g = prior[scope][sample_id], manifests[scope][sample_id]
    for key in ('source_id','source_sha256','target_pixel_sha256','task'):
        assert s[key] == p[key] == g[key], (sample_id,key)
    assert s['guide_file_sha256'] == g['guide_file_sha256']
    assert s['frame_indices'] == p['frame_indices'] == g['frame_indices'] == list(range(22))
    assert g['history_indices'] == [[max(0,t-2),max(0,t-1),t] for t in range(22)]
    if scope == 'controls':
        assert s['control_recipe'] == p['control_recipe'] == g['control_recipe']
        assert s['control'] == p['control'] == g['control']
    assert s['baseline_metric_parity']
    for arm in ('input','rphi_only'):
        for key in quality_keys:
            tol = 1e-6 if key == 'lpips' else 1e-10
            assert abs(s['arms'][arm]['quality'][key] - p['arms'][arm]['quality'][key]) <= tol
        assert s['arms'][arm]['temporal'] == p['arms'][arm]['temporal']
    for label, obs in s['observations'].items():
        call = p['sampling']['calls'][label]
        assert obs['guide_pixel_sha256'] == p['sampling']['guide_pixel_sha256'][label]
        assert obs['encoded_latent_sha256_float32'] == call['guide_latent_sha256_float32']
        assert call['warm_start_latent_sha256_float32'] == p['sampling']['calls']['raw']['guide_latent_sha256_float32']
        assert obs['latent_matches_prior_H3_exactly'] and obs['clamp_parity_exact']
        assert obs['clamp_parity_max_abs_error'] == 0
        assert obs['encoded_latent_shape'] == [1,24,7,30,52]
        assert obs['encoded_latent_dtype'] == 'torch.float32'
        assert obs['captured_parts'] == 2 and obs['captured_frames_before_final_crop'] == 22
        rec = call['sampling_record']
        assert rec['seed'] == 101 and rec['steps'] == 3 and rec['audio_init'] == 'zeros'
        assert rec['correction_events'] == [] and rec['base_parameter_versions_unchanged']
        assert rec['pretrained']['manifest_sha256'] == vae['contract']['input_hashes']['pretrained_manifest']
    assert s['observations']['rphi']['guide_pixel_sha256'] == g['arrays']['rgb']['array_sha256']
    if scope == 'controls':
        assert s['observations']['raw']['guide_pixel_sha256'] == g['arrays']['input_rgb']['array_sha256']
    for arm in list(s['arms'].values()) + list(p['arms'].values()):
        q,t = arm['quality'],arm['temporal']
        assert q['frames'] == 22 and t['pairs'] == t['valid_pairs'] == 21
        assert t['valid_coverage'] == s['arms']['input']['temporal']['valid_coverage']
        assert all(math.isfinite(q[k]) for k in quality_keys)
        assert all(math.isfinite(t[k]) for k in temporal_keys)
    assert len(s['cache']['arrays']) == 4 and len(s['cache']['file_sha256']) == 64
    for a in s['cache']['arrays'].values():
        assert a['shape'] == [22,480,832,3] and a['dtype'] == 'float32' and len(a['array_sha256']) == 64
    arms = {k:compact(v) for k,v in s['arms'].items()}
    arms.update({k:compact(p['arms'][k]) for k in ('h3_raw','h3_raw_anchor','h3_rphi','h3_rphi_anchor')})
    deltas = {}
    for name,a,b in [('vae_rphi_minus_rphi','rphi_as_returned_internal_clipped','rphi_only'),
                      ('h3_rphi_minus_vae_rphi','h3_rphi','rphi_as_returned_internal_clipped'),
                      ('anchored_h3_rphi_minus_rphi','h3_rphi_anchor','rphi_only')]:
        deltas[name] = {k:arms[a][k]-arms[b][k] for k in arms[a]}
    rows.append({'scope':scope,'sample_id':sample_id,'source_id':s['source_id'],'task':s['task'],
                 'control':s['control'],'arms':arms,'paired_deltas':deltas,
                 'preclamp_diagnostics':{label:{k:o[k] for k in ('pre_internal_clamp_min','pre_internal_clamp_max','outside_01_fraction')}
                                        for label,o in s['observations'].items()},
                 'input_and_latent_parity_verified':True})

assert set((s['scope'],s['sample_id']) for s in vae['per_sample']) == set((scope,key) for scope,v in prior.items() for key in v)
assert len({s['source_id'] for s in vae['per_sample']}) == 1
identities = [s for s in vae['per_sample'] if s['control']=='identity']
near = [s for s in vae['per_sample'] if s['control']=='near_clean']
assert len(identities) == len(near) == 4
duplicate = {}
for label in ('raw','rphi'):
    duplicate[label+'_guide_unique'] = len({s['observations'][label]['guide_pixel_sha256'] for s in identities})
    duplicate[label+'_latent_unique'] = len({s['observations'][label]['encoded_latent_sha256_float32'] for s in identities})
    assert duplicate[label+'_guide_unique'] == duplicate[label+'_latent_unique'] == 1
for key in identities[0]['cache']['arrays']:
    duplicate[key+'_unique'] = len({s['cache']['arrays'][key]['array_sha256'] for s in identities})
    assert duplicate[key+'_unique'] == 1
assert len({s['observations']['raw']['guide_pixel_sha256'] for s in near}) == 4
assert len({s['control_recipe']['seed'] for s in near}) == 4

def aggregate(selected):
    return {arm:{key:{'mean':mean(r['arms'][arm][key] for r in selected),
                     'min':min(r['arms'][arm][key] for r in selected),
                     'max':max(r['arms'][arm][key] for r in selected)}
                 for key in selected[0]['arms'][arm]} for arm in selected[0]['arms']}
groups = {name:aggregate([r for r in rows if predicate(r)]) for name,predicate in [
    ('identity_native_anchor_policy',lambda r:r['control']=='identity' and r['task']!='sr'),
    ('identity_sr_anchor_policy',lambda r:r['control']=='identity' and r['task']=='sr'),
    ('near_clean_four_task_keyed_draws',lambda r:r['control']=='near_clean'),
    ('near_clean_three_draws_native_anchor_policy',lambda r:r['control']=='near_clean' and r['task']!='sr'),
    ('near_clean_sr_draw_anchor_policy',lambda r:r['control']=='near_clean' and r['task']=='sr')
]}
all_observations = [o for s in vae['per_sample'] for o in s['observations'].values()]
clip_psnr_deltas = [r['arms'][label+'_as_returned_internal_clipped']['psnr'] - r['arms'][label+'_pre_internal_clamp']['psnr']
                   for r in rows for label in ('raw','rphi')]
clipping_summary = {'pre_internal_clamp_min':min(o['pre_internal_clamp_min'] for o in all_observations),
                    'pre_internal_clamp_max':max(o['pre_internal_clamp_max'] for o in all_observations),
                    'outside_01_fraction_min':min(o['outside_01_fraction'] for o in all_observations),
                    'outside_01_fraction_max':max(o['outside_01_fraction'] for o in all_observations),
                    'clip_minus_preclamp_psnr_min':min(clip_psnr_deltas),
                    'clip_minus_preclamp_psnr_max':max(clip_psnr_deltas),
                    'interpretation':'Internal clipping has small positive PSNR effect in these 24 paths; it does not account for the large identity/near-clean roundtrip degradation.'}
launch = read(ROOT/'vae_roundtrip_preparation_20261111_v2/launch_receipt.json')
started = launch['initial_status']['status']['updated_at_utc']
ended = queue['updated_at_utc']
timing = {'started_at_utc':started,'completed_at_utc':ended,
          'wall_seconds':(datetime.fromisoformat(ended)-datetime.fromisoformat(started)).total_seconds(),
          'encode_seconds_sum':sum(o['encode_seconds'] for s in vae['per_sample'] for o in s['observations'].values()),
          'decode_seconds_sum':sum(o['decode_seconds'] for s in vae['per_sample'] for o in s['observations'].values()),
          'vae_load_seconds':vae['model_record']['vae_load_seconds'],
          'wall_scope':'includes integrity checks, CPU LPIPS/flow, NPZ storage, loading and NPU work; not pure inference throughput'}
out = {'schema':'openh3.vae_roundtrip_scientific_review.v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
       'complete':True,'engineering_pass':True,'archive':{'file':'evidence.tar.gz','bytes':archive.stat().st_size,'sha256':sha(archive),'verified_files':len(receipt['files'])},
       'metrics_sha256':sha(vae_path),'prior_evidence':prior_paths,'review_script_sha256':sha(Path(__file__)),
       'verification':{'samples':12,'sources':1,'guide_and_latent_sha_exact':24,'clamp_parity_exact_max_abs_zero':24,
                       'baseline_quality_and_temporal_parity':12,'full_frames':22,'temporal_pairs':21,
                       'flow_coverage':vae['per_sample'][0]['arms']['input']['temporal']['valid_coverage'],
                       'lpips':vae['lpips'],'freeze_audit':vae['freeze_audit'],'sampler_calls':0,'NFE':0,
                       'cache_array_receipts_validated':48,'cache_pixels_downloaded_or_recomputed_locally':False,
                       'identity_duplicate_counts':duplicate,'near_clean_unique_noise_inputs':4},
       'timing':timing,'per_sample':rows,'control_aggregates':groups,'clipping_summary':clipping_summary,
       'interpretation':[
           'The fixed VAE path substantially reduces fidelity of identity, near-clean and compression Rphi guides. It is not an identity transform even without H3 sampling.',
           'VAE(Rphi) is not uniformly harmful by every metric: SR PSNR slightly rises; deblur and denoise LPIPS improve while PSNR falls. Do not generalize this one source to all restoration settings.',
           'Compared with clipped VAE(Rphi), H3-Rphi has lower PSNR and higher MAE on all four ordinary tasks. LPIPS improves for SR/deblur but worsens for denoise/compression. All controls have worse PSNR, MAE and LPIPS after H3-Rphi than after VAE(Rphi).',
           'Compare H3-Rphi against clipped VAE(Rphi), since original H3.decode already internally clips. True pre-internal-clamp VAE outputs are separate diagnostics; raw LPIPS is outside its usual input domain where RGB exceeds [0,1].',
           'H3-Rphi uses the raw-input latent warm start plus Rphi conditioning. VAE(Rphi) directly decodes the Rphi guide latent; this paired path comparison does not isolate a diffusion-only causal effect.',
           'MSE differences can telescope algebraically but are not independent additive component contributions: residual cross terms and changed latent trajectories remain. No degradation-explained percentage or irreducible VAE lower bound is claimed.',
           'Identity is one unique source/input repeated under four task labels. Native and SR anchor policies differ; their anchors are reported separately. Near-clean has four task-keyed noise draws on that same source, not four independent sources.',
           'The clean reference is an original-recording proxy. No frozen-test generalization, end-to-end causal claim, learned compensation result or quality gate pass follows from this diagnostic.'
       ]}
save('scientific_review.json',out)
save('verification_receipt.json',{'complete':True,'archive_sha256':sha(archive),'metrics_sha256':sha(vae_path),
     'scientific_review_sha256':sha(HERE/'scientific_review.json'),'review_script_sha256':sha(Path(__file__)),
     'archive_files_verified':len(receipt['files']),'prior_metrics_and_manifests_verified':4,'new_model_calls':0})
print(json.dumps({'review_sha256':sha(HERE/'scientific_review.json'),'timing':timing,
    'table':[{**{k:r[k] for k in ('scope','task','control')},'scores':{a:{k:r['arms'][a][k] for k in ('psnr','lpips','mae','flow_aligned_rgb_error_change_mae')} for a in ('rphi_only','rphi_as_returned_internal_clipped','h3_rphi','h3_rphi_anchor')}} for r in rows]},indent=2))
