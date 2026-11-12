"""One fixed TRAIN compression row: two VAE encodes and one Rphi-guide decode.

Default mode is CPU input inspection only. --execute requires independent frozen
pins and an externally admitted, owned, bounded single-card process. No Rphi or
20B DiT construction, target model input, optimizer, dataset expansion or retry.
Official Lanczos/PIL preprocessing (including uint8 truncation) is preserved.
"""
import argparse
import ast
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from eval_rphi import canonical_sha, sample_id, causal_indices, sha_file
from export_rphi_train_guides import (CHECKPOINT_SHA256, MODEL_CONFIG, array_sha,
    validate_train_contract, verify_cache)
from rphi_data import RphiDataset, assert_disjoint, prepare_observed
from check_h3_trajectory_grad import atomic_json, verify_pins, runtime_imports, memory_record
from check_h3_joint_rgb_grad import (frozen_inventory, freeze_snapshot, check_snapshot,
    override_method, wrapper_snapshot, wrappers_restored)

SCHEMA = 'openh3.train_vae_cache.v1'
SOURCE_ID = '2BdYaGuVAKo'
SOURCE_SHA = 'ab1d437859fa6de7a25d2235bd4c144e319749bee0249f9556f1c708b09f9394'
SAMPLE_ID = 'rphi-2ac5b3e69cafa70f77d0c6603615ebc56162f4c6bec6e64664ca5bbb7e4b33a9'
RGB_SHAPE = (22,480,832,3)
LATENT_SHAPE = (1,24,7,30,52)
SOURCE_CONTRACT_SHA = '9c2e2bdc64fffec33a2e08c90e36affc87edc4c14ea3bb79467166da277ffdc4'
FIXED_SHA = {
    'checkpoint': CHECKPOINT_SHA256,
    'train_manifest': 'a32736578195bcc17f16fbfc3614ccc7f38689de18680ba1a38fe393515bb1cb',
    'dev_manifest': '6ae49698cd332db88606820d4d59d0221087e5979816f247b3992693864caeba',
    'ordinary_guides_manifest': '1c73a26f1cb81157c2445fac44d33acc925256ec77343f50aa7d54f7ab9f0008',
    'pretrained_manifest': '1023bd6e1df1552240cc7ff4bfac960836d2eeda05e680ecd2eb4b752bf7d975',
    'source_contract': SOURCE_CONTRACT_SHA,
}
PREPROCESS_SOURCE_SHA = {
    'comfy/utils.py': '9320cf5c624e96acc1d9ec92ff120ed6387a2142707f3194cb21dd0d11ac2500',
    'comfy_extras/nodes_minimax_h3.py': '108ceff11e5a5c9a34a57bbc380447cf9941beb673a1bd421abc818de54ccd5c',
}


def checked_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root/path).resolve()


def check_contract(path):
    contract = json.loads(Path(path).read_text())
    if contract.get('schema') != 'openh3.train_vae_cache_execution.v1':
        raise ValueError('wrong execution schema')
    if contract.get('entry_sha256') != sha_file(__file__): raise ValueError('entrypoint not pinned')
    if contract.get('selected_sample_id') != SAMPLE_ID: raise ValueError('fixed sample identity differs')
    root = Path(contract['root']).resolve()
    verified = verify_pins(root, contract['implementation_sha256'])
    paths = {}
    for key, expected in FIXED_SHA.items():
        paths[key] = checked_path(root, contract[key])
        field = 'manifest_sha256' if key == 'pretrained_manifest' else key+'_sha256'
        if contract.get(field) != expected or sha_file(paths[key]) != expected:
            raise ValueError('fixed input SHA mismatch: '+key)
    comfy = checked_path(root, contract['comfy_root'])
    for relative, expected in PREPROCESS_SOURCE_SHA.items():
        file = (comfy/relative).resolve()
        if sha_file(file) != expected or verified.get(str(file)) != expected:
            raise ValueError('official preprocessing source not pinned: '+relative)
    return contract, paths, comfy, verified


def selected_row(manifest, record):
    """Only the accepted ordinary TRAIN row, never controls/dev or invented metadata."""
    if (manifest.get('schema') != 'rphi.train_rgb_guide_export.v1' or not manifest.get('complete')
        or manifest.get('role') != 'train' or manifest.get('split') != 'train'
        or manifest.get('samples') != 4 or manifest.get('source_ids') != [SOURCE_ID]
        or manifest.get('checkpoint_sha256') != CHECKPOINT_SHA256 or manifest.get('checkpoint_step') != 3000
        or manifest.get('model_config') != MODEL_CONFIG or manifest.get('vae_encoded') is not False):
        raise ValueError('wrong accepted train guide manifest')
    if (record['source_id'] != SOURCE_ID or record['source_sha256'] != SOURCE_SHA
        or record['split'] != 'train' or record['task'] != 'compression' or sample_id(record) != SAMPLE_ID):
        raise ValueError('fixed train compression record mismatch')
    rows = [r for r in manifest['per_sample'] if r['sample_id'] == SAMPLE_ID]
    if len(rows) != 1: raise ValueError('missing/duplicate selected guide row')
    row = rows[0]
    fields = ('source_id','source_sha256','target_key','target_pixel_sha256','observation_pixel_sha256','task','split')
    if any(row.get(k) != record[k] for k in fields): raise ValueError('guide/source/target binding mismatch')
    if (row.get('role') != 'train' or row.get('control') is not None
        or row.get('record_sha256') != canonical_sha(record)
        or row.get('checkpoint_sha256') != CHECKPOINT_SHA256
        or row.get('condition_mode') != 'observations_only'
        or row.get('target_role') != 'supervision_reference_only_never_model_input'
        or row.get('frame_indices') != list(range(22))
        or row.get('history_indices') != causal_indices(range(22),3,22).tolist()):
        raise ValueError('guide record/role/history contract mismatch')
    for key in ('train_manifest_sha256','dev_manifest_sha256'):
        if row.get(key) != manifest.get(key) or row.get(key) != FIXED_SHA[key.removesuffix('_sha256')]:
            raise ValueError('full train/dev manifest binding mismatch')
    return row


def extract_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    matches = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name]
    if len(matches) != 1 or matches[0].decorator_list: raise ValueError('unexpected official helper definition')
    exec(compile(ast.Module(body=matches,type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def official_preprocess(pixels, comfy_root):
    """Execute exactly the pinned pure helper bodies on CPU, without importing Comfy."""
    import torch
    import PIL
    from PIL import Image
    for relative, expected in PREPROCESS_SOURCE_SHA.items():
        if sha_file(Path(comfy_root)/relative) != expected: raise ValueError('preprocess source SHA changed')
    if pixels.dtype != np.float32 or pixels.ndim != 4 or pixels.shape[-1] != 3 or not np.isfinite(pixels).all():
        raise ValueError('expected finite float32 THWC')
    if pixels.min() < 0 or pixels.max() > 1: raise ValueError('P requires existing clipped/normalized input')
    uint8_hasher, calls = hashlib.sha256(), []
    class ImageProxy:
        Resampling = Image.Resampling
        @staticmethod
        def fromarray(value):
            if value.dtype != np.uint8 or value.shape != pixels.shape[1:]:
                raise ValueError('unexpected actual PIL input')
            uint8_hasher.update(np.ascontiguousarray(value).tobytes())
            calls.append(list(value.shape))
            return Image.fromarray(value)
    ns = {'np':np,'torch':torch,'Image':ImageProxy}
    extract_function(Path(comfy_root)/'comfy/utils.py','lanczos',ns)
    upscale = extract_function(Path(comfy_root)/'comfy/utils.py','common_upscale',ns)
    resize = extract_function(Path(comfy_root)/'comfy_extras/nodes_minimax_h3.py','_resize',
        {'comfy':SimpleNamespace(utils=SimpleNamespace(common_upscale=upscale))})
    before = array_sha(pixels)
    result = resize(torch.from_numpy(pixels.copy()),pixels.shape[2],pixels.shape[1],'center').contiguous().numpy()
    if array_sha(pixels) != before or result.shape != pixels.shape or result.dtype != np.float32 or len(calls) != len(pixels):
        raise ValueError('preprocessing mutated input/geometry/dtype/frame count')
    delta = result.astype(np.float64)-pixels.astype(np.float64)
    proof = {'before_P_float32_sha256':before,'PIL_uint8_sha256':uint8_hasher.hexdigest(),
        'after_P_float32_sha256':array_sha(result),'shape':list(result.shape),'dtype':'float32',
        'PIL_frames':len(calls),'quantization':'clip(255*x,0,255).astype(uint8), truncation, not round',
        'resize':'official same-size Lanczos center; no custom bypass',
        'changed_components':int(np.count_nonzero(result != pixels)),
        'max_abs_difference':float(np.max(np.abs(delta))),'mse_difference':float(np.mean(delta*delta)),
        'versions':{'torch':torch.__version__,'numpy':np.__version__,'Pillow':PIL.__version__},
        'source_sha256':dict(PREPROCESS_SOURCE_SHA)}
    return result, proof


def inspect_inputs(paths, comfy_root):
    """Complete32/8 source metadata + selected current NPZ/receipt, no model construction."""
    import torch
    train = RphiDataset(paths['train_manifest'],verify_hashes=True,cache_bytes=0)
    dev = RphiDataset(paths['dev_manifest'],verify_hashes=True,cache_bytes=0)
    payload = torch.load(paths['checkpoint'],map_location='cpu',weights_only=True)
    isolation = validate_train_contract(payload,train.manifest_sha256,train.records,dev.manifest_sha256,dev.records)
    assert_disjoint(train,dev)
    manifest = json.loads(paths['ordinary_guides_manifest'].read_text())
    indices = [i for i,r in enumerate(train.records) if r['source_id']==SOURCE_ID and r['task']=='compression']
    if len(indices) != 1: raise ValueError('fixed train selection missing or duplicate')
    record = train.records[indices[0]]
    row = selected_row(manifest,record)
    # Recompute CPU checkpoint state without constructing RphiModel.
    state_sha = canonical_sha({key:{'shape':list(value.shape),'dtype':str(value.dtype),
        'sha256':array_sha(value.detach().cpu().numpy())} for key,value in sorted(payload['state_dict'].items())})
    if manifest.get('frozen_state_sha256_before') != state_sha or manifest.get('frozen_state_sha256_after') != state_sha:
        raise ValueError('frozen cached Rphi state differs from actual checkpoint')
    for name in ('rphi_data.py','rphi_model.py'):
        if payload['implementation_sha256'].get('code/'+name) != sha_file(Path(__file__).parent/name):
            raise ValueError('checkpoint source differs from cached exporter dependency')
    del payload
    base = paths['ordinary_guides_manifest'].parent.resolve()
    def contained(relative):
        p=(base/relative).resolve()
        if not p.is_relative_to(base) or not p.is_file(): raise ValueError('cache path outside guide root')
        return p
    guide = contained(row['guide_npz'])
    sidecar = contained(row['provenance_json'])
    if sha_file(sidecar) != row['provenance_file_sha256']: raise ValueError('guide sidecar SHA mismatch')
    side = json.loads(sidecar.read_text())
    if side != {k:v for k,v in row.items() if k not in ('provenance_json','provenance_file_sha256')}:
        raise ValueError('manifest/sidecar binding mismatch')
    verify_cache(guide,row)
    loaded = train.load(indices[0])
    if tuple(loaded['target'].shape) != RGB_SHAPE or tuple(loaded['observed'].shape) != RGB_SHAPE:
        raise ValueError('fixed native22x480x832 geometry required')
    for prefix in ('target','observation'):
        p=train.resolve_path(record[prefix+'_npz'])
        if str(p) != row[prefix+'_npz'] or sha_file(p) != row[prefix+'_file_sha256']:
            raise ValueError('materialized '+prefix+' file binding mismatch')
    with np.load(guide,allow_pickle=False) as cache:
        raw=np.ascontiguousarray(cache['raw_draft']);rphi=np.ascontiguousarray(cache['rphi_rgb'])
        if not np.array_equal(cache['rphi_rgb'],np.clip(cache['rphi_raw_rgb'],0,1)):
            raise ValueError('Rphi clip relation changed')
    observed = prepare_observed(loaded['observed'],RGB_SHAPE[1:3]).permute(0,2,3,1).contiguous().numpy()
    if not np.array_equal(raw,observed): raise ValueError('raw warmstart pixels differ from observation')
    if any(tuple(x.shape) != RGB_SHAPE for x in (raw,rphi)): raise ValueError('cache RGB shape mismatch')
    preprocess={}
    for label,value in [('raw',raw),('rphi',rphi)]:
        _,preprocess[label]=official_preprocess(value,comfy_root)
    evidence = {'schema':'openh3.train_vae_cache_input_inspection.v1','complete':True,
        'role':'train','source_id':SOURCE_ID,'source_sha256':SOURCE_SHA,'task':'compression',
        'sample_id':SAMPLE_ID,'record':record,'record_sha256':canonical_sha(record),'export_row':row,
        'source_isolation':isolation,'Rphi_checkpoint_state_sha256':state_sha,
        'input_files':{str(p):sha_file(p) for p in [paths[k] for k in FIXED_SHA] if p.is_file()},
        'selected_asset_files':{str(guide):sha_file(guide),str(sidecar):sha_file(sidecar),
            **{str(train.resolve_path(record[k+'_npz'])):row[k+'_file_sha256'] for k in ('target','observation')},
            **train.receipt_hashes},
        'preprocessing':preprocess,'geometry':list(RGB_SHAPE),'expected_latent_shape':list(LATENT_SHAPE),
        'target_usage':'selected target loaded only for receipt/pixel/geometry integrity; never returned as a model input',
        'target_in_model_input':False,'model_input_arrays':['raw_draft','rphi_rgb'],
        'original_source_verification':'materialization_attestation_only; no live original video SHA claim',
        'full_manifest_scope':'complete32train/8dev source metadata and checkpoint contract; current pixel/file verification is selected compression row only',
        'model_constructed':False,'VAE_calls':0,'H3_calls':0,'Rphi_calls':0}
    # Inputs can change during CPU decoding/PIL work; fail before model load.
    for p,digest in {**evidence['input_files'],**evidence['selected_asset_files']}.items():
        if sha_file(p)!=digest: raise ValueError('input changed during inspection: '+p)
    return evidence,raw,rphi


class Evidence:
    def __init__(self,out,contract,inspection):
        self.path=Path(out);self.path.mkdir(parents=True,exist_ok=False)
        (self.path/'arrays').mkdir();self.started=time.monotonic()
        self.data={'schema':SCHEMA,'complete':False,'contract':contract,'inspection':inspection,
            'phase':'created','events':[],'arrays':{},'observed_preprocessing':{},
            'actual':{'encode_calls':0,'decode_calls':0,'stage_encode_calls':0,'stage_decode_calls':0,
                'PIL_fromarray_calls':{},'forbidden_calls':[],'Rphi_calls':0,'H3_sampler_calls':0,
                'NFE':0,'backward_calls':0,'optimizer_steps':0}}
        self.save()
    def save(self):
        self.data['elapsed_seconds']=time.monotonic()-self.started
        atomic_json(self.path/'partial.json',self.data)
    def phase(self,label,torch=None):
        self.data['phase']=label
        row={'phase':label,'elapsed_seconds':time.monotonic()-self.started}
        if torch is not None:row['memory']=memory_record(torch)
        self.data['events'].append(row);self.save();print(json.dumps(row),flush=True)


def require(value,message,e):
    e.save()
    if not value:raise RuntimeError(message)


def save_array(e,name,tensor):
    value=tensor.detach().cpu().contiguous().numpy()
    p=e.path/'arrays'/(name+'.npy');np.save(p,value,allow_pickle=False)
    row={'path':str(p.relative_to(e.path)),'file_sha256':sha_file(p),'array_sha256':array_sha(value),
         'shape':list(value.shape),'dtype':str(value.dtype),'bytes':p.stat().st_size}
    e.data['arrays'][name]=row;e.save();return row


@contextmanager
def bounded_calls(h3,e,torch,expected_preprocessing):
    """Observe exact deployed P and VAE inputs; preserve native internal tile/chunk paths."""
    from PIL import Image
    vae,stage=h3.vae,h3.vae.first_stage_model
    actual=e.data['actual'];uint8_hashers={}
    with ExitStack() as scope:
        for owner,names in [(h3,('sample','encode','encode_guide','ref_block')),
            (vae,('encode_tiled','encode_tiled_','encode_tiled_1d','encode_tiled_3d','_encode_tiled_owned',
                  'decode_tiled','decode_tiled_','decode_tiled_1d','decode_tiled_3d','_decode_tiled_owned'))]:
            for name in names:
                if not hasattr(owner,name):continue
                def forbidden(*args,_name=name,**kwargs):
                    actual['forbidden_calls'].append(_name);e.save()
                    raise RuntimeError('undeclared path forbidden: '+_name)
                scope.enter_context(override_method(owner,name,forbidden))
        original_image=Image.fromarray
        def image_fromarray(value,*args,**kwargs):
            phase=e.data['phase'];label=phase.removeprefix('encode_')
            if phase not in ('encode_raw','encode_rphi') or value.dtype!=np.uint8:
                raise RuntimeError('unexpected PIL/model preprocessing')
            require(list(value.shape)==expected_preprocessing[label]['shape'][1:],'PIL frame geometry changed',e)
            actual['PIL_fromarray_calls'][label]=actual['PIL_fromarray_calls'].get(label,0)+1
            require(actual['PIL_fromarray_calls'][label]<=22,'extra PIL frame processing',e)
            uint8_hashers.setdefault(label,hashlib.sha256()).update(np.ascontiguousarray(value).tobytes())
            return original_image(value,*args,**kwargs)
        scope.enter_context(override_method(Image,'fromarray',image_fromarray))
        original_encode=vae.encode
        def encode(pixels,*args,**kwargs):
            phase=e.data['phase'];label=phase.removeprefix('encode_')
            actual['encode_calls']+=1
            require(phase in ('encode_raw','encode_rphi') and actual['encode_calls']<=2,'encode budget/phase',e)
            spec=expected_preprocessing[label]
            value=pixels.detach().cpu().contiguous().numpy()
            row={'actual_vae_input_sha256':array_sha(value),'dtype':str(value.dtype),'shape':list(value.shape),
                'actual_PIL_uint8_sha256':uint8_hashers[label].hexdigest(),
                'PIL_calls':actual['PIL_fromarray_calls'][label],
                'matches_CPU_preprocessing':array_sha(value)==spec['after_P_float32_sha256'] and
                    uint8_hashers[label].hexdigest()==spec['PIL_uint8_sha256']}
            e.data['observed_preprocessing'][label]=row
            require(row['matches_CPU_preprocessing'] and value.dtype==np.float32 and row['shape']==spec['shape']
                    and row['PIL_calls']==22,'actual deployed preprocessing differs from CPU inspection',e)
            return original_encode(pixels,*args,**kwargs)
        scope.enter_context(override_method(vae,'encode',encode))
        original_decode=vae.decode
        def decode(latent,*args,**kwargs):
            actual['decode_calls']+=1
            row={'array_sha256':array_sha(latent.detach().cpu().contiguous().numpy()),
                 'shape':list(latent.shape),'dtype':str(latent.dtype)}
            e.data['actual_decode_input']=row
            require(e.data['phase']=='decode_rphi' and actual['decode_calls']==1,'decode budget/phase',e)
            require(row['array_sha256']==e.data['arrays']['rphi_guide_latent']['array_sha256'],
                    'decode input is not exactly saved Rphi latent',e)
            return original_decode(latent,*args,**kwargs)
        scope.enter_context(override_method(vae,'decode',decode))
        for method in ('encode','decode'):
            original=getattr(stage,method)
            def stage_call(value,*args,_method=method,_original=original,**kwargs):
                actual['stage_'+_method+'_calls']+=1
                key=e.data['phase']
                entries=e.data.setdefault('stage_inputs',{})
                if key in entries:raise RuntimeError('whole VAE stage retried')
                entries[key]={'shape':list(value.shape),'dtype':str(value.dtype),
                    'array_sha256':array_sha(value.detach().cpu().contiguous().numpy())}
                require(actual['stage_'+_method+'_calls']<=(2 if _method=='encode' else 1),'stage call budget',e)
                return _original(value,*args,**kwargs)
            scope.enter_context(override_method(stage,method,stage_call))
        yield


def encode_pair(h3,raw_pixels,rphi_pixels,e,torch,expected_preprocessing):
    """The numerical boundary accepts exactly two observed RGB arrays, never target."""
    inputs={'raw':raw_pixels,'rphi':rphi_pixels}
    shape=raw_pixels.shape
    if shape!=rphi_pixels.shape or len(shape)!=4 or shape[0]!=22 or shape[-1]!=3 or shape[1]%32 or shape[2]%32:
        raise ValueError('native full22 geometry required')
    expected=(1,24,7,shape[1]//16,shape[2]//16)
    latents={}
    for label,array in inputs.items():
        if array.dtype!=np.float32 or not np.isfinite(array).all() or array.min()<0 or array.max()>1:
            raise ValueError('only finite existing clipped float32 observed arrays accepted')
        require(array_sha(array)==expected_preprocessing[label]['before_P_float32_sha256'],'input pixels changed',e)
        e.phase('encode_'+label,torch if hasattr(torch,'npu') else None)
        latent=h3.encode_guide_pixels(torch.from_numpy(array.copy()),list(shape[1:3]))
        name='raw_warm_start_latent' if label=='raw' else 'rphi_guide_latent'
        save_array(e,name,latent)
        require(tuple(latent.shape)==expected and latent.dtype==torch.float32 and not latent.requires_grad
                and bool(torch.isfinite(latent).all()),'encoded native latent invalid',e)
        require(array_sha(array)==expected_preprocessing[label]['before_P_float32_sha256'],'encoder mutated input',e)
        latents[label]=latent
    e.phase('decode_rphi',torch if hasattr(torch,'npu') else None)
    pixels=h3.decode(latents['rphi'].clone())
    save_array(e,'vae_rphi_as_returned_internal_clipped',pixels)
    require(tuple(pixels.shape)==shape and pixels.dtype==torch.float32 and not pixels.requires_grad
        and bool(torch.isfinite(pixels).all()) and float(pixels.min())>=0 and float(pixels.max())<=1,
        'as-returned VAE RGB invalid; no remedial clamp permitted',e)
    for label,latent in latents.items():
        name='raw_warm_start_latent' if label=='raw' else 'rphi_guide_latent'
        require(array_sha(latent.detach().cpu().contiguous().numpy())==e.data['arrays'][name]['array_sha256'],
                'encoded latent mutated during decode',e)
    return latents,pixels


def execute(paths,pretrained,raw,rphi,e,verified):
    import torch
    import torch_npu
    from run_vae_roundtrip_diagnostic import load_vae_only
    e.phase('load_VAE_only',torch)
    h3=load_vae_only(pretrained)
    require(h3.patcher is None and not hasattr(h3,'frozen'),'DiT unexpectedly loaded',e)
    vae=h3.vae;stage=vae.first_stage_model
    vae.device=vae.offload_device=h3.dev;vae.disable_offload=True
    stage.eval().requires_grad_(False).to(h3.dev)
    import comfy.model_management as mm
    mm.load_models_gpu([vae.patcher],force_full_load=True)
    require(vae.vae_dtype==torch.float16,'VAE dtype differs from deployed FP16',e)
    require(all(v.device==torch.device(h3.dev) for v in list(stage.parameters())+list(stage.buffers())),
            'VAE not fully resident',e)
    e.data['pretrained']=h3.record
    e.data['asset_verification']={'VAE_and_context_actual_SHA':True,'other_recipe_assets':'sizes_only; not loaded or rehashed',
        'DiT_loaded':False,'Rphi_model_constructed':False,'Rphi_checkpoint_deserialized_on_CPU':True,'actual_vae_dtype':str(vae.vae_dtype)}
    e.data['environment']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,
        'physical_visible_card':os.environ['ASCEND_RT_VISIBLE_DEVICES'],'logical_device':str(h3.dev)}
    math_before={'in_training':bool(mm.in_training),'jit_compile_false':bool(torch.npu.is_jit_compile_false()),
        'matmul_allow_hf32':bool(torch.npu.matmul.allow_hf32),'conv_allow_hf32':bool(torch.npu.conv.allow_hf32)}
    e.data['math_before']=math_before
    require(not math_before['in_training'] and math_before['jit_compile_false'] and not math_before['matmul_allow_hf32']
        and not math_before['conv_allow_hf32'],'deployment inference math mismatch',e)
    e.data['native_geometry_plan']={'tiling':stage.tiling,'tile_size':stage.tile_size,'vae_ratio':stage.vae_ratio,
        'vae_ratio_t':stage.vae_ratio_t,'decode_temporal_pad_and_chunks':list(stage._decode_temporal_chunks(7)),
        'height_tile_plan':stage.split_tiles(480),'width_tile_plan':stage.split_tiles(832),
        'output_shape':list(stage.decode_output_shape(LATENT_SHAPE)),
        'policy_changed':False,'latent_scale':'native normalized (mean-latents_mean)/latents_std; no extra external scale'}
    e.data['actual_imports_before']=runtime_imports(verified)
    require(e.data['actual_imports_before']['passed'],'unbound imported runtime',e)
    e.phase('frozen_before_hash',torch)
    initial=frozen_inventory(stage,torch);snapshot=freeze_snapshot(stage,torch)
    atomic_json(e.path/'frozen_initial.json',initial)
    e.data['frozen_initial']={'path':'frozen_initial.json','sha256':sha_file(e.path/'frozen_initial.json')}
    methods=[(vae,n) for n in ('encode','decode','encode_tiled','encode_tiled_','encode_tiled_1d','encode_tiled_3d','_encode_tiled_owned',
        'decode_tiled','decode_tiled_','decode_tiled_1d','decode_tiled_3d','_decode_tiled_owned') if hasattr(vae,n)]
    from PIL import Image
    methods += [(Image,'fromarray'),(stage,'encode'),(stage,'decode')]
    methods += [(h3,n) for n in ('sample','encode','encode_guide','ref_block') if hasattr(h3,n)]
    original=wrapper_snapshot(methods)
    torch.npu.reset_peak_memory_stats()
    try:
        with bounded_calls(h3,e,torch,e.data['inspection']['preprocessing']):
            encode_pair(h3,raw,rphi,e,torch,e.data['inspection']['preprocessing'])
    finally:
        if sys.exc_info()[0] is not None:e.data['failure_origin_phase']=e.data['phase']
        e.data['wrappers_restored']=wrappers_restored(original)
        e.data['math_after']={'in_training':bool(mm.in_training),'jit_compile_false':bool(torch.npu.is_jit_compile_false()),
            'matmul_allow_hf32':bool(torch.npu.matmul.allow_hf32),'conv_allow_hf32':bool(torch.npu.conv.allow_hf32)}
        e.data['math_restored']=e.data['math_after']==math_before
        e.phase('frozen_final_hash',torch)
        for key,fn in [('frozen_check',lambda:check_snapshot(stage,snapshot,torch,device=h3.dev)),
                       ('frozen_final',lambda:frozen_inventory(stage,torch)),
                       ('actual_imports',lambda:runtime_imports(verified))]:
            try:
                value=fn()
                if key=='frozen_final':
                    atomic_json(e.path/'frozen_final.json',value)
                    e.data[key]={'path':'frozen_final.json','sha256':sha_file(e.path/'frozen_final.json'),
                        'parameter_bytes_unchanged':value['parameters_sha256']==initial['parameters_sha256'],
                        'registered_buffer_bytes_unchanged':value['registered_buffers_sha256']==initial['registered_buffers_sha256']}
                else:e.data[key]=value
            except BaseException as error:e.data[key+'_error']=repr(error)
        e.save()
    actual=e.data['actual']
    require(actual['encode_calls']==actual['stage_encode_calls']==2 and actual['decode_calls']==actual['stage_decode_calls']==1
        and actual['PIL_fromarray_calls']=={'raw':22,'rphi':22} and not actual['forbidden_calls'],'fixed2encode1decode budget failed',e)
    require(e.data['wrappers_restored'] and e.data['math_restored'] and e.data.get('frozen_check',{}).get('passed')
        and e.data.get('actual_imports',{}).get('passed')
        and e.data.get('frozen_final',{}).get('parameter_bytes_unchanged')
        and e.data.get('frozen_final',{}).get('registered_buffer_bytes_unchanged')
        and 'frozen_final_error' not in e.data
        and sha_file(e.path/e.data['frozen_final']['path'])==e.data['frozen_final']['sha256'],
        'runtime/frozen/restoration gate failed',e)
    for p,digest in {**e.data['inspection']['input_files'],**e.data['inspection']['selected_asset_files']}.items():
        require(sha_file(p)==digest,'input changed after export: '+p,e)
    e.data.update(complete=True,scope='single TRAIN compression engineering cache; no training/quality/generalization proof',
        warm_start_role='raw_warm_start_latent fixed for both future direct and compensation arms',
        compensation_R='existing float32 clipped rphi_rgb BEFORE deployed P uint8 preprocessing',
        target_model_input=False,target_encode_calls=0)
    e.phase('completed',torch)
    atomic_json(e.path/'manifest.json',e.data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-contract',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.out.exists() or args.out.is_symlink():raise FileExistsError(args.out)
    contract,paths,comfy,verified=check_contract(args.execution_contract)
    proof,raw,rphi=inspect_inputs(paths,comfy)
    from openh3_assets import read_manifest
    pretrained=read_manifest(paths['pretrained_manifest'],verify=False)
    proof['pretrained_fingerprint']=pretrained['_fingerprint']
    proof['asset_inspection']='all recipe asset sizes only; no DiT bytes read; VAE/context actualSHA only on execute'
    proof['verified_runtime_files']=len(verified)
    proof['execution_contract_sha256']=sha_file(args.execution_contract)
    if not args.execute:
        args.out.mkdir(parents=True,exist_ok=False)
        atomic_json(args.out/'inspection.json',proof)
        print(json.dumps({'inspection_complete':True,'sample_id':SAMPLE_ID,'model_calls':0,
            'postP_sha256':{k:v['after_P_float32_sha256'] for k,v in proof['preprocessing'].items()}}),flush=True)
        return
    if not os.environ.get('ASCEND_RT_VISIBLE_DEVICES','').isdecimal():
        raise ValueError('explicit externally admitted single physical NPU required')
    e=Evidence(args.out,{**contract,'execution_contract_sha256':sha_file(args.execution_contract)},proof)
    try:execute(paths,pretrained,raw,rphi,e,verified)
    except BaseException as error:
        e.data['complete']=False
        e.data['failure']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
            'phase':e.data.get('failure_origin_phase',e.data['phase'])}
        e.save();atomic_json(e.path/'failure.json',e.data);raise


if __name__=='__main__':main()
