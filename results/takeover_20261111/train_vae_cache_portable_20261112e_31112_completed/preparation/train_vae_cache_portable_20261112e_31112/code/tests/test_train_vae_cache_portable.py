"""Used-asset admission and unchanged-exporter delegation; CPU only."""
import builtins
import ast
import copy
from contextlib import contextmanager
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import openh3_assets as all_assets
import openh3_vae_only_assets as assets
import export_train_vae_cache_portable as portable


class PortableAssets(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.manifest={"schema_version":1,"layout":"comfyui_minimax_h3","recipe":"refiner-turbo-comfy"}
        self.files={}
        for role in ('base','vae','context','refiner','turbo'):
            path=self.root/(role+'.bin');path.write_bytes((role*3).encode());self.files[role]=path
            record={'path':path.name,'bytes':path.stat().st_size,'sha256':all_assets.sha256(path)}
            if role in ('base','vae','context'): self.manifest[role]=record
            else: self.manifest.setdefault('components',[]).append({'role':role,**record})
        self.path=self.root/'manifest.json';self.write_manifest()
    def tearDown(self): self.tmp.cleanup()
    def write_manifest(self):
        self.path.write_text(json.dumps(self.manifest))
        return all_assets.sha256(self.path)
    def read(self):
        return assets.read_vae_only_manifest(self.path,expected_manifest_sha256=all_assets.sha256(self.path))

    def test_parent_fingerprint_identical_even_with_unused_files_absent(self):
        full=all_assets.read_manifest(self.path,verify=True)
        for role in ('base','refiner','turbo'): self.files[role].unlink()
        used=self.read()
        self.assertEqual(used['_fingerprint'],full['_fingerprint'])
        self.assertEqual(used['_manifest_sha256'],full['_manifest_sha256'])
        self.assertEqual(set(used['_paths']),{'vae','context'})
        self.assertFalse(used['_files_sha256_verified'])
        self.assertTrue(used['_used_files_sha256_verified'])
        self.assertFalse(used['_verification_scope']['safe_for_DiT_pretrained_merge'])
        self.assertEqual(used['_verification_scope']['unchecked_roles'],['base','refiner','turbo'])

    def test_unused_paths_never_resolved_statted_or_opened(self):
        unused={'base.bin','refiner.bin','turbo.bin'}
        original_stat,original_resolve=Path.stat,Path.resolve
        original_open,original_io_open=builtins.open,io.open
        def guard_path(path):
            if isinstance(path,(str,Path)) and Path(path).name in unused:
                raise AssertionError('unused asset filesystem access: '+str(path))
        def stat(path,*a,**kw): guard_path(path);return original_stat(path,*a,**kw)
        def resolve(path,*a,**kw): guard_path(path);return original_resolve(path,*a,**kw)
        def open_builtin(path,*a,**kw):guard_path(path);return original_open(path,*a,**kw)
        def open_io(path,*a,**kw):guard_path(path);return original_io_open(path,*a,**kw)
        with patch.object(Path,'stat',stat),patch.object(Path,'resolve',resolve),\
             patch.object(builtins,'open',open_builtin),patch.object(io,'open',open_io):
            used=self.read()
        self.assertEqual(set(used['_verification_scope']['checked']),{'vae','context'})

    def test_actual_used_bytes_size_and_manifest_SHA_fail_closed(self):
        original=self.files['vae'].read_bytes()
        self.files['vae'].write_bytes(b'x'*len(original))
        with self.assertRaisesRegex(ValueError,'actual SHA'):self.read()
        self.files['vae'].write_bytes(original+b'x')
        with self.assertRaisesRegex(ValueError,'size mismatch'):self.read()
        self.files['vae'].write_bytes(original)
        self.files['context'].unlink()
        with self.assertRaisesRegex(ValueError,'missing/size'):self.read()
        with self.assertRaisesRegex(ValueError,'parent pretrained manifest SHA'):
            assets.read_vae_only_manifest(self.path,expected_manifest_sha256='0'*64)

    def test_unused_metadata_still_validated_schema_order_digest_size(self):
        good=copy.deepcopy(self.manifest)
        cases=[('recipe',lambda m:m.update(recipe='unknown')),
            ('order',lambda m:m['components'].reverse()),
            ('SHA',lambda m:m['components'][0].update(sha256='bad')),
            ('size',lambda m:m['base'].update(bytes=0)),
            ('layout',lambda m:m.update(layout='diffusers'))]
        for message,change in cases:
            self.manifest=copy.deepcopy(good);change(self.manifest);self.write_manifest()
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):self.read()

    def test_truthful_scope_written_before_every_partial(self):
        used=self.read();proof={'component_verification':used['_verification_scope']}
        evidence=portable.Evidence(self.root/'evidence',{},proof)
        evidence.data['asset_verification']={'VAE_and_context_actual_SHA':True,
            'other_recipe_assets':'sizes_only; not loaded or rehashed','DiT_loaded':False}
        evidence.save()
        partial=json.loads((evidence.path/'partial.json').read_text())
        self.assertEqual(partial['asset_verification']['other_recipe_assets'],
            'metadata_only; not resolved, stat-ed, hashed or loaded')
        self.assertFalse(partial['asset_verification']['full_recipe_assets_verified'])
        self.assertEqual(partial['legacy_numeric_entry_sha256'],portable.LEGACY_ENTRY_SHA)
        with self.assertRaises(FileExistsError):portable.Evidence(evidence.path,{},proof)

    def test_main_delegates_identical_inputs_and_pretrained_subset_to_frozen_execute(self):
        # Admission is separately exercised below; this tests the new main's
        # only numerical boundary without a Comfy import or model construction.
        used=self.read();calls=[]
        raw,rphi=object(),object()
        contract_path=self.root/'execution.json';contract_path.write_text('{}')
        contract={'root':str(self.root),'asset_verification_scope':assets.SCOPE}
        paths={'pretrained_manifest':self.path}
        proof={'preprocessing':{'raw':{'after_P_float32_sha256':'raw'},'rphi':{'after_P_float32_sha256':'rphi'}}}
        def fake_execute(actual_paths,pretrained,actual_raw,actual_rphi,evidence,verified):
            calls.append((actual_paths,actual_raw,actual_rphi,verified))
            self.assertIs(actual_raw,raw);self.assertIs(actual_rphi,rphi)
            self.assertEqual(set(pretrained['_paths']),{'vae','context'})
            self.assertFalse(pretrained['_files_sha256_verified'])
            evidence.data['asset_verification']={'VAE_and_context_actual_SHA':True,
                'other_recipe_assets':'sizes_only; not loaded or rehashed'}
            evidence.data['complete']=True;evidence.save()
            portable.legacy.atomic_json(evidence.path/'manifest.json',evidence.data)
        with patch.object(portable,'check_contract',return_value=(contract,paths,self.root,{})),\
             patch.object(portable.legacy,'inspect_inputs',return_value=(proof,raw,rphi)),\
             patch.object(portable,'read_vae_only_manifest',return_value=used),\
             patch.object(portable.legacy,'execute',side_effect=fake_execute),\
             patch.object(all_assets,'read_manifest',side_effect=AssertionError('generic reader forbidden')),\
             patch.dict(os.environ,{'ASCEND_RT_VISIBLE_DEVICES':'0'}):
            portable.main(['--execution-contract',str(contract_path),'--out',str(self.root/'out'),'--execute'])
        self.assertEqual(len(calls),1)
        report=json.loads((self.root/'out/manifest.json').read_text())
        self.assertFalse(report['asset_verification']['full_recipe_assets_verified'])
        self.assertEqual(report['inspection']['pretrained_fingerprint'],used['_fingerprint'])

    def test_portable_entry_requires_explicit_scope_and_unchanged_numeric_source(self):
        contract={'schema':portable.EXECUTION_SCHEMA,'entry_sha256':portable.legacy.sha_file(portable.__file__),
            'legacy_entry_sha256':portable.LEGACY_ENTRY_SHA,'asset_verification_scope':'all_components'}
        path=self.root/'execution.json';path.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError,'explicit used-components-only'):portable.check_contract(path)
        contract['legacy_entry_sha256']='0'*64;path.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError,'frozen numerical exporter'):portable.check_contract(path)
        self.assertEqual(portable.legacy.sha_file(portable.legacy.__file__),portable.LEGACY_ENTRY_SHA)

    def test_failure_resets_complete_and_keeps_truthful_scope(self):
        used=self.read();contract_path=self.root/'execution.json';contract_path.write_text('{}')
        proof={'preprocessing':{}}
        def fail(paths,pretrained,raw,rphi,evidence,verified):
            evidence.data['complete']=True
            evidence.data['asset_verification']={'other_recipe_assets':'sizes_only; not loaded or rehashed'}
            raise RuntimeError('injected legacy execution failure')
        with patch.object(portable,'check_contract',return_value=({}, {'pretrained_manifest':self.path},self.root,{})),\
             patch.object(portable.legacy,'inspect_inputs',return_value=(proof,None,None)),\
             patch.object(portable,'read_vae_only_manifest',return_value=used),\
             patch.object(portable.legacy,'execute',side_effect=fail),patch.dict(os.environ,{'ASCEND_RT_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(RuntimeError,'injected legacy'):
                portable.main(['--execution-contract',str(contract_path),'--out',str(self.root/'failed'),'--execute'])
        failed=json.loads((self.root/'failed/failure.json').read_text())
        self.assertFalse(failed['complete'])
        self.assertEqual(failed['asset_verification']['other_recipe_assets'],
            'metadata_only; not resolved, stat-ed, hashed or loaded')
        self.assertFalse(failed['asset_verification']['full_recipe_assets_verified'])

    def test_actual_frozen_loader_AST_only_uses_VAE_context_and_restores_bindings(self):
        # Run the three pinned loader function bodies with fake torch/H3 only.
        # This verifies delegation and absence of a second generic manifest read;
        # it is not a real VAE forward or a device import.
        path=Path(portable.legacy.__file__).parent/'run_vae_roundtrip_diagnostic.py'
        source=path.read_text()
        expected='04e2f3e63b3caabec6f9998d97f7c33d39f920fa89b7d17776b48c73b20d1811'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
        tree=ast.parse(source)
        names={'verify_vae_binding','manifest_vae_binding','load_vae_only'}
        selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in selected},names)
        ns={'Path':Path,'sha_file':all_assets.sha256,'contextmanager':contextmanager,
            'hashlib':hashlib,'inspect':inspect}
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
        used=self.read();calls=[]
        class MiniMaxH3VideoVAE:
            def _finalize_pixels(self,value):return value
        implementation=ModuleType('h3_comfy')
        implementation.VAE_PATH=self.root/'old_vae';implementation.CONTEXT_PATH=self.root/'old_context'
        original=(implementation.VAE_PATH,implementation.CONTEXT_PATH)
        implementation.SHA={r:self.manifest[r]['sha256'] for r in ('vae','context')}
        implementation.SIZE={'vae':self.manifest['vae']['bytes']}
        def H3(**kwargs):
            self.assertEqual(kwargs,{'load_dit':False,'verify_sha':True})
            self.assertEqual(implementation.VAE_PATH,self.files['vae'])
            self.assertEqual(implementation.CONTEXT_PATH,self.files['context'])
            calls.append(kwargs)
            return SimpleNamespace(patcher=None,record={},vae=SimpleNamespace(
                first_stage_model=MiniMaxH3VideoVAE(),vae_dtype='torch.float16'))
        implementation.H3=H3
        fake_torch=ModuleType('torch')
        fake_torch.npu=SimpleNamespace(set_device=lambda n:calls.append(('device',n)),
            set_compile_mode=lambda **k:calls.append(('compile',k)))
        for role in ('base','refiner','turbo'):self.files[role].unlink()
        with patch.dict(sys.modules,{'torch':fake_torch,'h3_comfy':implementation}),\
             patch.object(all_assets,'read_manifest',side_effect=AssertionError('all-component stat forbidden')):
            model=ns['load_vae_only'](used)
        self.assertEqual((implementation.VAE_PATH,implementation.CONTEXT_PATH),original)
        self.assertEqual(calls[-1],{'load_dit':False,'verify_sha':True})
        self.assertFalse(model.record['vae_only']['dit_loaded'])


if __name__=='__main__':unittest.main(verbosity=2)
