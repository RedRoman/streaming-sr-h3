"""CPU-only contract tests: official preprocessing AST and tiny VAE call boundary."""
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
try:import torch
except ImportError:torch=None
import export_train_vae_cache as cache


def metadata_fixture():
    record={'source_id':cache.SOURCE_ID,'source_sha256':cache.SOURCE_SHA,'split':'train',
        'task':'compression','target_key':'4bf3899ab4b0549094789693104e84ed98d3e6ef5f1e4fdc0b11c0bbdec60fc0',
        'target_pixel_sha256':'da17b42cb45382deb3415384562ab09dde42eca78be1fee7b1475b6b4b36c614',
        'observation_pixel_sha256':'a'*64}
    row={**record,'sample_id':cache.SAMPLE_ID,'role':'train','record_sha256':cache.canonical_sha(record),
        'checkpoint_sha256':cache.CHECKPOINT_SHA256,'condition_mode':'observations_only',
        'target_role':'supervision_reference_only_never_model_input',
        'frame_indices':list(range(22)),'history_indices':cache.causal_indices(range(22),3,22).tolist(),
        'train_manifest_sha256':cache.FIXED_SHA['train_manifest'],'dev_manifest_sha256':cache.FIXED_SHA['dev_manifest']}
    manifest={'schema':'rphi.train_rgb_guide_export.v1','complete':True,'role':'train','split':'train',
        'samples':4,'source_ids':[cache.SOURCE_ID],'checkpoint_sha256':cache.CHECKPOINT_SHA256,
        'checkpoint_step':3000,'model_config':cache.MODEL_CONFIG,'vae_encoded':False,
        'train_manifest_sha256':cache.FIXED_SHA['train_manifest'],'dev_manifest_sha256':cache.FIXED_SHA['dev_manifest'],
        'per_sample':[row]}
    return manifest,record


class MetadataTests(unittest.TestCase):
    def test_selected_ordinary_train_binding_and_reject_controls_dev_source_target(self):
        manifest,record=metadata_fixture()
        self.assertEqual(cache.selected_row(manifest,record)['sample_id'],cache.SAMPLE_ID)
        for key,value in [('role','dev'),('checkpoint_sha256','b'*64),('target_role','model_input'),
                          ('control','identity'),('target_pixel_sha256','c'*64),('record_sha256','d'*64)]:
            broken=copy.deepcopy(manifest);broken['per_sample'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):cache.selected_row(broken,record)
        bad=copy.deepcopy(record);bad['source_sha256']='e'*64
        with self.assertRaises(ValueError):cache.selected_row(manifest,bad)

    def test_numeric_boundary_has_no_target_and_new_output_only(self):
        self.assertEqual(list(inspect.signature(cache.encode_pair).parameters),
            ['h3','raw_pixels','rphi_pixels','e','torch','expected_preprocessing'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'new'
            cache.Evidence(path,{},{});
            with self.assertRaises(FileExistsError):cache.Evidence(path,{},{});


@unittest.skipUnless(torch is not None and os.environ.get('OPENH3_COMFY_REFERENCE_ROOT'),
                     'requires Torch CPU and pinned official preprocessing sources')
class PreprocessAndCalls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(os.environ['OPENH3_COMFY_REFERENCE_ROOT'])
        self.raw=np.full((22,32,32,3),.501,dtype=np.float32)
        self.rphi=np.full_like(self.raw,.701)
        self.proofs={k:cache.official_preprocess(v,self.root)[1] for k,v in [('raw',self.raw),('rphi',self.rphi)]}
        self.e=cache.Evidence(Path(self.tmp.name)/'out',{}, {'preprocessing':self.proofs})
    def tearDown(self):self.tmp.cleanup()

    def test_same_size_official_P_preserves_truncation_and_not_identity(self):
        pixels=np.array([.50001,.50199,.9999],dtype=np.float32).reshape(1,1,1,3)
        output,proof=cache.official_preprocess(pixels,self.root)
        expected=np.clip(255*pixels,0,255).astype(np.uint8).astype(np.float32)/np.float32(255)
        np.testing.assert_array_equal(output,expected)
        self.assertFalse(np.array_equal(output,pixels))
        self.assertEqual(proof['PIL_uint8_sha256'],cache.array_sha(np.clip(255*pixels,0,255).astype(np.uint8)))
        self.assertEqual(proof['changed_components'],3)
        self.assertNotEqual(proof['before_P_float32_sha256'],proof['after_P_float32_sha256'])

    def fake_h3(self,mutate_P=False,fail_decode=False):
        root=self.root
        class Stage:
            def encode(self,x):
                self.encoded_inputs=getattr(self,'encoded_inputs',[])+[x.clone()]
                return x.float().mean().reshape(1,1,1,1,1).expand(1,24,7,2,2).clone()
            def decode(self,x):
                if fail_decode:raise RuntimeError('injected decode failure')
                return x.float().mean().reshape(1,1,1,1).expand(22,32,32,3).clone()
        class VAE:
            def __init__(self):self.first_stage_model=Stage()
            def encode(self,p):return self.first_stage_model.encode(p.half()).float()
            def decode(self,z):return self.first_stage_model.decode(z.half()).float()
            def encode_tiled_3d(self,x):return x
            def decode_tiled_3d(self,x):return x
        class H3:
            def __init__(self):self.vae=VAE()
            def encode_guide_pixels(self,pixels,hw):
                pre,_=cache.official_preprocess(pixels.numpy(),root)
                if mutate_P:pre=pre+np.float32(.01)
                return self.vae.encode(torch.from_numpy(pre))
            def decode(self,z):return self.vae.decode(z)
            def sample(self,*args):raise AssertionError('forbidden DiT')
            def encode(self,*args):raise AssertionError('wrong entry')
        return H3()

    def test_exact_two_encode_one_decode_and_fixed_raw_draft(self):
        h3=self.fake_h3();before_raw=self.raw.copy();before_rphi=self.rphi.copy()
        original=cache.wrapper_snapshot([(h3,'sample'),(h3,'encode'),(h3.vae,'encode'),(h3.vae,'decode'),
            (h3.vae.first_stage_model,'encode'),(h3.vae.first_stage_model,'decode')])
        with cache.bounded_calls(h3,self.e,torch,self.proofs):
            latents,pixels=cache.encode_pair(h3,self.raw,self.rphi,self.e,torch,self.proofs)
        self.assertTrue(cache.wrappers_restored(original))
        self.assertEqual(self.e.data['actual']['encode_calls'],2)
        self.assertEqual(self.e.data['actual']['decode_calls'],1)
        self.assertEqual(self.e.data['actual']['stage_encode_calls'],2)
        self.assertEqual(self.e.data['actual']['stage_decode_calls'],1)
        self.assertEqual(self.e.data['actual']['PIL_fromarray_calls'],{'raw':22,'rphi':22})
        self.assertEqual(set(self.e.data['arrays']),{'raw_warm_start_latent','rphi_guide_latent','vae_rphi_as_returned_internal_clipped'})
        self.assertFalse(torch.equal(latents['raw'],latents['rphi']))
        self.assertEqual(self.e.data['actual_decode_input']['array_sha256'],self.e.data['arrays']['rphi_guide_latent']['array_sha256'])
        np.testing.assert_array_equal(self.raw,before_raw);np.testing.assert_array_equal(self.rphi,before_rphi)
        for name,row in self.e.data['arrays'].items():
            p=self.e.path/row['path'];value=np.load(p,allow_pickle=False)
            self.assertEqual(cache.sha_file(p),row['file_sha256']);self.assertEqual(cache.array_sha(value),row['array_sha256'])
        self.assertEqual(self.e.data['actual']['H3_sampler_calls'],0)

    def test_actual_preprocess_mismatch_stops_before_stage_call(self):
        h3=self.fake_h3(mutate_P=True)
        with self.assertRaisesRegex(RuntimeError,'actual deployed preprocessing'):
            with cache.bounded_calls(h3,self.e,torch,self.proofs):
                cache.encode_pair(h3,self.raw,self.rphi,self.e,torch,self.proofs)
        self.assertEqual(self.e.data['actual']['encode_calls'],1)
        self.assertEqual(self.e.data['actual']['stage_encode_calls'],0)
        self.assertEqual(self.e.data['actual']['decode_calls'],0)

    def test_wrong_input_sha_and_out_of_range_fail_closed(self):
        h3=self.fake_h3()
        changed=self.raw.copy();changed[0,0,0,0]+=.01
        with self.assertRaisesRegex(RuntimeError,'input pixels changed'):
            cache.encode_pair(h3,changed,self.rphi,self.e,torch,self.proofs)
        with self.assertRaisesRegex(ValueError,'clipped'):
            cache.encode_pair(h3,self.raw+1,self.rphi,self.e,torch,self.proofs)
        self.assertEqual(self.e.data['actual']['encode_calls'],0)

    def test_decode_exception_restores_wrappers_preserves_completed_latents(self):
        h3=self.fake_h3(fail_decode=True)
        original=cache.wrapper_snapshot([(h3,'sample'),(h3,'encode'),(h3.vae,'encode'),(h3.vae,'decode'),
            (h3.vae.first_stage_model,'encode'),(h3.vae.first_stage_model,'decode')])
        with self.assertRaisesRegex(RuntimeError,'injected decode'):
            with cache.bounded_calls(h3,self.e,torch,self.proofs):
                cache.encode_pair(h3,self.raw,self.rphi,self.e,torch,self.proofs)
        self.assertTrue(cache.wrappers_restored(original))
        self.assertEqual(set(self.e.data['arrays']),{'raw_warm_start_latent','rphi_guide_latent'})
        self.assertFalse(self.e.data['complete'])
        self.assertEqual(self.e.data['actual']['stage_decode_calls'],1)

    def test_second_decode_and_oom_fallback_are_forbidden(self):
        h3=self.fake_h3()
        with cache.bounded_calls(h3,self.e,torch,self.proofs):
            latents,_=cache.encode_pair(h3,self.raw,self.rphi,self.e,torch,self.proofs)
            with self.assertRaisesRegex(RuntimeError,'decode budget'):
                h3.vae.decode(latents['rphi'])
            with self.assertRaisesRegex(RuntimeError,'undeclared path'):
                h3.vae.encode_tiled_3d(torch.zeros(1))
            with self.assertRaisesRegex(RuntimeError,'undeclared path'):
                h3.sample(None)
        self.assertEqual(self.e.data['actual']['stage_decode_calls'],1)

if __name__=='__main__':unittest.main(verbosity=2)
