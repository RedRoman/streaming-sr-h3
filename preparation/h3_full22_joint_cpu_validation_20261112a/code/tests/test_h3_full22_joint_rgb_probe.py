"""CPU fixtures for full22 cache contracts and retained joint orchestration.

Tiny fake DiT/VAE operators; actual pinned sampler and tile-row function bodies
are executed via AST. No H3 weights, Comfy runtime import or NPU calls.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import check_h3_full22_joint_rgb_grad as P
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    import openh3_train_trajectory as T
    from test_openh3_train_trajectory import fixture, reference_function


class InputTests(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_module_import_does_not_load_torch_comfy_or_model(self):
        env = dict(os.environ, PYTHONPATH=str(Path(P.__file__).parent), TORCH_DEVICE_BACKEND_AUTOLOAD='0')
        code = 'import check_h3_full22_joint_rgb_grad,sys; assert not any(x in sys.modules for x in ("torch","torch_npu","comfy"))'
        subprocess.run([sys.executable, '-c', code], check=True, env=env, timeout=20)

    def test_target_loss_conversion_exact_and_tamper_rejected(self):
        a = np.arange(72, dtype=np.uint8).reshape(2,3,4,3)
        path = self.root/'target.npz'; np.savez(path, frames=a)
        converted = a.astype(np.float32)/np.float32(255.)
        spec = dict(npz_file_sha256=P.sha_file(path), key='frames', shape=list(a.shape),
                    array_sha256=P.array_sha(a), loss_target_float32_sha256=P.array_sha(converted))
        np.testing.assert_array_equal(P.clean_target(path,spec),converted)
        wrong = dict(spec, loss_target_float32_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'conversion'):P.clean_target(path,wrong)
        np.savez(path,frames=a+1)
        with self.assertRaisesRegex(ValueError,'file'):P.clean_target(path,spec)

    def test_array_file_shape_dtype_pixels_and_finite_bound(self):
        p = self.root/'latent.npy';a=np.ones((1,24,7,2,2),np.float32);np.save(p,a)
        spec=dict(path=str(p),bytes=p.stat().st_size,file_sha256=P.sha_file(p),array_sha256=P.array_sha(a),shape=list(a.shape),dtype='float32')
        np.testing.assert_array_equal(P.checked_array(spec),a)
        with self.assertRaisesRegex(ValueError,'geometry'):P.checked_array(dict(spec,shape=[1,24,2,2,2]))
        a[0,0,0,0,0]=np.nan;np.save(p,a)
        spec.update(file_sha256=P.sha_file(p),array_sha256=P.array_sha(a))
        with self.assertRaisesRegex(ValueError,'nonfinite'):P.checked_array(spec)

    def test_train_source_not_dev_or_oracle_identity(self):
        expected=dict(source_id='train',source_sha256='a',sample_id='s',record_sha256='r',role='train',task='compression',geometry=list(P.RGB_SHAPE))
        i={**expected,'source_isolation':dict(source_id_overlap=0,source_sha256_overlap=0,train_source_ids=['train'],dev_source_ids=['dev'],train_source_sha256=['a'],dev_source_sha256=['b'])}
        P.validate_identity(i,expected)
        with self.assertRaisesRegex(ValueError,'identity'):P.validate_identity(dict(i,role='dev'),expected)
        i['source_isolation']['dev_source_sha256'].append('a')
        with self.assertRaisesRegex(ValueError,'source bytes'):P.validate_identity(i,expected)

    def test_admission_payload_cannot_hide_budget_or_input_change(self):
        c={'budget':6,'inputs':'fixed','root_admission':None};old=P.execution_payload_sha256(c)
        self.assertEqual(old,P.execution_payload_sha256(dict(c,root_admission={'sha256':'x'})))
        self.assertNotEqual(old,P.execution_payload_sha256(dict(c,budget=9)))
        self.assertNotEqual(old,P.execution_payload_sha256(dict(c,inputs='target-in-guide')))


@unittest.skipUnless(torch is not None and os.environ.get('OPENH3_COMFY_REFERENCE_ROOT'), 'requires CPU Torch and pinned AST reference sources')
class NativeAndJointTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.e=P.Evidence(Path(self.tmp.name)/'probe',{'fixture':True})
        self.h3,self.backend,self.state=fixture();self.dit=self.h3.patcher.model.diffusion_model
        with torch.no_grad():self.dit.blocks[0].lora_b.zero_()
        self.kind='dit';self.pixel_calls=0;self.memory_free=1<<30
        test=self
        class VBlock(torch.nn.Module):
            def forward(self,x):
                if test.kind!='vae':raise RuntimeError('VAE math missing')
                return x+x.sin()*.001
        class Stage(torch.nn.Module):
            tiling=True;tile_size=256;vae_ratio=16;vae_ratio_t=4
            def __init__(self):
                super().__init__();self.frozen=torch.nn.Parameter(torch.tensor(.1),requires_grad=False)
                self.register_buffer('buf',torch.tensor(.1),persistent=False)
                self.decoder=torch.nn.Module();self.decoder.transformer_blocks=torch.nn.ModuleList([VBlock() for _ in range(36)])
            def _decode_pixels(self,z):
                test.pixel_calls+=1
                for b in self.decoder.transformer_blocks:z=b(z)
                return z.float().mean((1,2,3,4),keepdim=True)
            def _decode_temporal_chunks(self,t):return (0,1)
            def split_tiles(self,n):
                return ([0,112,224],[256]*3,[144]*2) if n==480 else ([0,192,384,576],[256]*4,[64]*3)
            def decode_output_shape(self,s):return [1,3,22,480,832]
            def decode(self,z):
                means=[]
                for y in [0,7,14]:
                    g=self._decode_tile_row(z[...,y:y+16,:],[0,192,384,576],[256]*4)
                    # Exactly the native four next() calls, no fifth resume.
                    for _ in range(4):means.append(next(g).mean())
                return torch.stack(means).mean().sigmoid().reshape(1,1,1,1).expand(22,4,4,3)
        src=Path(os.environ['OPENH3_COMFY_REFERENCE_ROOT'])/'comfy/ldm/minimax/vae.py'
        self.assertEqual(P.sha_file(src),P.VAE_SOURCE_SHA)
        ns={'torch':torch,'comfy':SimpleNamespace(model_management=SimpleNamespace(get_free_memory=lambda device:test.memory_free))}
        Stage._decode_tile_row=self.actual_row(src,ns)
        self.stage=Stage().eval();self.h3.vae=SimpleNamespace(first_stage_model=self.stage,device='cpu',offload_device='cpu',disable_offload=True,vae_dtype=torch.float16,decode=self.stage.decode)
        self.guide=torch.full(P.SHAPE,.2);self.raw=torch.full(P.SHAPE,.4)
        original=self.dit.blocks[0].forward
        def checked_dit(*a,**kw):
            if self.kind!='dit':raise RuntimeError('decoder math leaked to DiT')
            return original(*a,**kw)
        self.dit.blocks[0].forward=checked_dit
    def actual_row(self,src,ns):
        import ast
        tree=ast.parse(src.read_text())
        cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and any(isinstance(n,ast.FunctionDef) and n.name=='_decode_tile_row' for n in c.body))
        return reference_function(src,'_decode_tile_row',ns,cls.name)
    def tearDown(self):self.tmp.cleanup()

    @contextmanager
    def decoder_context(self):
        before=self.kind;phase=self.e.data['phase'];self.kind='vae'
        try:yield
        finally:
            self.kind=before
            self.e.data['decoder_context_restorations'].append({'phase':phase,'restored':self.kind==before})
    @contextmanager
    def reference_context(self):
        before=self.kind;self.kind='vae'
        try:yield
        finally:self.kind=before
    def session(self,checkpoint):
        return T.ResidentH3Trajectory(self.h3,_backend=self.backend,checkpoint_blocks=checkpoint,checkpoint_decoder=checkpoint,decoder_context=self.decoder_context)
    def observe(self,group,phase,index):
        self.assertEqual(self.kind,'vae' if group=='VAE' else 'dit')
        if group=='DiT':self.assertTrue(self.state['math'])

    def test_native_plan_is_one_temporal_chunk_and_three_spatial_rows(self):
        plan=P.validate_native_plan(self.stage)
        self.assertEqual(plan['temporal_padding_and_chunks'],[0,1])
        self.stage._decode_temporal_chunks=lambda t:(5,1)
        with self.assertRaisesRegex(ValueError,'temporal'):P.validate_native_plan(self.stage)

    def test_dynamic_batch_change_blocks_real_decode_and_restores(self):
        snap=P.wrapper_snapshot([(self.stage,n) for n in ('decode','_decode_tile_row','_decode_pixels')])
        self.e.phase('reference_decode');self.memory_free=256<<20
        with self.assertRaisesRegex(RuntimeError,'batch must be4'):
            with P.native_tile_guard(self.stage,self.e,torch):
                list(self.stage._decode_tile_row(torch.zeros((1,24,7,16,52),dtype=torch.float16),[0,192,384,576],[256]*4))
        self.assertEqual(self.pixel_calls,0);self.assertTrue(P.wrappers_restored(snap))
        self.assertEqual(self.e.data['actual']['backward_attempts'],0)

    def test_fourth_block_call_fails_before_original_and_restores(self):
        block=torch.nn.Identity();calls=[];snap=P.wrapper_snapshot([(block,'forward')])
        self.e.phase('candidate')
        with self.assertRaisesRegex(RuntimeError,'block count'):
            with P.bounded_blocks([block],'DiT',self.e,lambda *a:calls.append(a)):
                for _ in range(4):block(torch.ones(1))
        self.assertEqual(len(calls),3);self.assertTrue(P.wrappers_restored(snap))

    def test_actual_joint_three_rows_217_contexts_and_only_RGB_backward(self):
        original=P.wrapper_snapshot([(b,'forward') for b in list(self.dit.blocks)+list(self.stage.decoder.transformer_blocks)]+[(self.stage,n) for n in ('decode','_decode_tile_row','_decode_pixels')])
        frozen=P.freeze_snapshot(self.stage,torch);adapters=P.adapter_snapshot(self.dit,torch)
        with P.guarded_vae(self.h3,self.e,torch),P.native_tile_guard(self.stage,self.e,torch),P.bounded_blocks(self.dit.blocks,'DiT',self.e,self.observe),P.bounded_blocks(self.stage.decoder.transformer_blocks,'VAE',self.e,self.observe):
            self.e.phase('reference')
            with torch.no_grad(),self.session(False) as s:reference=s.sample(self.guide,self.raw)
            self.e.phase('reference_decode')
            with torch.no_grad(),self.reference_context():ref_rgb=self.h3.vae.decode(reference['video'].to(torch.float16))
            with self.session(True) as s:
                self.e.phase('candidate');candidate=s.sample(self.guide,self.raw)
                P.save_latent_parity(reference,candidate,self.e,torch)
                self.assertEqual(tuple(candidate['audio'].shape),P.AUDIO_SHAPE)
                candidate['video'].retain_grad()
                self.e.phase('candidate_decode');rgb=s.decode(candidate['video'])
                self.assertEqual(self.kind,'dit');self.assertTrue(P.tile_budget_passed(self.e.data['native_tile_rows']))
                target=torch.linspace(.1,.3,rgb.numel()).reshape(rgb.shape)
                P.full_rgb_backward(ref_rgb,rgb,candidate['video'],target,self.e,torch,expected_shape=tuple(rgb.shape),target_sha256=P.array_sha(target.numpy()))
                P.save_gradients(self.dit,candidate['video'],self.e,torch,expected=2)
                self.assertTrue(self.e.data['adapter_gradients']['passed']);self.assertTrue(self.e.data['retained_video_gradient']['passed'])
                self.assertTrue(torch.equal(self.dit.blocks[0].lora_a.grad,torch.zeros_like(self.dit.blocks[0].lora_a)))
                self.assertEqual(self.kind,'dit')
            with self.assertRaises(RuntimeError):s._guard_backward(torch.zeros(()),s.generation)
        self.assertTrue(P.wrappers_restored(original));self.assertFalse(self.state['math'])
        self.assertTrue(P.check_snapshot(self.stage,frozen,torch)['passed']);self.assertTrue(P.check_adapters(self.dit,adapters,torch,'cpu')['passed'])
        for phase in ('reference_decode','candidate_decode','backward'):
            self.assertEqual(self.e.data['block_forward_calls']['VAE'][phase],{str(i):3 for i in range(36)})
        for phase in ('reference','candidate','backward'):self.assertEqual(self.e.data['block_forward_calls']['DiT'][phase],{'0':3})
        contexts=self.e.data['decoder_context_restorations'];self.assertEqual(len(contexts),217)
        self.assertEqual(sum(r['phase']=='candidate_decode' for r in contexts),109)
        self.assertTrue(all(r['restored'] for r in contexts));self.assertEqual(self.pixel_calls,6)
        self.assertEqual(self.e.data['actual']['completed_backward_calls'],1)
        self.assertEqual(self.e.data['actual']['optimizer_steps'],0)
        self.assertEqual(self.e.data['actual']['adapter_updates'],0)
        self.assertEqual(self.e.data['actual']['stage_decode_calls'],{'reference_decode':1,'candidate_decode':1})
        self.assertIsNone(target.grad)
        self.assertTrue(torch.equal(self.state['guide'],self.guide))
        print(json.dumps({'fixture':'actual pinned sampler/tile-row AST with tiny fake DiT/VAE','contexts':217,'VAE_block_entries':324,'pixel_groups':6,'CPU_RGB_backward':1,'NPU_calls':0}),flush=True)

    def test_RGB_or_target_mismatch_prevents_backward(self):
        for mismatch in ('rgb','target'):
            a=torch.nn.Parameter(torch.tensor(.2));v=a*torch.ones(1,2,3,3);v.retain_grad();rgb=v.sigmoid();target=torch.zeros_like(rgb)
            ref=torch.zeros_like(rgb) if mismatch=='rgb' else rgb.detach().clone()
            with self.assertRaisesRegex(RuntimeError,'parity|target bytes'):
                P.full_rgb_backward(ref,rgb,v,target,self.e,torch,expected_shape=tuple(rgb.shape),target_sha256='0'*64)
            self.assertEqual(self.e.data['actual']['backward_attempts'],0);self.assertIsNone(a.grad)

if __name__=='__main__':unittest.main()
