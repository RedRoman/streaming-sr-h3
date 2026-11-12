"""CPU numerical/serialization tests only; no real H3/VAE or NPU loading."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
try:
    import torch
except ImportError:
    torch=None
import check_h3_nonzero_adapter as P
import check_h3_joint_rgb_grad as J
if torch is not None:
    import lora as L
    from openh3_runtime import inject_adaptation


def tiny_model(blocks=1):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn=torch.nn.Module(); self.mlp=torch.nn.Module()
            for target in L.TARGETS:
                parent,attribute=target.split('.')
                layer=torch.nn.Linear(1,1,bias=True,dtype=torch.bfloat16)
                with torch.no_grad(): layer.weight.fill_(.1);layer.bias.fill_(.01)
                setattr(getattr(self,parent),attribute,layer)
        def forward(self,x,index=0,transformer_options=None):
            value=x.mean(-1,keepdim=True).to(torch.bfloat16)
            total=torch.zeros_like(value,dtype=torch.float32)
            for target in L.TARGETS:
                parent,attribute=target.split('.')
                total=total+getattr(getattr(self,parent),attribute)(value).float()
            return total*1e-4
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base=torch.nn.Parameter(torch.tensor(.65),requires_grad=False)
            self.blocks=torch.nn.ModuleList([Block() for _ in range(blocks)])
            self.register_buffer('nonpersistent',torch.tensor(.001),persistent=False)
            self._openh3_pretrained_record={'fingerprint':'CPU_fixture_not_real_pretrained'}
            self.eval().requires_grad_(False)
        def forward(self,x,index=0):
            value=self.base*x+x.mean(-1,keepdim=True)*.07
            for b in self.blocks: value=value+b(x,index)
            return value
    model=Model()
    names=inject_adaptation(model,rank=8,seed=101)
    return model,names


@unittest.skipUnless(torch is not None,'requires Torch CPU')
class ArithmeticAndIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.e=P.Evidence(Path(self.tmp.name)/'run',{'test_only':True})
        self.model,self.names=tiny_model()
        self.expected=P.expected_adapters(self.model,torch,blocks=1)
        self.pretrained=self.model._openh3_pretrained_record
        self.diagnostic={'role':'synthetic','lr':1e-4,'unscale':1024.,'inputs':'fixed'}
    def tearDown(self): self.tmp.cleanup()

    def gradients(self):
        for n,p in L.lora_parameters(self.model):
            p.grad=torch.full_like(p,0. if n.endswith('.lora_a') else 1024.)
        value=torch.ones(1,requires_grad=True)*2
        value.retain_grad();value.sum().backward()
        J.save_gradients(self.model,value,self.e,torch,expected=len(self.expected))

    def test_SGD_once_unscale_and_zero_A_legitimate(self):
        self.gradients()
        before={n:p.clone() for n,p in L.lora_parameters(self.model)}
        P.one_sgd_update(self.model,self.expected,self.e,torch)
        for n,p in L.lora_parameters(self.model):
            wanted=before[n] if n.endswith('.lora_a') else torch.full_like(p,-1e-4)
            self.assertTrue(torch.equal(p,wanted))
            self.assertEqual(self.e.data['update']['tensors'][n]['exact_SGD_arithmetic'],True)
        self.assertEqual(self.e.data['actual']['optimizer_steps'],1)
        self.assertEqual(self.e.data['actual']['unscale_calls'],1)
        with self.assertRaisesRegex(RuntimeError,'second update'):
            P.one_sgd_update(self.model,self.expected,self.e,torch)

    def test_missing_or_nonfinite_gradient_stops_before_update(self):
        for broken in ('missing','nan'):
            self.gradients()
            p=next(iter(dict(L.lora_parameters(self.model)).values()))
            p.grad=None if broken=='missing' else torch.full_like(p,float('nan'))
            with self.assertRaisesRegex(RuntimeError,'missing/nonfinite'):
                P.one_sgd_update(self.model,self.expected,self.e,torch)
            self.assertEqual(self.e.data['actual']['optimizer_steps'],0)

    def test_strict_file_keys_dtype_shape_metadata_and_destructive_reload(self):
        from safetensors.torch import save_file
        from openh3_runtime import adapter_metadata
        self.gradients();P.one_sgd_update(self.model,self.expected,self.e,torch)
        path=P.save_checkpoint(self.model,self.expected,self.pretrained,self.diagnostic,self.e,torch)
        original_sha=P.sha_file(path)
        P.clear_reload(self.model,self.expected,path,self.pretrained,self.diagnostic,self.e,torch)
        self.assertEqual(P.sha_file(path),original_sha)
        self.assertTrue(all(r['all_zero'] for r in self.e.data['reload']['cleared'].values()))
        self.assertTrue(all(r['equals_saved'] for r in self.e.data['reload']['reloaded'].values()))
        with self.assertRaisesRegex(RuntimeError,'second save'):
            P.save_checkpoint(self.model,self.expected,self.pretrained,self.diagnostic,self.e,torch)
        state=L.state_dict(self.model)
        first=next(iter(state))
        cases=[]
        bad=state.copy();bad[first]=bad[first][:1];cases.append(('shape',bad,self.diagnostic))
        bad=state.copy();bad[first]=bad[first].half();cases.append(('FP32',bad,self.diagnostic))
        bad=state.copy();bad['extra']=torch.zeros(1);cases.append(('coverage',bad,self.diagnostic))
        bad=state.copy();bad.pop(first);cases.append(('coverage',bad,self.diagnostic))
        bad=state.copy();bad[first]=torch.full_like(bad[first],float('nan'));cases.append(('nonfinite',bad,self.diagnostic))
        cases.append(('objective',state,{'role':'different'}))
        for i,(message,bad,meta) in enumerate(cases):
            badpath=Path(self.tmp.name)/f'bad{i}.safetensors'
            save_file(bad,str(badpath),metadata=adapter_metadata(self.pretrained,meta,8))
            with self.assertRaisesRegex(ValueError,message):
                P.strict_saved_state(badpath,self.expected,self.pretrained,self.diagnostic,torch)

    def test_expected_shapes_from_real_linears_scale_and_alias_rejected(self):
        module=self.model.get_submodule(self.names[0])
        module.scale=2.
        with self.assertRaisesRegex(ValueError,'scale'): P.expected_adapters(self.model,torch,blocks=1)
        module.scale=1.
        module.lora_a=torch.nn.Parameter(torch.zeros(8,2))
        with self.assertRaisesRegex(ValueError,'shape'): P.expected_adapters(self.model,torch,blocks=1)

    def test_merge_BF16_rounding_descriptive_and_only_declared_writes(self):
        self.gradients();P.one_sgd_update(self.model,self.expected,self.e,torch)
        original=J.freeze_snapshot(self.model,torch,self.names)
        revised=P.merge_once(self.model,self.names,original,self.e,torch)
        self.assertFalse(L.lora_parameters(self.model))
        self.assertFalse(J.check_snapshot(self.model,original,torch,self.names)['passed'])
        self.assertTrue(J.check_snapshot(self.model,revised,torch,self.names)['passed'])
        self.assertTrue(all(r['exact_declared_write'] for r in self.e.data['merge']['layers'].values()))
        self.assertGreater(sum(r['nonzero_ideal_rounded_to_zero_count'] for r in self.e.data['merge']['layers'].values()),0)
        self.assertFalse(self.e.data['merge']['equivalence_claim'])
        with torch.no_grad(): self.model.blocks[0].attn.qkv_proj.bias.add_(1)
        self.assertFalse(J.check_snapshot(self.model,revised,torch,self.names)['passed'])

    def test_nonfinite_merge_preserves_JSON_evidence_before_any_base_write(self):
        original=J.freeze_snapshot(self.model,torch,self.names)
        with torch.no_grad():
            for _,p in L.lora_parameters(self.model): p.fill_(1e30)
        with self.assertRaisesRegex(RuntimeError,'nonfinite expected merged weights'):
            P.merge_once(self.model,self.names,original,self.e,torch)
        partial=json.loads((self.e.path/'partial.json').read_text())
        self.assertEqual(partial['actual']['merge_calls'],0)
        self.assertTrue(any(not row['all_finite'] for row in partial['merge']['layers'].values()))
        self.assertTrue(all(row['ideal_delta_l2'] is None for row in partial['merge']['layers'].values()))
        self.assertTrue(J.check_snapshot(self.model,original,torch,self.names)['passed'])

    def test_reload_output_equality_is_strict_merged_is_descriptive(self):
        for mode,value in (('updated',0.),('reloaded',1e-6),('merged',1.)):
            for stream in ('video','audio','rgb'): J.save_array(self.e,mode+'_'+stream,torch.tensor([value]),torch)
        with self.assertRaisesRegex(RuntimeError,'same-mode unmerged reload'):
            P.compare_modes('updated','reloaded',self.e,torch,strict=True)
        row=P.compare_modes('reloaded','merged',self.e,torch)
        self.assertFalse(row['rgb']['torch_equal'])
        self.assertEqual(row['rgb']['acceptance'],'descriptive only; no equivalence tolerance')

    def test_guard_budget_restoration_and_native_tiling(self):
        class Stage:
            def tiled_decode(self,x): return x+1
            def decode(self,x): return self.tiled_decode(x)
        class VAE:
            def __init__(self): self.first_stage_model=Stage()
            def decode(self,x): return self.first_stage_model.decode(x)
            def encode(self,x): return x
            def decode_tiled_3d(self,x): return x
        h3=SimpleNamespace(vae=VAE(),sample=lambda:None)
        snap=J.wrapper_snapshot([(h3,'sample'),(h3.vae,'decode'),(h3.vae,'encode'),
            (h3.vae,'decode_tiled_3d'),(h3.vae.first_stage_model,'decode')])
        with self.assertRaisesRegex(RuntimeError,'second decode'):
            with P.guarded_vae(h3,self.e,torch):
                self.e.phase('zero_decode')
                self.assertTrue(torch.equal(h3.vae.decode(torch.zeros(1)),torch.ones(1)))
                h3.vae.decode(torch.zeros(1))
        self.assertTrue(J.wrappers_restored(snap))
        with P.guarded_vae(h3,self.e,torch):
            with self.assertRaisesRegex(RuntimeError,'fallback'): h3.vae.decode_tiled_3d(torch.zeros(1))
            with self.assertRaisesRegex(RuntimeError,'sampler'): h3.sample()
        self.assertTrue(J.wrappers_restored(snap))

    def test_identity_ignores_timing_not_recipe_and_out_never_overwrites(self):
        keys=('openh3_pretrained','comfy_commit','dit','guide_helper_sha256','jit_compile_false',
            'mm_bmm_format_nd','precision_used','torch','torch_npu','torch_npu_module_to_cast_weight_dropped')
        a={k:k for k in keys};a['dit_load_seconds']=100
        b=copy.deepcopy(a);b['dit_load_seconds']=200
        self.assertEqual(P.stable_pretrained(a),P.stable_pretrained(b))
        b['openh3_pretrained']='other'
        self.assertNotEqual(P.stable_pretrained(a),P.stable_pretrained(b))
        with self.assertRaises(FileExistsError): P.Evidence(self.e.path,{'test':True})


@unittest.skipUnless(torch is not None and os.environ.get('OPENH3_COMFY_REFERENCE_ROOT'),
                    'requires Torch CPU and pinned sampler AST fixture')
class CompleteTinyPath(unittest.TestCase):
    def test_four_actual_sampler_trajectories_RGB_SGD_save_clear_reload_merge(self):
        from test_openh3_train_trajectory import fixture
        from openh3_train_trajectory import ResidentH3Trajectory
        with tempfile.TemporaryDirectory() as tmp:
            e=P.Evidence(Path(tmp)/'run',{'CPU_fixture_only':True})
            # The full 400-key fixture writes every actual array/checkpoint but
            # batches its partial.json metadata. Small tests above exercise
            # production atomic writes. This avoids >1GiB of redundant JSON I/O
            # while testing the unchanged sampler/gradient/SGD/load/merge math.
            flush_metadata=e.save
            e.save=lambda:None
            h3,backend,state=fixture()
            model,names=tiny_model(50)
            h3.patcher.model.diffusion_model=model
            expected=P.expected_adapters(model,torch)
            self.assertEqual(len(expected),400)
            math={'kind':'dit','entered':0,'restored':0}
            class VBlock(torch.nn.Module):
                def forward(self,x):
                    if math['kind']!='vae': raise AssertionError('native VAE context missing')
                    return x+x.sin()*.001
            class Stage(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.frozen=torch.nn.Parameter(torch.tensor(.1),requires_grad=False)
                    self.decoder=torch.nn.Module()
                    self.decoder.transformer_blocks=torch.nn.ModuleList([VBlock() for _ in range(36)])
                def decode(self,x):
                    for block in self.decoder.transformer_blocks: x=block(x)
                    return x.float().mean().sigmoid().reshape(1,1,1,1).expand(5,128,128,3)
            stage=Stage().eval()
            h3.vae=SimpleNamespace(first_stage_model=stage,device=torch.device('cpu'),offload_device=torch.device('cpu'),
                disable_offload=True,vae_dtype=torch.float16,decode=stage.decode)
            @contextmanager
            def decoder_context():
                prior=math['kind'];math['kind']='vae';math['entered']+=1
                try: yield
                finally: math['kind']=prior;math['restored']+=1
            def observe(group,phase,index):
                self.assertTrue(state['math'])
                self.assertEqual(math['kind'],'vae' if group=='VAE' else 'dit')
            generator=torch.Generator().manual_seed(101)
            guide=torch.randn(P.SHAPE,generator=generator)
            raw=torch.randn(P.SHAPE,generator=generator)
            original=J.freeze_snapshot(model,torch,names)
            wrappers=J.wrapper_snapshot([(b,'forward') for b in model.blocks]+[(b,'forward') for b in stage.decoder.transformer_blocks])
            diagnostic={'synthetic':True,'lr':P.LR,'scale':P.LOSS_SCALE}
            pretrained=model._openh3_pretrained_record
            with J.counted_blocks(model.blocks,'DiT',e,observe),J.counted_blocks(stage.decoder.transformer_blocks,'VAE',e,observe):
                for mode in P.MODES:
                    if mode=='updated':
                        P.one_sgd_update(model,expected,e,torch)
                        path=P.save_checkpoint(model,expected,pretrained,diagnostic,e,torch)
                    elif mode=='reloaded': P.clear_reload(model,expected,path,pretrained,diagnostic,e,torch)
                    elif mode=='merged': original=P.merge_once(model,names,original,e,torch)
                    session=ResidentH3Trajectory(h3,_backend=backend,checkpoint_blocks=mode=='zero',
                        checkpoint_decoder=mode=='zero',decoder_context=decoder_context)
                    from contextlib import nullcontext
                    with (nullcontext() if mode=='zero' else torch.no_grad()),session:
                        e.phase(mode)
                        out=session.sample(guide,raw)
                        self.assertEqual(out['contract']['actual_forward_NFE'],3)
                        e.data['actual']['NFE'][mode]+=3
                        if mode=='zero': out['video'].retain_grad()
                        e.phase(mode+'_decode')
                        rgb=session.decode(out['video'])
                        self.assertEqual(math['kind'],'dit')
                        P.save_mode(mode,out,rgb,e,torch)
                        if mode=='zero':
                            # Simulate a separately frozen accepted reference; numerical
                            # fail-fast gates have separate negative tests above/in joint.
                            J.rgb_backward(rgb.detach().clone(),rgb,out['video'],e,torch)
                            J.save_gradients(model,out['video'],e,torch)
                            self.assertTrue(e.data['adapter_gradients']['passed'])
                            self.assertTrue(e.data['retained_video_gradient']['passed'])
                    self.assertFalse(session.active)
                    self.assertIsNone(h3.patcher.model.current_patcher)
                    self.assertTrue(J.check_snapshot(model,original,torch,names)['passed'])
                    if mode=='reloaded': P.compare_modes('updated','reloaded',e,torch,strict=True)
                    if mode=='merged': P.compare_modes('reloaded','merged',e,torch)
            self.assertTrue(J.wrappers_restored(wrappers))
            self.assertEqual(math['entered'],math['restored'])
            self.assertEqual(math['kind'],'dit')
            self.assertFalse(state['math'])
            self.assertEqual(e.data['actual']['NFE'],{m:3 for m in P.MODES})
            self.assertEqual(e.data['actual']['optimizer_steps'],1)
            self.assertEqual(e.data['actual']['completed_backward_calls'],1)
            self.assertEqual(e.data['actual']['checkpoint_saves'],1)
            self.assertEqual(e.data['actual']['clear_reload_cycles'],1)
            self.assertEqual(e.data['actual']['merge_calls'],1)
            flush_metadata()
            for phase in (*P.MODES,'backward'):
                self.assertEqual(e.data['block_forward_calls']['DiT'][phase],{str(i):3 for i in range(50)})
            for phase in (*(m+'_decode' for m in P.MODES),'backward'):
                self.assertEqual(e.data['block_forward_calls']['VAE'][phase],{str(i):1 for i in range(36)})
            print(json.dumps({'CPU_tiny_joint':{'adapter_tensors':400,'trajectories':4,'NFE':12,
                'backward':1,'SGD_steps':1,'strict_unmerged_reload':True,'merge_equivalence_claim':False,
                'actual_VAE_native_context_entries':math['entered'],'actual_context_restorations':math['restored']}}),flush=True)


if __name__=='__main__': unittest.main(verbosity=2)
