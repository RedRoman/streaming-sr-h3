"""Stdlib only: immutable small evidence and semantic failure injection."""
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("vae_audit",ROOT/"code/audit_vae_gradient_probe.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
BASE=ROOT/"results/takeover_20261111"
PRIOR=BASE/"vae_decode_grad_temporal_smoke_20261111c_completed.tar.gz"
PRIOR_SHA="0eb1209b7944311729018bb17027a6055d0da83850b032496713ae53b4cfd640"


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PRIOR.exists():raise unittest.SkipTest("immutable prior-c small archive absent")
        cls.files,cls.info=mod.read_archive(PRIOR,PRIOR_SHA)
        cls.summary,cls.report,cls.queue=mod.inspect_bundle(cls.files,cls.info)
        cls.status=json.loads(cls.files[mod.exactly_one(cls.files,"/queue_status.json")])

    def current(self):
        r,q=copy.deepcopy(self.report),copy.deepcopy(self.queue)
        r["contract"].update(checkpoint_vae_blocks=True,expected_gradient_sha256=r["latent_gradient"]["sha256_float32"])
        r["expected_gradient_sha256_matched"]=True
        r["decoder_checkpoint"]={"schema":"openh3.vae_decoder_checkpoint_context.v1","enabled":True,"active":False,"restored":True,"block_count":36,"expected_block_count":36,"wrapped":36,"recomputed_block_indices":list(range(36)),"use_reentrant":False,"preserve_rng_state":True,"parameter_or_buffer_mutation_performed":False,"checkpoint_forward_calls":72,"recompute_calls":72,"output_gradient_guards":72,"backward_guard_calls":72}
        q["steps"][0]["argv"] += ["--checkpoint-vae-blocks","--expected-gradient-sha256",r["latent_gradient"]["sha256_float32"]]
        q["implementation_sha256"][self.summary["report_member"]]=self.summary["report_sha256"]
        return r,q

    def test_existing_temporal_success_not_full_memory_gate(self):
        self.assertEqual(self.summary["state"],"small_decoder_gradient_passed")
        self.assertEqual(self.summary["full_decoder_memory_gate"],"not_tested_small_geometry")
        self.assertEqual(self.summary["full_H3_DiT_LoRA_memory_gate"],"not_tested")

    def test_completed_checkpoint_new_admission_format_and_prior_exact(self):
        path=BASE/"vae_decode_grad_checkpoint_smoke_20261111e_completed/evidence.tar.gz"
        if not path.exists():self.skipTest("new checkpoint archive not available")
        files,info=mod.read_archive(path,"35faa752dd99814d6da79f05396dce52d46229460f4722ac7cb7ed17ba2987b9")
        result,current,queue=mod.inspect_bundle(files,info)
        self.assertEqual(result["checkpoint_audit"]["recompute_calls"],36)
        compare=mod.compare_prior(current,self.report,queue,self.queue,self.summary["report_sha256"])
        self.assertEqual(compare["state"],"exact_gradient_matched_under_verified_preconditions")

    def test_admission_incomplete_pins_rejected(self):
        files=dict(self.files)
        name=mod.exactly_one(files,"/admission.json");record=json.loads(files[name]);record["pins_verified"]-=1
        files[name]=json.dumps(record).encode()
        with self.assertRaisesRegex(ValueError,"pin count incomplete"):
            mod.inspect_bundle(files,self.info)

    def test_existing_full_failure_not_forward_or_backward_pass(self):
        p=BASE/"vae_decode_grad_fullres_failed_20261111d/evidence_failed.tar.gz"
        files,info=mod.read_archive(p,"03a3aa2d113f09e8075c41893a2fff82a5f2ab814465d02ce811fccc83ae0baf")
        summary,_,_=mod.inspect_bundle(files,info)
        self.assertEqual(summary["full_decoder_memory_gate"],"failed")
        self.assertTrue(summary["no_grad_forward_completed"])
        self.assertFalse(summary["grad_forward_completed"])
        self.assertFalse(summary["backward_returned"])

    def test_bad_external_archive_sha_rejected(self):
        with self.assertRaisesRegex(ValueError,"archive byte size/SHA"):
            mod.read_archive(PRIOR,"0"*64)

    def test_manifest_tamper_rejected_even_with_new_archive_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"bad.tar.gz"
            with tarfile.open(p,"x:gz") as tar:
                for name,data in self.files.items():
                    data=data+b" " if name.endswith("/report.json") else data
                    member=tarfile.TarInfo(name);member.size=len(data)
                    tar.addfile(member,io.BytesIO(data))
            with self.assertRaisesRegex(ValueError,"manifest member mismatch"):
                mod.read_archive(p,mod.sha(p.read_bytes()))

    def test_exact_prior_preconditions_and_temporal_chunk_counts(self):
        r,q=self.current()
        result=mod.validate_report(r,self.status,q)
        self.assertEqual(result["checkpoint_audit"]["checkpoint_forward_calls"],72)
        compare=mod.compare_prior(r,self.report,q,self.queue,self.summary["report_sha256"])
        self.assertEqual(compare["state"],"exact_gradient_matched_under_verified_preconditions")

    def test_prior_latent_sha_change_rejected_despite_equal_grad_sha(self):
        r,q=self.current();r["latent_sha256_float32"]="0"*64
        with self.assertRaisesRegex(ValueError,"input/weights/loss"):
            mod.compare_prior(r,self.report,q,self.queue,self.summary["report_sha256"])

    def test_prior_precision_change_rejected(self):
        r,q=self.current();r["model"]["precision_used"]["conv_hf32"]=True
        with self.assertRaisesRegex(ValueError,"runtime/model"):
            mod.compare_prior(r,self.report,q,self.queue,self.summary["report_sha256"])

    def test_missing_recomputed_block_rejected(self):
        r,q=self.current();r["decoder_checkpoint"]["recomputed_block_indices"].pop()
        with self.assertRaisesRegex(ValueError,"block coverage"):
            mod.validate_report(r,self.status,q)

    def test_unrestored_scope_rejected(self):
        r,q=self.current();r["decoder_checkpoint"]["restored"]=False
        with self.assertRaisesRegex(ValueError,"safely restored"):
            mod.validate_report(r,self.status,q)

    def test_complete_missing_backward_rejected(self):
        r,q=self.current();r["actual_completed_backwards"]=0
        with self.assertRaisesRegex(ValueError,"both forwards and all planned backwards"):
            mod.validate_report(r,self.status,q)

    def paired(self):
        r,q=self.current();scale=1024
        r["contract"].update(schema=mod.PAIR_SCHEMA,diagnostic_kind="loss_scale_pair",paired_loss_scale_diagnostic=scale,expected_gradient_sha256=None,planned_backward_calls=2)
        argv=q["steps"][0]["argv"];i=argv.index("--expected-gradient-sha256");del argv[i:i+2]
        argv += ["--diagnose-loss-scale",str(scale)]
        r.update(actual_completed_backwards=2,gradient_loss_scale=scale,expected_gradient_sha256_matched=None)
        r["decoder_checkpoint"].update(checkpoint_forward_calls=36,recompute_calls=72,backward_guard_calls=72,output_gradient_guards=36)
        g=r["latent_gradient"];shape=r["contract"]["latent_shape"];n=math.prod(shape)
        def stats(zero=False,scale_factor=1):
            nonzero=0 if zero else g['nonzero_count'];zeros=n-nonzero
            return {'present':True,'shape':shape,'elements':n,'dtype':'torch.float32','zero_count':zeros,'zero_fraction':zeros/n,
                'finite_count':n,'nan_count':0,'inf_count':0,'finite_nonzero_count':nonzero,
                'finite_l2':0 if zero else g['l2']*scale_factor,'finite_max_abs':0 if zero else g['max_abs']*scale_factor,
                'finite_min_abs_nonzero':None if zero else 1e-9*scale_factor,
                'sha256_float32':'0'*64 if zero else ('1'*64 if scale_factor!=1 else g['sha256_float32'])}
        r['gradient_diagnostics']={'unscaled_loss_backward':stats(zero=True),'scaled_loss_backward_raw':stats(scale_factor=scale),'scaled_loss_backward_unscaled_float32':stats()}
        return r,q

    def test_v2_zero_recovery_is_diagnostic_not_full_training_memory_pass(self):
        r,q=self.paired()
        result=mod.validate_report(r,self.status,q)
        self.assertEqual(result['state'],'loss_scale_pair_diagnostic_completed')
        self.assertEqual(result['full_decoder_memory_gate'],'not_established_by_paired_backward')
        self.assertEqual(result['paired_scale_gradient_gate'],'passed_numerical_probe')
        self.assertEqual(result['loss_scale_pair']['states']['unscaled_loss_backward'],'all_zero')
        self.assertEqual(result['loss_scale_pair']['interpretation'],'scale_recovers_nonzero_gradient_supports_numerical_scale_sensitivity')

    def test_full_size_v2_still_does_not_establish_single_backward_memory(self):
        r,q=self.paired();shape=[1,24,7,30,52];n=math.prod(shape)
        r['contract'].update(latent_shape=shape,expected_rgb_shape=[22,480,832,3])
        r['decoded_shape']=[22,480,832,3];r['latent_gradient']['elements']=n
        for stats in r['gradient_diagnostics'].values():
            stats.update(shape=shape,elements=n,finite_count=n,zero_count=n-stats['finite_nonzero_count'])
            stats['zero_fraction']=stats['zero_count']/n
        q['steps'][0]['argv']+=['--geometry','full']
        result=mod.validate_report(r,self.status,q)
        self.assertEqual(result['full_decoder_memory_gate'],'not_established_by_paired_backward')
        self.assertEqual(result['single_backward_training_memory_gate'],'not_tested_by_paired_diagnostic')
        self.assertEqual(result['paired_scale_gradient_gate'],'passed_numerical_probe')

    def test_v1_two_backwards_not_silently_accepted(self):
        r,q=self.current();r['actual_completed_backwards']=2
        with self.assertRaisesRegex(ValueError,'completed-call'):
            mod.validate_report(r,self.status,q)

    def test_v2_scaled_unscale_ratio_mismatch_rejected(self):
        r,q=self.paired();r['gradient_diagnostics']['scaled_loss_backward_raw']['finite_l2']*=2
        with self.assertRaisesRegex(ValueError,'magnitude mismatch'):
            mod.validate_report(r,self.status,q)

    def test_v2_missing_second_statistics_rejected(self):
        r,q=self.paired();del r['gradient_diagnostics']['scaled_loss_backward_raw']
        with self.assertRaisesRegex(ValueError,'second backward raw'):
            mod.validate_report(r,self.status,q)

    def test_v2_zero_scaled_failure_is_preserved_not_recovery(self):
        r,q=self.paired();r.update(complete=False,error='FloatingPointError: gradient missing, nonfinite or zero')
        r['gradient_diagnostics']['scaled_loss_backward_raw']=copy.deepcopy(r['gradient_diagnostics']['unscaled_loss_backward'])
        del r['gradient_diagnostics']['scaled_loss_backward_unscaled_float32']
        status=copy.deepcopy(self.status);status['state']='failed'
        result=mod.validate_report(r,status,q)
        self.assertEqual(result['state'],'failed')
        self.assertEqual(result['loss_scale_pair']['states']['scaled_loss_backward_raw'],'all_zero')
        self.assertNotIn('recovers',result['loss_scale_pair']['interpretation'])

    def test_v2_nan_and_missing_are_distinct(self):
        r,q=self.paired();shape=r['contract']['latent_shape'];n=math.prod(shape)
        self.assertEqual(mod.gradient_diagnostic_state({'present':False},shape),'missing')
        stats=copy.deepcopy(r['gradient_diagnostics']['unscaled_loss_backward'])
        stats.update(zero_count=n-1,zero_fraction=(n-1)/n,finite_count=n-1,nan_count=1)
        self.assertEqual(mod.gradient_diagnostic_state(stats,shape),'nonfinite')

    def scaled_single(self):
        prior,prior_queue=self.paired();r,q=copy.deepcopy(prior),copy.deepcopy(prior_queue)
        scale=r['contract']['paired_loss_scale_diagnostic']
        r['contract'].update(schema=mod.SCALED_SINGLE_SCHEMA,diagnostic_kind='scaled_single_backward',paired_loss_scale_diagnostic=None,fixed_loss_scale=scale,planned_backward_calls=1,expected_gradient_sha256=r['latent_gradient']['sha256_float32'])
        r.update(actual_completed_backwards=1,expected_gradient_sha256_matched=True)
        r['decoder_checkpoint'].update(recompute_calls=36,backward_guard_calls=36)
        del r['gradient_diagnostics']['unscaled_loss_backward']
        argv=q['steps'][0]['argv'];i=argv.index('--diagnose-loss-scale');del argv[i:i+2]
        argv+=['--loss-scale',str(scale),'--expected-gradient-sha256',r['latent_gradient']['sha256_float32']]
        q['implementation_sha256']['runs/synthetic_prior_g/report.json']='3'*64
        return r,q,prior,prior_queue

    def test_v3_one_backward_and_declared_v2_reference_comparison(self):
        r,q,prior,pq=self.scaled_single()
        result=mod.validate_report(r,self.status,q)
        self.assertEqual(result['scaled_single_gradient_gate'],'passed_numerical_probe')
        self.assertEqual(result['actual_completed_backwards'],1)
        compare=mod.compare_prior(r,prior,q,pq,'3'*64)
        self.assertEqual(compare['comparison_kind'],'v3_single_scaled_vs_v2_paired_scaled')

    def test_v3_full_memory_gate_requires_single_backward(self):
        r,q,_,_=self.scaled_single();shape=[1,24,7,30,52];n=math.prod(shape)
        r['contract'].update(latent_shape=shape,expected_rgb_shape=[22,480,832,3])
        r['decoded_shape']=[22,480,832,3];r['latent_gradient']['elements']=n
        for stats in r['gradient_diagnostics'].values():
            stats.update(shape=shape,elements=n,finite_count=n,zero_count=n-stats['finite_nonzero_count'])
            stats['zero_fraction']=stats['zero_count']/n
        q['steps'][0]['argv']+=['--geometry','full']
        result=mod.validate_report(r,self.status,q)
        self.assertEqual(result['full_decoder_memory_gate'],'passed_decoder_only')
        self.assertEqual(result['full_H3_DiT_LoRA_memory_gate'],'not_tested')
        r['actual_completed_backwards']=2
        with self.assertRaisesRegex(ValueError,'completed-call'):
            mod.validate_report(r,self.status,q)

    def test_v3_vs_v2_different_scales_rejected(self):
        r,q,prior,pq=self.scaled_single();prior['contract']['paired_loss_scale_diagnostic']=512
        with self.assertRaisesRegex(ValueError,'effective loss scale'):
            mod.compare_prior(r,prior,q,pq,'3'*64)

    def test_v3_vs_v2_raw_gradient_sha_mismatch_rejected(self):
        r,q,prior,pq=self.scaled_single();r['gradient_diagnostics']['scaled_loss_backward_raw']['sha256_float32']='4'*64
        with self.assertRaisesRegex(ValueError,'statistics differ'):
            mod.compare_prior(r,prior,q,pq,'3'*64)

    def test_v3_raw_unscaled_ratio_mismatch_rejected(self):
        r,q,_,_=self.scaled_single();r['gradient_diagnostics']['scaled_loss_backward_raw']['finite_l2']*=2
        with self.assertRaisesRegex(ValueError,'magnitude mismatch'):
            mod.validate_report(r,self.status,q)

    def test_v1_loss_scale_cannot_silently_change(self):
        r,q=self.current();r['contract']['fixed_loss_scale']=1024
        with self.assertRaisesRegex(ValueError,'v1 may not silently use fixed'):
            mod.validate_report(r,self.status,q)

    def test_nonfinite_gradient_rejected(self):
        r,q=self.current();r["latent_gradient"]["finite"]=False
        with self.assertRaisesRegex(ValueError,"gradient missing"):
            mod.validate_report(r,self.status,q)

    def test_weights_mutation_rejected(self):
        r,q=self.current();r["VAE_parameter_sha256_after"]="0"*64
        with self.assertRaisesRegex(ValueError,"weights changed"):
            mod.validate_report(r,self.status,q)

    def test_full_decoder_gate_requires_recorded_memory_and_never_joint_gate(self):
        r,q=self.current()
        r["contract"].update(latent_shape=[1,24,7,30,52],expected_rgb_shape=[22,480,832,3])
        r["decoded_shape"]=[22,480,832,3]
        r["latent_gradient"]["elements"]=1*24*7*30*52
        q["steps"][0]["argv"] += ["--geometry","full"]
        result=mod.validate_report(r,self.status,q)
        self.assertEqual(result["full_decoder_memory_gate"],"passed_decoder_only")
        self.assertEqual(result["full_H3_DiT_LoRA_memory_gate"],"not_tested")
        r["npu_peak_reserved_bytes_after_load"]=0
        with self.assertRaisesRegex(ValueError,"peak memory"):
            mod.validate_report(r,self.status,q)

    def test_pack_writes_new_archive_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"root";root.mkdir()
            for name,data in self.files.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            queue_name=mod.exactly_one(self.files,"/queue.json");prep=str(Path(queue_name).parent)
            status_name=mod.exactly_one(self.files,"/queue_status.json");run=str(Path(status_name).parent)
            # Historical archive omitted the small asset manifest. A dummy byte
            # file here tests packing only, not a scientific/model assertion.
            manifest=mod.option(self.queue["steps"][0]["argv"],"--pretrained-manifest")
            path=root/manifest;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"{}\n")
            out=Path(temp)/"packed.tar.gz"
            result=mod.pack(root,run,prep,out)
            packed,_=mod.read_archive(out,result["archive_sha256"])
            self.assertIn(manifest,packed)
            for name,data in self.files.items():self.assertEqual((root/name).read_bytes(),data)
            with self.assertRaisesRegex(ValueError,"existing archive"):
                mod.pack(root,run,prep,out)


if __name__=="__main__":unittest.main()
