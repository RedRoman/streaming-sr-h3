"""Small CPU tests. No H3/NPU import or model weights are needed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import diagnose_cached_rgb_compensation as module
from diagnose_rphi_residual_mix import array_sha, canonical_sha, sha_file


class CandidateTests(unittest.TestCase):
    def inputs(self):
        shape=(2,3,4,3)
        return [np.full(shape,x,np.float32) for x in (.2,.6,.5)]

    def test_endpoints_two_families_and_target_absence(self):
        r,g,v=self.inputs()
        for family in module.FAMILIES:
            self.assertTrue(np.array_equal(module.construct_candidate(r,g,v,0,family)[1],r))
        self.assertTrue(np.array_equal(module.construct_candidate(r,g,v,1,'scalar')[1],g))
        compensated=module.construct_candidate(r,g,v,1,'vae_compensation')[1]
        self.assertFalse(np.array_equal(compensated,g))
        np.testing.assert_allclose(compensated,.3,rtol=0,atol=1e-7)
        with self.assertRaises(TypeError):
            module.construct_candidate(r,g,v,1,'scalar',target=np.zeros_like(r))
        with self.assertRaises(ValueError):
            module.construct_candidate(r,g,v,.3,'scalar')
        for a,value in zip((r,g,v),(.2,.6,.5)):
            np.testing.assert_array_equal(a,np.full(a.shape,value,np.float32))

    def test_native_anchor_exact_and_compensation_clipping_statistics(self):
        r,g,v=self.inputs()
        post=module.construct_candidate(r,g,v,.75,'scalar')[1]
        expected=(g.astype(np.float64)+.25*(r.astype(np.float64)-g)).astype(np.float32)
        np.testing.assert_array_equal(post,expected)
        r.fill(.9);g.fill(.9);v.fill(0)
        pre,post,stats=module.construct_candidate(r,g,v,1,'vae_compensation')
        self.assertGreater(pre.min(),1)
        self.assertTrue(np.all(post==1))
        self.assertEqual(stats['clipped_component_fraction'],1)
        self.assertEqual(stats['clipped_pixel_fraction'],1)
        self.assertEqual(stats['changed_pixel_fraction_vs_R'],1)
        self.assertEqual(module.construct_candidate(r,g,v,0,'scalar')[2]['changed_pixel_fraction_vs_R'],0)

    def test_all_grid_metrics_common_support_and_baseline_gates(self):
        r,g,v=self.inputs(); target=np.full(r.shape,102,np.uint8)
        support=[object()]
        real_rgb=module.rgb_metrics
        def rgb(a,b,metric=None):
            out=real_rgb(a,b)
            out['lpips']=out['mae'] if metric is not None else None
            return out
        seen=[]
        def temporal(a,b,s):
            self.assertIs(s,support); seen.append(id(s))
            return {'pairs':1,'valid_pairs':1,'valid_coverage':1.,
                    'unwarped_rgb_error_change_mae':0.,'flow_aligned_rgb_error_change_mae':0.}
        gt=target.astype(np.float32)/np.float32(255)
        def score(a): return {'quality':rgb(a,gt,True),'temporal':temporal(a,gt,support)}
        anchor=module.construct_candidate(r,g,v,.75,'scalar')[1]
        prior={'task':'denoise','arms':{'rphi_only':score(r),'h3_rphi':score(g),'h3_rphi_anchor':score(anchor)}}
        vr={'arms':{'rphi_as_returned_internal_clipped':score(v)}}
        with patch.object(module,'rgb_metrics',rgb),patch.object(module,'temporal_metrics',temporal):
            result=module.score_grid(r,g,v,target,True,support,prior,vr)
        self.assertEqual(len(result['grid']),14)
        self.assertEqual([(x['family'],x['beta']) for x in result['grid']],[(f,b) for f in module.FAMILIES for b in module.GRID])
        self.assertTrue(all(x['passed'] for x in result['reproduction_checks'].values()))
        self.assertTrue(seen)
        corrupt=copy.deepcopy(prior);corrupt['arms']['rphi_only']['quality']['mse']+=.01
        with patch.object(module,'rgb_metrics',rgb),patch.object(module,'temporal_metrics',temporal):
            with self.assertRaisesRegex(ValueError,'reproduction mismatch'):
                module.score_grid(r,g,v,target,True,support,corrupt,vr)


class IntegrityTests(unittest.TestCase):
    def test_reused_flow_support_cannot_be_replaced_silently(self):
        flow={'support_sha256':'a'*64,'target_pixel_sha256':'b'*64,'pairs':21}
        self.assertEqual(module.accept_common_flow(copy.deepcopy(flow),flow),flow)
        for key,value in [('support_sha256','c'*64),('target_pixel_sha256','c'*64),('pairs',20)]:
            with self.assertRaisesRegex(ValueError,'flow support changed'):
                module.accept_common_flow(flow,{**flow,key:value})

    def test_actual_file_array_sha_and_geometry_are_required(self):
        a=np.full((2,3,4,3),.5,np.float32)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'cache.npz';np.savez(p,rphi=a)
            expected={'rphi':array_sha(a)}
            with patch.object(module,'GEOMETRY',list(a.shape)):
                actual=module.verify_cached_npz(p,sha_file(p),expected)
                np.testing.assert_array_equal(actual['rphi'],a)
                with self.assertRaisesRegex(ValueError,'array SHA'):
                    module.verify_cached_npz(p,sha_file(p),{'rphi':'0'*64})
                with self.assertRaisesRegex(ValueError,'file SHA'):
                    module.verify_cached_npz(p,'0'*64,expected)
            with self.assertRaisesRegex(ValueError,'geometry'):
                module.verify_cached_npz(p,sha_file(p),expected)

    def test_no_overwrite_and_new_directory_hash_bound_reuse(self):
        contract={'cache_evidence':[{'scope':'ordinary','sample_id':'one'}]}
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'new'
            result=module.open_result(out,contract)
            with self.assertRaises(FileExistsError):module.open_result(out,contract)
            row={'scope':'ordinary','sample_id':'one','evidence':contract['cache_evidence'][0],
                 'grid':[{'family':f,'beta':b} for f in module.FAMILIES for b in module.GRID]}
            result['per_sample']=[row];result['sample_receipts']=[canonical_sha(row)]
            partial=out/'partial_metrics.json';partial.write_text(json.dumps(result))
            original=partial.read_bytes()
            self.assertEqual(module.open_result(Path(temp)/'reuse',contract,partial,sha_file(partial))['per_sample'],[row])
            self.assertEqual(partial.read_bytes(),original)
            with self.assertRaisesRegex(ValueError,'contract mismatch'):
                module.open_result(Path(temp)/'bad', {**contract,'changed':True},partial,sha_file(partial))
            with self.assertRaisesRegex(ValueError,'file SHA'):
                module.open_result(Path(temp)/'badsha',contract,partial,'0'*64)
            result['per_sample'][0]['sample_id']='wrong';partial.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'coverage mismatch'):
                module.open_result(Path(temp)/'badcoverage',contract,partial,sha_file(partial))
            (out/'metrics.json').write_text('{}')
            with self.assertRaises(FileExistsError):module.open_result(out,contract)

    def test_exact_vae_guide_latent_and_scope_binding(self):
        base={'sample_id':'one','source_id':'src','source_sha256':'a'*64,'task':'sr','target_pixel_sha256':'b'*64}
        exported={**base,'guide_file_sha256':'c'*64}
        sampling={'guide_pixel_sha256':{'raw':'d'*64,'rphi':'e'*64},'calls':{
            k:{'guide_latent_sha256_float32':k*16} for k in ('raw','rphi')}}
        item={'scope':'ordinary','export':exported,'reference':{**base,'sampling':sampling}}
        row={**base,'scope':'ordinary','guide_file_sha256':'c'*64,'baseline_metric_parity':True,
             'frame_indices':list(range(22)),'observations':{k:{'latent_matches_prior_H3_exactly':True,
               'clamp_parity_exact':True,'clamp_parity_max_abs_error':0,
               'guide_pixel_sha256':sampling['guide_pixel_sha256'][k],
               'encoded_latent_sha256_float32':sampling['calls'][k]['guide_latent_sha256_float32'],
               'encoded_latent_shape':[1,24,7,30,52]} for k in ('raw','rphi')}}
        module.validate_vae_row(row,item)
        for mutation in ('latent','source','scope'):
            bad=copy.deepcopy(row)
            if mutation=='latent':bad['observations']['rphi']['encoded_latent_sha256_float32']='wrong'
            elif mutation=='source':bad['source_sha256']='wrong'
            else:bad['scope']='controls'
            with self.assertRaises(ValueError):module.validate_vae_row(bad,item)


if __name__=='__main__':
    unittest.main()
