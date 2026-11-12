"""Synthetic cached-head training boundaries; no real dataset/restorer/NPU."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_input_preservation_heads as M
import prepare_input_preservation_donors as D


def toy_row(i):
    sid=str(i%2);r={'row_index':i,'source_id':sid,'source_group_id':sid,'source_sha256':D.value_sha(sid),
        'split':'train','frame_index':0,'history_indices':[0,0,0],'true_availability':[0,0],
        'external_mask':[1,1],'profile_id':D.PROFILES[0],'severity_phase':'static','teacher_mask':None}
    r['row_sha256']=D.value_sha(r);return r


def fixture_cache(folder):
    folder=Path(folder);(folder/'arrays').mkdir();rows=[toy_row(i) for i in range(32)]
    mapping=D.build_donor_map(rows,expected_rows=32);entries=[];records=[]
    for bi in range(2):
        values={'raw_features':np.full((16,2,11,6,6),bi+.5,dtype=np.float32),
            'normalized_benefit':np.full((16,1,6,6),.5-bi,dtype=np.float32),
            'availability':np.zeros((16,2),dtype=np.float32),'k1_rgb':np.full((16,3,4,4),np.nan,dtype=np.float32),
            **{k:np.zeros((16,1,6,6),dtype=np.float64) for k in ('input_mae','k1_mae','raw_benefit')}}
        p=folder/'arrays'/f'batch_{bi:04d}.npz';np.savez(p,**values)
        selected=rows[bi*16:(bi+1)*16]
        entry={'path':str(p),'bytes':p.stat().st_size,'sha256':M.T.file_sha(p),'batch_index':bi,
            'row_indices':[r['row_index'] for r in selected],'row_sha256':[r['row_sha256'] for r in selected],
            'arrays':{k:{'dtype':str(v.dtype),'shape':list(v.shape),'sha256':M.array_sha(v)} for k,v in values.items()}}
        entries.append(entry)
        for j,r in enumerate(selected):
            rec={**r,'batch_index':bi,'batch_row':j,'cache_role':'train','usable_for_training':True,
                'cache_file_sha256':entry['sha256'],'array_row_sha256':{k:M.array_sha(v[j]) for k,v in values.items()}}
            rec['cache_record_sha256']=M.T.value_sha(rec);records.append(rec)
    manifest={}
    for name,lines in [('batches',entries),('records',records)]:
        p=folder/(name+'.jsonl');p.write_text(''.join(json.dumps(x)+'\n' for x in lines))
        manifest[name]={'path':str(p),'bytes':p.stat().st_size,'sha256':M.T.file_sha(p)}
    return manifest,rows,mapping


class NumericBoundaries(unittest.TestCase):
    def test_only_whole_labels_permuted_zero_availability_retained(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,rows,mapping=fixture_cache(d)
            x,y,donors,p=M.load_training_arrays(manifest,rows,mapping,{},expected_rows=32)
            self.assertEqual(x.shape,(32,2,11,6,6));self.assertEqual(p['K1_or_GT_arrays_loaded'],0)
            self.assertEqual(p['dev_NPZ_files_opened'],0);self.assertTrue(p['donor_invariants']['bijection_verified'])
            self.assertEqual(sorted(donors.tolist()),list(range(32)))
            a,ai=M.selected_targets(y,donors,0,'matched');b,bi=M.selected_targets(y,donors,0,'shuffled')
            self.assertTrue(np.array_equal(a,y[:16]));self.assertTrue(np.array_equal(b,y[bi]))
            self.assertEqual(len(ai),16);self.assertTrue(all(row['true_availability']==[0,0] for row in rows))

    def test_tampered_actual_cache_file_fails_before_training(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,rows,mapping=fixture_cache(d);p=Path(d)/'arrays/batch_0000.npz'
            with p.open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'NPZ differs'):M.load_training_arrays(manifest,rows,mapping,{},expected_rows=32)

    def test_record_dev_role_or_wrong_label_sha_rejected(self):
        for mutation in ['role','label']:
            with tempfile.TemporaryDirectory() as d:
                manifest,rows,mapping=fixture_cache(d);p=Path(manifest['records']['path'])
                records=[json.loads(l) for l in p.read_text().splitlines()]
                if mutation=='role':records[0]['cache_role']='dev'
                else:records[0]['array_row_sha256']['normalized_benefit']='0'*64
                records[0]['cache_record_sha256']=M.T.value_sha({k:v for k,v in records[0].items() if k!='cache_record_sha256'})
                p.write_text(''.join(json.dumps(r)+'\n' for r in records));manifest['records'].update(bytes=p.stat().st_size,sha256=M.T.file_sha(p))
                with self.assertRaisesRegex(ValueError,'provenance'):M.load_training_arrays(manifest,rows,mapping,{},expected_rows=32)

    def test_actual_donor_audit_schema_and_missing_00_gate(self):
        files={k:{'path':k,'sha256':k} for k in ('donor_manifest','donor_map','student_schedule')}
        audit={'schema':'rphi.input_preservation_donor_independent_actual_audit.v1','complete':True,'donor_ready':True,
            'input_bytes_unchanged':True,'inputs':{n:{**files[k],'bytes':1} for n,k in [('manifest','donor_manifest'),('mapping','donor_map'),('schedule','student_schedule')]},
            'validation':{'actual_rows':16000,'row_semantic_SHA_verified':16000,'actual_source_count':19,'actual_strata':36,
                'actual_both_absent_rows':1815,**{k:True for k in ('nonself_cross_source_AND_group','all_recipients_and_donors_exactly_once',
                    'all_strata_frequency_preserved','declared_seeded_rotation_verified','profile_phase_effective_availability_exact')}}}
        M.check_donor_audit(audit,files);audit['validation']['actual_both_absent_rows']=0
        with self.assertRaises(ValueError):M.check_donor_audit(audit,files)

    def test_comparison_identity_arm_only_not_numerical_changes(self):
        c={'arm':'matched','admission':{'fixture':1},'mode':'research','backend_smokes':{},'optimizer':M.OPTIMIZER,'loss':M.LOSS}
        other=copy.deepcopy(c);other.update(arm='shuffled',admission={'fixture':2})
        self.assertEqual(M.comparison_identity(c),M.comparison_identity(other))
        other['optimizer']['lr']=1e-3;self.assertNotEqual(M.comparison_identity(c),M.comparison_identity(other))


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'local Torch missing; exact CPU fixtures remain required')
class TorchTraining(unittest.TestCase):
    def setUp(self):
        import torch
        from rphi_input_preservation import InputPreservationHead
        self.t=torch;torch.set_num_threads(2);self.H=InputPreservationHead

    def counts(self):return {k:0 for k in ('forward_attempts','forwards_completed','backward_attempts','backwards_completed','optimizer_attempts','optimizer_updates')}

    def test_same_init_actual_two_updates_label_only_difference_and_tail_gradient(self):
        t=self.t;a,b=self.H(),self.H();self.assertTrue(M.T.tree_equal(a.state_dict(),b.state_dict()))
        initial=M.T.state_sha(a.state_dict());x=t.full((16,2,11,6,6),.25)
        oa,ob=M.make_optimizer(a,t),M.make_optimizer(b,t);ca,cb=self.counts(),self.counts()
        for step in (1,2):
            ra=M.training_step(a,oa,x,t.ones((16,1,6,6)),step,ca,t)
            rb=M.training_step(b,ob,x,-t.ones((16,1,6,6)),step,cb,t)
        self.assertTrue(all(v==2 for v in ca.values()));self.assertEqual(ca,cb)
        self.assertTrue(any(v>0 for v in ra['tail_gradient_norms'].values()))
        self.assertNotEqual(M.T.state_sha(a.state_dict()),initial);self.assertNotEqual(M.T.state_sha(a.state_dict()),M.T.state_sha(b.state_dict()))
        self.assertEqual(ra['lr'],rb['lr']);self.assertEqual(ra['lr'],1e-4*(2/50))

    def test_actual_serialization_clear_reload_full_optimizer_rng_and_no_overwrite(self):
        t=self.t;m=self.H();o=M.make_optimizer(m,t);initial=M.T.state_sha(m.state_dict())
        for step in (1,2):M.training_step(m,o,t.ones((16,2,11,6,6)),t.ones((16,1,6,6)),step,self.counts(),t)
        c={'arm':'matched','mode':'backend_smoke','files':{'fixture':{}},'backend_smokes':{},'admission':{}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'backend.pt';r,s=M.save_checkpoint(p,m,o,2,c,initial,{'device':'cpu'},t)
            self.assertEqual(s['artifact_role'],'backend_smoke_not_research_resume');self.assertEqual(s['row_cursor'],32)
            with self.assertRaisesRegex(ValueError,'overwrite'):M.save_checkpoint(p,m,o,2,c,initial,{},t)
            del m,o
            self.assertTrue(M.verify_fresh_reload(s,t));self.assertEqual(len(s['optimizer_state_dict']['state']),4)

    def test_nonfinite_features_or_labels_do_not_update(self):
        t=self.t;m=self.H();o=M.make_optimizer(m,t);before=M.T.state_sha(m.state_dict());counts=self.counts()
        with self.assertRaises(ValueError):M.training_step(m,o,t.ones((16,2,11,6,6)),t.full((16,1,6,6),float('nan')),1,counts,t)
        self.assertEqual(counts['optimizer_updates'],0);self.assertEqual(before,M.T.state_sha(m.state_dict()))


if __name__=='__main__':unittest.main(verbosity=2)
