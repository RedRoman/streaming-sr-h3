"""Synthetic metadata only: no saved schedule, pixels, labels or model imports."""
import copy,importlib.util,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'prepare_input_preservation_donors.py'
spec=importlib.util.spec_from_file_location('donors',P);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def row(i,sid,group=None,t=30,ext=(1,1),profile=M.PROFILES[0]):
    r={'row_index':i,'source_id':sid,'source_group_id':group or sid,'source_sha256':M.value_sha(sid),'split':'train',
       'frame_index':t,'history_indices':[max(t-2,0),max(t-1,0),t],'true_availability':[int(t>=2),int(t>=1)],
       'external_mask':list(ext),'profile_id':profile,'severity_phase':M.phase(profile,t),'teacher_mask':None}
    r['row_sha256']=M.value_sha(r);return r

def changed(r,**kw):
    r={**r,**kw};r['row_sha256']=M.value_sha({k:v for k,v in r.items() if k!='row_sha256'});return r

def reseal(m):m['donor_map_sha256']=M.value_sha({k:v for k,v in m.items() if k!='donor_map_sha256'});return m

class DonorTests(unittest.TestCase):
    def test_full16000_one_map_per_row_including_both_absent(self):
        masks=((1,1),(0,1),(1,0),(0,0));rows=[row(i,str(i%4),ext=masks[(i//4)%4]) for i in range(16000)]
        m=M.build_donor_map(rows);v=M.validate_donor_map(rows,m)
        self.assertTrue(v['bijection_verified']);self.assertEqual(m['mapped_rows'],16000);self.assertEqual(m['both_absent_rows'],4000);self.assertEqual(len(m['strata']),4)
        self.assertEqual({e['recipient']['row_index'] for e in m['entries']},{e['donor']['row_index'] for e in m['entries']})
    def test_effective_boundary_patterns_merge_and_zero_rows_remain(self):
        rows=[row(0,'a',t=1,ext=(1,1)),row(1,'b',t=1,ext=(0,1)),row(2,'a',t=0),row(3,'b',t=30,ext=(0,0))]
        m=M.build_donor_map(rows,expected_rows=4);self.assertTrue(m['donor_ready']);self.assertEqual(len(m['strata']),2);self.assertEqual(m['both_absent_rows'],2)
        self.assertEqual({tuple(s['stratum'][2]) for s in m['strata']},{(0,1),(0,0)})
    def test_shared_group_exclusion_and_seed_determinism(self):
        rows=[row(0,'a','shared'),row(1,'b','shared'),row(2,'c','other'),row(3,'d','other')]
        m=M.build_donor_map(rows,expected_rows=4);self.assertEqual(m,M.build_donor_map(rows,expected_rows=4))
        self.assertTrue(all(e['recipient']['source_group_id']!=e['donor']['source_group_id'] for e in m['entries']))
        with self.assertRaises(ValueError):M.build_donor_map(rows,seed=1,expected_rows=4)
    def test_any_infeasible_stratum_blocks_entire_mapping(self):
        rows=[row(0,'a'),row(1,'a'),row(2,'b'),row(3,'c',ext=(0,0)),row(4,'d',ext=(0,0))]
        m=M.build_donor_map(rows,expected_rows=5);self.assertFalse(m['donor_ready']);self.assertTrue(m['both_head_training_arms_blocked']);self.assertEqual(m['entries'],[])
        self.assertEqual(len(m['impossible_strata']),1);self.assertEqual(m['impossible_strata'][0]['recipient_row_indices'],[0,1,2])
        self.assertEqual(M.validate_donor_map(rows,m,5)['mapped_rows'],0)
    def test_validator_rejects_duplicate_self_and_cross_stratum_donors(self):
        rows=[row(i,chr(97+i%4),ext=(1,1) if i<4 else (0,0)) for i in range(8)];m=M.build_donor_map(rows,expected_rows=8)
        bad=copy.deepcopy(m);bad['entries'][0]['donor']=bad['entries'][1]['donor'];reseal(bad)
        with self.assertRaises(ValueError):M.validate_donor_map(rows,bad,8)
        bad=copy.deepcopy(m);bad['entries'][0]['donor']=bad['entries'][0]['recipient'];reseal(bad)
        with self.assertRaises(ValueError):M.validate_donor_map(rows,bad,8)
        units=M.units_from_rows(rows,8);bad=copy.deepcopy(m);bad['entries'][0]['donor']=units[5];reseal(bad)
        with self.assertRaisesRegex(ValueError,'cross-stratum'):M.validate_donor_map(rows,bad,8)
    def test_bad_role_hash_group_or_partial_schedule_rejected(self):
        base=[row(0,'a'),row(1,'b')]
        for invalid in (dict(base[0],source_id='bad'),changed(base[0],split='dev'),changed(base[0],true_availability=[0,1]),changed(base[0],teacher_mask={'kind':'all_available'})):
            with self.assertRaises(ValueError):M.build_donor_map([invalid,base[1]],expected_rows=2)
        with self.assertRaises(ValueError):M.build_donor_map([row(0,'a','g1'),row(1,'a','g2')],expected_rows=2)
        with self.assertRaises(ValueError):M.build_donor_map(base)
    def test_family_phase_strata_are_never_coarsened(self):
        rows=[]
        for profile in (*M.PROFILES,'identity','near_clean'):
            for t in (10,30,60,80):
                for sid in ('a','b'):rows.append(row(len(rows),sid,t=t,profile=profile))
        m=M.build_donor_map(rows,expected_rows=len(rows));self.assertTrue(m['donor_ready'])
        self.assertEqual(len(m['strata']),9)
        self.assertTrue(all(e['recipient']['stratum']==e['donor']['stratum'] for e in m['entries']))
    def test_new_artifact_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json';r=M.write_new(p,{'status':'fixture'});self.assertEqual(r['file_sha256'],M.file_sha(p))
            with self.assertRaises(FileExistsError):M.write_new(p,{'status':'replace'})

if __name__=='__main__':unittest.main(verbosity=2)
