"""Synthetic metadata only; no NPZ or models. Does not claim real data validation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from collections import Counter

P=Path(__file__).resolve().parents[1]/'prepare_history_utility_schedules.py'
spec=importlib.util.spec_from_file_location('prepare_schedules',P);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def row(i,sid,group=None,t=30,mask=(1,1),profile=M.PROFILES[0]):
    x={'row_index':i,'source_id':sid,'source_group_id':group or sid,'source_sha256':M.value_sha(sid),'split':'train',
       'frame_index':t,'history_indices':[max(t-2,0),max(t-1,0),t],'true_availability':[int(t>=2),int(t>=1)],
       'external_mask':list(mask),'profile_id':profile,'severity_phase':M.phase_for(profile,t),'teacher_mask':None}
    x['row_sha256']=M.value_sha(x);return x

def changed(r,**kw):
    x={**r,**kw};x['row_sha256']=M.value_sha({k:v for k,v in x.items() if k!='row_sha256'});return x

class Donors(unittest.TestCase):
    def test_bijection_whole_maps_seed_determinism(self):
        rows=[row(i,chr(65+i%4)) for i in range(28)]
        a=M.donor_map(rows);self.assertEqual(a,M.donor_map(rows));self.assertTrue(a['donor_ready'])
        recipients={(e['recipient']['row_index'],e['recipient']['slot']) for e in a['entries']}
        donors={(e['donor']['row_index'],e['donor']['slot']) for e in a['entries']}
        self.assertEqual(recipients,donors);self.assertEqual(len(donors),56)
        for e in a['entries']:
            self.assertNotEqual(e['recipient']['source_id'],e['donor']['source_id']);self.assertNotEqual(e['recipient']['source_group_id'],e['donor']['source_group_id'])
            self.assertEqual(e['recipient']['slot'],e['donor']['slot'])
    def test_same_group_distinct_source_still_forbidden(self):
        rows=[row(0,'a','shared'),row(1,'b','shared'),row(2,'c','other'),row(3,'d','other')]
        x=M.donor_map(rows);self.assertTrue(x['donor_ready'])
        self.assertTrue(all(e['recipient']['source_group_id']!=e['donor']['source_group_id'] for e in x['entries']))
    def test_impossible_capacity_preserved_students_blocked(self):
        rows=[row(0,'a'),row(1,'a'),row(2,'b')]
        x=M.donor_map(rows);self.assertFalse(x['donor_ready']);self.assertTrue(x['student_training_blocked'])
        self.assertEqual(len(x['impossible_strata']),2);self.assertEqual(x['valid_slots'],6);self.assertEqual(x['mapped_slots'],0)
    def test_raw_mask_not_merged_at_boundary(self):
        rows=[row(0,'a',t=1,mask=(1,1)),row(1,'b',t=1,mask=(1,1)),row(2,'a',t=1,mask=(0,1)),row(3,'b',t=1,mask=(0,1)),row(4,'a',t=0)]
        x=M.donor_map(rows);self.assertEqual(x['valid_slots'],4);self.assertEqual(len(x['strata']),2)
        self.assertEqual({tuple(s['stratum'][3]) for s in x['strata']},{(1,1),(0,1)})
        self.assertTrue(all(e['recipient']['slot']==1 for e in x['entries']))
    def test_phase_family_lag_preserved(self):
        rows=[]
        for profile in (*M.PROFILES,'identity','near_clean'):
            for t in (10,30,60,80):
                for sid in ('a','b'):rows.append(row(len(rows),sid,t=t,profile=profile))
        x=M.donor_map(rows);self.assertTrue(x['donor_ready'])
        for e in x['entries']:
            a,b=rows[e['recipient']['row_index']],rows[e['donor']['row_index']]
            self.assertEqual((a['profile_id'],a['severity_phase'],a['external_mask']),(b['profile_id'],b['severity_phase'],b['external_mask']))
    def test_tamper_invalid_padding_and_wrong_role_rejected(self):
        base=[row(0,'a'),row(1,'b')]
        for bad in [dict(base[0],source_id='tampered'),changed(base[0],split='dev'),changed(base[0],true_availability=[0,1]),changed(base[0],severity_phase='wrong')]:
            with self.assertRaises(ValueError):M.donor_map([bad,base[1]])
    def test_group_consistency_and_seed_locked(self):
        with self.assertRaises(ValueError):M.donor_map([row(0,'a','g1'),row(1,'a','g2')])
        with self.assertRaises(ValueError):M.donor_map([row(0,'a'),row(1,'b')],seed=1)
    def test_empty_support_has_no_fake_maps(self):
        x=M.donor_map([row(0,'a',t=0),row(1,'b',mask=(0,0))]);self.assertTrue(x['donor_ready']);self.assertEqual(x['valid_slots'],0);self.assertEqual(x['entries'],[])
    def test_no_overwrite_and_file_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.json';r=M.save_new(p,{'a':1});self.assertEqual(r['file_sha256'],M.file_sha(p))
            with self.assertRaises(FileExistsError):M.save_new(p,{'a':2})
    def test_accepted_audit_hash_gate(self):
        n={'rows':[{}]*16000};n['schedule_sha256']=M.value_sha(n)
        s={'schema':'openh3.history_utility_revised_schedule.v1','role':'teacher','seed':M.SEEDS['teacher'],'steps':1000,'batch_size':16,'numeric_schedule':n,'dataset_sha256':'d','cohort_sha256':M.COHORT_SHA,'legacy_research_ready':False,'training_admitted':False,'revised_cohort_ready':True};s['schedule_sha256']=M.value_sha(s)
        audit={'datasets':{'train':{'dataset_sha256':'d'}},'schedules':{'teacher':{'schedule_sha256':s['schedule_sha256'],'numeric_schedule_sha256':n['schedule_sha256']}}}
        self.assertEqual(M.validate_schedule(s,audit,'teacher')['rows'],16000)
        audit['schedules']['teacher']['schedule_sha256']='wrong'
        with self.assertRaises(ValueError):M.validate_schedule(s,audit,'teacher')

if __name__=='__main__':unittest.main()
