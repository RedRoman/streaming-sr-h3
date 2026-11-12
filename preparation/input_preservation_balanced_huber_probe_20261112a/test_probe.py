import copy
import unittest
import numpy as np
import probe as B


class BalancedHuberTests(unittest.TestCase):
    def test_only_weights_change_target_and_zero_missing_sign_rejected(self):
        z=np.column_stack((np.arange(4),np.ones(4)));y=np.array([.75,.75,.75,-.25])
        x,t,w,c=B.balanced_problem(z,y)
        self.assertIs(x,z);self.assertIs(t,y)
        np.testing.assert_allclose(w,[1/6,1/6,1/6,.5]);self.assertEqual(c['positive_tiles'],3)
        with self.assertRaisesRegex(ValueError,'zero'):B.balanced_problem(z,np.array([0.,1,1,-1]))
        with self.assertRaisesRegex(ValueError,'both'):B.balanced_problem(z,np.ones(4))

    def test_weighted_constant_preserves_magnitudes_and_flat_nearest_zero(self):
        v=B.weighted_constant(np.array([.75,.75,.75,-.25]));self.assertAlmostEqual(v['selected_nearest_zero'],.25)
        flat=B.weighted_constant(np.array([4.,4.,4.,-4.]));np.testing.assert_allclose(flat['minimizer_interval'],[-3,3],atol=1e-12)
        self.assertEqual(flat['selected_nearest_zero'],0.)
        unequal=B.weighted_constant(np.array([1.,1.,.5,-.25]))
        self.assertNotEqual(unequal['weighted_training_Huber_loss'],unequal['unweighted_training_Huber_loss'])

    def test_no_information_prior_shift_is_not_classification_success(self):
        z=np.ones((4,1));y=np.array([1.,1.,1.,-1.]);count=[]
        w,proof=B.fit_balanced_huber(z,y,lambda:count.append(1))
        self.assertEqual(len(count),200);self.assertLess(abs(w[0]),1e-14)
        metrics=B.classify(y,z@w);self.assertEqual(metrics['balanced_accuracy'],.5);self.assertEqual(metrics['AUROC'],.5)
        self.assertEqual(metrics['positive_K1_recall'],metrics['negative_false_K1']);self.assertEqual(proof['iterations'],200)

    def test_informative_signal_fixed_solver_and_Hessian_bound(self):
        y=np.array([.5,.5,.5,-.5]*16);z=np.column_stack((2*y,np.ones(len(y))))
        w,r=B.fit_balanced_huber(z,y);m=B.classify(y,z@w)
        self.assertEqual(m['balanced_accuracy'],1.);self.assertLess(r['final_weighted_objective'],r['initial_weighted_objective'])
        _,_,weights,_=B.balanced_problem(z,y);curvature=z.T@(z*weights[:,None])+np.diag([.001,0])
        self.assertEqual(r['Lipschitz_upper_bound'],float(np.linalg.eigvalsh(curvature)[-1]))

    def test_AUC_exact_ties_single_class_and_positive_offset(self):
        y=np.array([-1.,1.,-1.,1.]);s=np.array([1.,2.,1.,3.])
        self.assertEqual(B.auc(y,s),1.);self.assertEqual(B.classify(y,s)['balanced_accuracy'],.5)
        self.assertEqual(B.auc(y,-s),0.);self.assertEqual(B.auc(y,np.zeros(4)),.5)
        self.assertIsNone(B.auc(-np.ones(4),s));self.assertIsNone(B.classify(-np.ones(4),s)['balanced_accuracy'])
        self.assertEqual(B.classify(y,np.full(4,-0.))['K1_tile_fraction'],0.)

    def test_accepted_old_scores_and_standardization_require_exact_replay(self):
        P=B.legacy();rng=np.random.default_rng(31);x=rng.normal(size=(180,11));y=np.where(x[:,0]>0,1.,-1.)
        split={'row_folds':[0,1,2,3,4]};folds=np.repeat(np.arange(5),36);old={'result':{'folds':[]}}
        pred=np.empty(180);const=np.empty(180)
        for f in range(5):
            tr=folds!=f;te=~tr;st=P.standardizer(x[tr]);w=np.arange(12)/20
            pred[te]=P.transform(x[te],st)@w;const[te]=.25
            old['result']['folds'].append({'fold':f,'training_tiles':144,'held_out_tiles':36,'training_mean':st[0].tolist(),
              'training_scale':st[1].tolist(),'zero_variance_channel_indices':np.flatnonzero(st[2]).tolist(),
              'probe_fit':{'coefficients':w.tolist()},'constant':{'selected_nearest_zero':.25}})
        old['result']['held_out_score_sha256']={'linear':P.array_sha(pred),'constant':P.array_sha(const)}
        _,_,r=B.replay_old(P,x,y,split,old);self.assertEqual(P.array_sha(r['old_Huber_linear']),P.array_sha(pred))
        wrong=copy.deepcopy(old);wrong['result']['folds'][0]['probe_fit']['coefficients'][0]+=.01
        with self.assertRaisesRegex(ValueError,'score exact'):B.replay_old(P,x,y,split,wrong)

    def test_source_macro_and_single_class_denominator(self):
        rows=[{'source_id':'a','profile_id':'identity'},{'source_id':'b','profile_id':'native'}]
        y=np.r_[-np.ones(36),np.tile([-1.,1.],18)];base=np.ones(72)
        scores={'old_Huber_linear':base,'old_Huber_constant':base,'balanced_Huber':y,'weighted_Huber_constant':np.zeros(72)}
        r=B.summaries(y,scores,rows)
        self.assertEqual(r['source_macro_primary']['balanced_Huber']['AUROC']['valid_sources'],1)
        self.assertEqual(r['profile_source_macro']['identity']['balanced_Huber']['AUROC']['valid_sources'],0)
        self.assertEqual(r['source_macro_primary']['balanced_Huber']['negative_false_K1'],{'mean':0.,'valid_sources':2})
        self.assertFalse(r['automatic_admission'])


if __name__=='__main__':unittest.main()
