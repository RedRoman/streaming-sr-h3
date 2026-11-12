import copy,unittest
import torch
import audit_checkpoint as A

class SavedState(unittest.TestCase):
    def fixture(self):
        base={'p':torch.zeros(2,3,dtype=torch.float32)}
        model={'p':torch.ones(2,3,dtype=torch.float32)}
        spec={'eps':1e-8,'weight_decay':0.,'amsgrad':False,'foreach':False,'fused':False,'betas':[.9,.999]}
        s={'model_state':model,'model_sha256':A.state_sha(model),'parameter_names':['p'],
            'optimizer_state':{'state':{0:{'step':torch.tensor(2.),'exp_avg':torch.ones(2,3),'exp_avg_sq':torch.ones(2,3)}},'param_groups':[dict(spec,params=[0],lr=A.lr(2))]}}
        return s,base,spec
    def test_finite_changed_saved_state(self):
        s,b,c=self.fixture();r=A.check_tensor_state(s,b,2,c);self.assertEqual(r['changed_model_tensors'],1);self.assertEqual(r['optimizer_tensors'],3)
    def test_nonfinite_moment_rejected(self):
        s,b,c=self.fixture();s['optimizer_state']['state'][0]['exp_avg'][0,0]=float('nan')
        with self.assertRaises(ValueError):A.check_tensor_state(s,b,2,c)
    def test_wrong_step_shape_rejected(self):
        s,b,c=self.fixture();s['optimizer_state']['state'][0]['step']=torch.tensor(1.)
        with self.assertRaises(ValueError):A.check_tensor_state(s,b,2,c)
        s,b,c=self.fixture();s['optimizer_state']['state'][0]['exp_avg']=torch.ones(2,2)
        with self.assertRaises(ValueError):A.check_tensor_state(s,b,2,c)
    def test_wrong_model_digest_or_unchanged_rejected(self):
        s,b,c=self.fixture();s['model_state']['p'][0,0]+=1
        with self.assertRaises(ValueError):A.check_tensor_state(s,b,2,c)
        s,b,c=self.fixture();s['model_state']=copy.deepcopy(b);s['model_sha256']=A.state_sha(b)
        with self.assertRaises(ValueError):A.check_tensor_state(s,b,2,c)
if __name__=='__main__':unittest.main()
