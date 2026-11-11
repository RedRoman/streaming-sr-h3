"""Few focused semantic gates: budget, restoration, and saved pixel loss."""
import copy
import unittest

import numpy as np
import audit_terminal as audit


def budgets():
    return {"actual":{"reference_NFE":3,"candidate_NFE":3,"reference_VAE_decode_calls":1,
            "candidate_VAE_decode_calls":1,"stage_decode_calls":{"reference_decode":1,"candidate_decode":1},
            "RGB_loss_expressions":1,"backward_attempts":1,"completed_backward_calls":1,
            "optimizer_steps":0,"adapter_updates":0,"forbidden_calls":[]},"budget_passed":True,"block_budget_passed":True,
            "block_forward_calls":{"DiT":{p:{str(i):3 for i in range(50)} for p in ("reference","candidate","backward")},
                                   "VAE":{p:{str(i):1 for i in range(36)} for p in ("reference_decode","candidate_decode","backward")}}}


def math_fixture():
    report=budgets()
    state={"in_training":True,"jit_compile_false":True,"matmul_allow_hf32":False,"conv_allow_hf32":False,
           "deterministic_algorithms":False,"deterministic_warn_only":False,
           "rope":{"id":1,"module":"comfy_kitchen.backends.eager.rope","qualname":"rms_rope_split_half"},
           "scale_shift":{"id":2},"gate":{"id":3}}
    before={**state,"in_training":False}
    native={**state,"rope":{"id":4,"module":"comfy_kitchen.backends.ascend.rope","qualname":"rms_rope_split_half"}}
    report.update(math_before=before,math_after=copy.deepcopy(before),math_restored=True,DiT_math_active=state,
        DiT_math_after_decode=copy.deepcopy(state),wrapper_attributes_restored=True,session_inactive=True,stale_backward_guard_rejected=True,
        patcher_restoration={k:True for k in ("session_shapes_restored","session_current_patcher_restored","whole_run_current_patcher_restored","whole_run_shapes_restored")})
    report["patcher_restoration"]["session_state_initialized"]=True
    report["math_observations"]=[{"group":g,"phase":p,"block":int(i),"state":copy.deepcopy(native if g=="VAE" and p!="reference_decode" else state)}
        for g,phases in report["block_forward_calls"].items() for p,blocks in phases.items() for i,n in blocks.items() for _ in range(n)]
    report["decoder_context_restorations"]=[{"phase":phase,"before":state,"after":copy.deepcopy(state),"restored":True}
        for phase,n in (("candidate_decode",37),("backward",36)) for _ in range(n)]
    return report


class SemanticTests(unittest.TestCase):
    def test_no_budget_promotion_and_wrong_recompute_count_rejected(self):
        report=budgets();self.assertEqual(audit.budget(report,True)["explicit_NFE"],6)
        report["block_forward_calls"]["VAE"]["backward"]["35"]=0
        with self.assertRaisesRegex(ValueError,"block counts"):audit.budget(report,True)
        report=budgets();report["actual"]["completed_backward_calls"]=0
        with self.assertRaisesRegex(ValueError,"budget"):audit.budget(report,True)
        self.assertEqual(audit.budget(report,False)["actual"]["completed_backward_calls"],0)

    def test_context_and_patcher_restoration_evidence_required(self):
        report=math_fixture();self.assertEqual(audit.contexts(report)["decoder_context_restorations"],73)
        report["decoder_context_restorations"][-1]["after"]={"leaked":True}
        with self.assertRaisesRegex(ValueError,"restored"):audit.contexts(report)
        report=math_fixture();report["patcher_restoration"]["whole_run_current_patcher_restored"]=False
        with self.assertRaisesRegex(ValueError,"patcher"):audit.contexts(report)

    def test_independent_pixel_loss_cannot_accept_wrong_saved_scalar(self):
        # The actual backward reduction is Torch32; a separate float64 metric
        # is exactly recomputed from saved pixels and must not be fabricated.
        target=np.linspace(.2,.8,128,dtype=np.float32)[None,None,:,None]
        rgb=np.full((5,128,128,3),.5,np.float32)
        actual=audit.recompute_rgb_mse(rgb,target,audit.array_sha(target))
        audit.verify_reduction(actual,actual,"saved RGB MSE")
        with self.assertRaisesRegex(ValueError,"reduction"):
            audit.verify_reduction(actual+.001,actual,"saved RGB MSE")


if __name__=="__main__":unittest.main()
