"""Two small actual Torch updates, no real checkpoint/data/backend load."""
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_history_utility as T
import check_history_utility_backend as B
from test_train_history_utility import ToyTeacher, batch


class BackendTests(unittest.TestCase):
    def test_two_steps_strict_reload_and_research_resume_rejected(self):
        torch.set_num_threads(1); torch.manual_seed(29)
        model = ToyTeacher(); opt = T.make_optimizer(model)
        binding, runtime = {'schedule': 'fixed'}, {'device': 'cpu'}
        with tempfile.TemporaryDirectory() as td:
            report = {}; out = Path(td)
            B.run_core(model, opt, [batch(), batch()], binding, runtime, torch.device('cpu'), out, report)
            self.assertEqual(report['actual'], {'teacher_forwards': 2, 'teacher_backwards': 2,
                'teacher_optimizer_updates': 2, 'save': 1, 'clear_reload': 1})
            self.assertTrue(report['strict_model_optimizer_RNG_reload'])
            self.assertFalse(report['checkpoint_eligible_as_research_warmstart'])
            self.assertEqual([x['step'] for x in report['step_metrics']], [1, 2])
            with self.assertRaisesRegex(ValueError, 'identity'):
                T.restore_checkpoint(out/'backend_smoke_step2.pt', model, opt, binding, runtime, torch.device('cpu'))

    def test_backend_will_not_extend_batch_budget(self):
        model = ToyTeacher(); opt = T.make_optimizer(model)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'exactly two'):
                B.run_core(model, opt, [batch()]*3, {}, {}, torch.device('cpu'), Path(td), {})
        self.assertEqual(len(opt.state), 0)


if __name__ == '__main__': unittest.main()
