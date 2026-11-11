import unittest

import torch

from rphi_model import RphiModel, uncertainty_nll
from train_rphi import fidelity_losses, state_digest


class TrainingContractTests(unittest.TestCase):
    def test_aligned_identity_guard(self):
        target = torch.full((2, 3, 8, 8), .5)
        _, exact = fidelity_losses(target, target, target)
        _, damaged = fidelity_losses(target + .1, target, target)
        self.assertEqual(float(exact['identity_guard']), 0.)
        self.assertGreater(float(damaged['identity_guard']), .099)
        with self.assertRaises(ValueError):
            fidelity_losses(target, target, target[..., :4, :4])

    def test_confidence_does_not_change_backbone_training(self):
        x = torch.rand(2, 3, 3, 16, 16)
        y = torch.rand(2, 3, 16, 16)
        models = []
        for variant in ('blind', 'blind_confidence'):
            torch.manual_seed(12)
            models.append(RphiModel(variant=variant, ch=8, blocks=1))
        for model in models:
            params = [p for name, p in model.named_parameters() if not name.startswith('confidence_head.')]
            optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss, _ = fidelity_losses(out['rgb'], y, x[:, -1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.)
                if out['log_variance'] is not None:
                    uncertainty_nll(out, y).backward()
                optimizer.step()
        self.assertEqual(state_digest(models[0].backbone.state_dict()),
                         state_digest(models[1].backbone.state_dict()))


if __name__ == '__main__':
    unittest.main()
