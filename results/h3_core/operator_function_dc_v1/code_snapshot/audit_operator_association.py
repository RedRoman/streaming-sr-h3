"""Declared finite-prior association before/after each actual LR cue.

No generation, optimization, held-query fitting, or likelihood temperature
selection. Uses the exact same two fixed far-only atoms and sigma.005.
"""
import argparse
import json
from pathlib import Path

import torch

from operator_function_data import COORDS, cameras_for
from observation_quotient import ObservationQuotient


def describe(log_density):
    joint = torch.softmax(log_density.flatten(), 0).reshape(2, 5)
    pu = joint.sum(1)
    pg = joint.sum(0)
    positive = joint > 0
    independent = pu[:, None]*pg[None]
    mi = float((joint[positive]*(joint[positive].log()-independent[positive].log())).sum())
    conditional = torch.softmax(log_density, 1)
    def entropy(p):
        positive = p > 0
        return float(-(p[positive]*p[positive].log()).sum())
    return dict(joint=joint.tolist(), carrier_marginal=pu.tolist(), operator_marginal=pg.tolist(),
                operator_given_carrier=conditional.tolist(),
                carrier_entropy_nats=entropy(pu), operator_entropy_nats=entropy(pg),
                joint_mutual_information_nats=mi,
                conditional_operator_entropy_nats=[entropy(p) for p in conditional],
                conditional_operator_mean=(conditional@torch.tensor(COORDS, dtype=torch.float64)).tolist(),
                conditional_operator_mode=conditional.argmax(1).tolist(),
                normalized_log_density=(log_density-torch.logsumexp(log_density.flatten(), 0)).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', nargs='+', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    q = ObservationQuotient(device='cpu')
    for name in args.roots:
        root = Path(name)
        manifest = json.loads((root/'manifest.json').read_text())
        rows = []
        for source in manifest['sources']:
            item = torch.load(root/source['id']/'family.pt', map_location='cpu', weights_only=False)
            a = torch.empty((2, 5), dtype=torch.float64)
            b = torch.empty_like(a)
            for u in range(2):
                for j, g in enumerate(COORDS):
                    camera = cameras_for(g, q)
                    for key, scores in [('A', a), ('B', b)]:
                        residual = camera[key].observe(item['basergb'][u:u+1])-item['observations'][key]
                        scores[u, j] = -.5*residual.double().square().sum()/.005**2
            early = describe(a)
            late = describe(a+b)
            old_early = item['joint_early_weights'].double()
            old_late = item['joint_late_weights'].double()
            rows.append(dict(source=source['id'], split=source['split'],
                             prior_far_only=describe(torch.zeros_like(a)), after_A=early, after_AB=late,
                             native_cached_joint_A_maxabs=float((torch.tensor(early['joint'])-old_early).abs().max()),
                             native_cached_joint_AB_maxabs=float((torch.tensor(late['joint'])-old_late).abs().max()),
                             conditional_g_change_AB_minus_A=float((torch.tensor(late['operator_given_carrier'])-torch.tensor(early['operator_given_carrier'])).abs().max()),
                             compression_timing='After both A+B packets have arrived; every support guide contains A+B. No claimed earlier-A compressed state in this pilot.',
                             packet_ids=item['packet_ids']))
        report = dict(status='complete', sources=rows, actual_D_calls=0, actual_E_calls=0,
                      optimizer_steps=0, new_DiT_calls=0,
                      sigma=.005, prior='Uniform over the declared 2 far-only atoms x5 operator coordinates.',
                      interpretation='Final concentration alone does not invalidate an association probe; inspect early joint dependence and when compression occurs. B camera does not depend on g, so B can change carrier weights but conditional q(g|u) is algebraically unchanged.')
        (root/'association_prefix_audit.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(root=name, sources=[dict(source=r['source'], A=r['after_A']['joint_mutual_information_nats'],
                                                       AB=r['after_AB']['joint_mutual_information_nats'],
                                                       A_modes=r['after_A']['conditional_operator_mode'],
                                                       A_Hg=r['after_A']['conditional_operator_entropy_nats'],
                                                       A_pu=r['after_A']['carrier_marginal'],
                                                       AB_pu=r['after_AB']['carrier_marginal']) for r in rows])), flush=True)


if __name__ == '__main__':
    main()
