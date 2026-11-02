"""The half of 2026-11-01 that only this host can see, frozen from its local judge files.

The day's evidence is split across two hosts that cannot exchange files: port 22 is closed between them
and the shared /temp mount refuses writes with "Disk quota exceeded". What makes a split file safe is
that the held-out set was fingerprinted on both and matches byte for byte
(manifest+meta 0d1c920b..., pixels 887f4506...), so LPIPS from either host is on the same 96 clips.

freeze_today.py on the other host covers vid4, the cross-operator deployment, the latency/attention
measurement and the audio bounds. This covers the conditioning comparison, the x6 holdout and the
sharpness negative, whose source files live here.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/cache/zihaohe/streaming_sr0')
OUT = ROOT / 'results/FROZEN_20261101_b.json'
NF = {'lpips_2step': 0.0009, 'lpips_4step': 0.0018}
OPS = ['2', '4', '8', 'deblur', 'denoise']


def jl(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def main():
    o = {'date': '2026-11-01', 'part': 'b', 'host': __import__('socket').gethostname(),
         'noise_floors': NF, 'sections': {}}
    S = o['sections']

    # 1 + 2: conditioning signal, with the seed spread that decides whether its deltas are readable
    seeds = {}
    for t in ('omni_mtf_k3', 'omni_mtf_k3_s1', 'omni_mtf_k3_s2', 'omni_mtf_k3_s3',
              'omni_onehot_k3', 'omni_onehot_k3_s1', 'omni_onehot_k3_s2', 'omni_onehot_k3_s3'):
        j = jl(f'results/omni/{t}.json')
        if j and 'final' in j:
            seeds[t] = {'cond': j.get('cond'), 'seed': j.get('seed'), 'k': j.get('k'),
                        'params': j.get('params'),
                        'gain_dB': {str(s): round(v['gain_dB'], 4) for s, v in j['final']['val'].items()},
                        'vs_single_dB': {str(s): (round(v['vs_single'], 4)
                                                  if v.get('vs_single') is not None else None)
                                         for s, v in j['final']['val'].items()}}
    agg, deltas = {}, {}
    for cond in ('mtf', 'onehot'):
        g = [r['gain_dB'] for r in seeds.values() if r['cond'] == cond]
        if g:
            agg[cond] = {op: {'mean': round(float(np.mean([x[op] for x in g])), 4),
                              'sd': round(float(np.std([x[op] for x in g], ddof=1)), 4),
                              'n_seeds': len(g)} for op in OPS if op in g[0]}
    if 'mtf' in agg and 'onehot' in agg:
        for op in agg['mtf']:
            d = agg['mtf'][op]['mean'] - agg['onehot'][op]['mean']
            sd = max(agg['mtf'][op]['sd'], agg['onehot'][op]['sd'])
            deltas[op] = {'mtf_minus_onehot_dB': round(d, 4), 'max_seed_sd_dB': round(sd, 4),
                          'sigma': round(abs(d) / sd, 2) if sd > 0 else None,
                          'verdict': 'inside_noise' if abs(d) <= 2 * sd else 'real'}
    S['conditioning_signal'] = {
        'question': 'does MEASURING the operator (radial MTF of U(D(.)), 16 bands + severity) beat '
                    'NAMING it (one-hot over 5 operators + severity)?',
        'matched': 'same five operators, same K=3 causal stack, same data, same schedule, same seeds; '
                   'only the FiLM input differs. 2.045M params (mtf) vs 2.020M (onehot).',
        'per_seed': seeds, 'aggregate_gain_dB': agg, 'mtf_minus_onehot': deltas,
        'finding': ('mtf is not a better conditioner. On x2/x4/x8/denoise the delta is inside the seed '
                    'spread; on deblur mtf is behind by 0.109 dB with no overlap across four seeds. The '
                    'deblur MTF (0.90/0.57/0.24/0.06) overlaps the scale operators heavily while a '
                    'one-hot is orthogonal by construction, so a measured operator is least '
                    'discriminative exactly where two operators have similar spectra.'),
        'claim_that_survives': ('expressiveness, not accuracy: a one-hot cannot be evaluated at an '
                                'operator outside its vocabulary at all, an MTF can. Measured price '
                                'inside the vocabulary is about 0.1 dB.'),
        'K3_result': ('K=3 omni beats the K=1 single-operator specialists by +0.36/+0.31/+0.86 dB at '
                      'x2/x4/x8, so the causal stack added to OmniRange today is net positive and the '
                      'earlier caveat in cmp_m4.sh -- "K=1, because OmniRange has no causal stack" -- '
                      'no longer applies.')}

    ho, mis = jl('results/omni/omni_holdout_s0.json'), jl('results/omni/omni_holdout_miscond.json')
    S['operator_holdout_x6'] = {
        'setup': 'trained on {2,4,8,deblur,denoise}; x6 pairs built by the same generator, never trained '
                 'on by any arm. 72 scored frames.',
        'mtf': {'base_dB': 29.907, 'psnr_dB': 31.701, 'gain_dB': 1.794},
        'onehot': 'UNREPRESENTABLE -- raises by name, OPS has no slot for 6',
        'trained_operator_reference_gain_dB': {'x2': 2.633, 'x4': 2.831, 'x8': 2.781},
        'fraction_of_trained_gain': round(1.794 / 2.831, 3),
        'probe_structure': ('x6 sits strictly between x4 and x8 in every radial band: band0 '
                            '1.00/0.98/0.88, band1 0.83/0.44/0.17, band2 0.32/0.04/0.00'),
        'mis_conditioning_control': {
            'x6_data_fed': {'x6_correct': 1.794, 'x4': 1.690, 'x8': 1.685, 'x2': 1.471,
                            'denoise': 1.397},
            'cost_of_wrong_descriptor_dB': {'x4': -0.104, 'x8': -0.109, 'x2': -0.323,
                                            'denoise': -0.398},
            'reading': ('the ordering is monotone in operator distance -- nearest neighbours cost least, '
                        'the identity costs most -- so the descriptor is read as a position in operator '
                        'space, not ignored. But the CORRECT descriptor leads its neighbour by only '
                        '0.104 dB, the same size as the deblur deficit, so most of the +1.794 dB is '
                        'broad robustness to nearby operators. "Interpolation" is not supported at this '
                        'magnitude.')},
        'files': {'holdout_json': bool(ho), 'miscond_json': bool(mis)}}

    # 5: the sharpness negative, with the mechanism measured rather than argued
    def pooled(name):
        j = jl(f'results/{name}')
        if not j:
            return None
        return {a: {b: {'energy_ratio': round(v['energy_ratio'], 4), 'cos': round(v['cos'], 4)}
                    for b, v in r.items()} for a, r in j['arms'].items()}

    jr = jl('results/deploy/judge_r7.json')
    delivered = None
    if jr:
        rows = [r for r in jr['per_clip'] if r.get('degradation') == 'realistic']
        delivered = {}
        for a in jr['arms']:
            v = [r['arms'][a]['lpips'] for r in rows if r['arms'].get(a)]
            p = [r['arms'][a]['psnr'] for r in rows if r['arms'].get(a) and r['arms'][a].get('psnr')]
            if v:
                delivered[a] = {'n': len(v), 'lpips': round(float(np.mean(v)), 4),
                                'psnr': round(float(np.mean(p)), 2) if p else None}
    tr = {}
    for t in ('r7_energy', 'r7_deficit'):
        j = jl(f'results/range_branch/{t}.json')
        if j and 'final' in j:
            v = j['final']['val']['realistic']
            tr[t] = {'iter': j['final']['iter'], 'gain_dB': round(v['gain_dB'], 3),
                     'mid_ratio': round(v['mid_ratio'], 4), 'mid_ratio_y': round(v['mid_ratio_y'], 4)}
    d = (round(delivered['r7energy']['lpips'] - delivered['plain2']['lpips'], 4)
         if delivered and 'r7energy' in delivered and 'plain2' in delivered else None)
    S['sharpness_negative'] = {
        'objective_tested': ('hf_mode=energy with hf_w=2 -- match the local RMS of the sigma=2 high pass '
                             'rather than the high pass itself, so the model is asked for the right '
                             'AMOUNT of detail where it cannot know the phase'),
        'estimate_layer_final': tr,
        'observation_itself': ('mid_ratio_y = 1.0463: the degraded observation already carries 4.6% MORE '
                               'mid-band energy than the truth, because compression adds high-frequency '
                               'junk. r7_energy lands at 1.0143, closer to the truth than the input is, '
                               'so this is not a naive overshoot.'),
        'pre_registered_gate': 'mid_ratio > 0.93 and estimate PSNR not down more than 0.3 dB',
        'gate_result': 'PASSED on both arms',
        'delivered_lpips': delivered or 'missing_here',
        'delta_vs_shipped': d,
        'verdict': ('worse' if d is not None and d > NF['lpips_2step'] else None),
        'per_clip_better': '27/77',
        'mechanism_measured': pooled('band_energy_r7.json') or 'missing_here',
        'mechanism_reading': ('band_energy settles it. r7_energy moves mid-band energy 0.8100 -> 0.9519 '
                             'and null-band energy 0.5441 -> 0.5979, exactly as designed, and the cosine '
                             'against the truth FALLS in both bands, 0.8754 -> 0.8622 and 0.4876 -> '
                             '0.4789. Energy up, correlation down: the added detail is the right amount '
                             'in the wrong phase. A moment-matching objective is phase-blind by '
                             'construction, so this family of losses cannot improve delivered sharpness, '
                             'and guide mid_ratio is not a valid proxy for it. The quota gate then '
                             'amplifies the error, because it scales generated detail by the guide local '
                             'std.'),
        'reframing_of_the_complaint': ('FlashVSR+ has the HIGHEST null-band energy (0.6437) and the '
                                       'LOWEST null-band cosine (0.4207); our shipped arm has the '
                                       'highest cosine in all three bands. The perceived softness is the '
                                       'price of phase correctness under this objective family, which is '
                                       'also why we lead on LPIPS, 0.1929 against 0.2072.'),
        'what_would_be_needed': ('an objective that constrains phase rather than a moment -- perceptual, '
                                 'adversarial or flow-consistent -- or more conditioning information '
                                 '(K>3, cross-block memory), not another moment term.'),
        'confound_stated': ('the deployed checkpoint was iteration 16000 (mid_ratio 1.0293), not the '
                            'final 24000 (1.0143); the gap is small against a 0.0112 LPIPS deficit and '
                            'is not expected to flip the sign, but it is untested. r7_deficit was never '
                            'deployed: its run OOMed when the host reservation daemon reclaimed the '
                            'cards.')}

    OUT.write_text(json.dumps(o, indent=1))
    print(json.dumps({'wrote': str(OUT), 'sections': sorted(S),
                      'seeds_found': len(seeds),
                      'deblur_delta': deltas.get('deblur'),
                      'delivered_delta_lpips': d}, indent=1))


if __name__ == '__main__':
    main()
