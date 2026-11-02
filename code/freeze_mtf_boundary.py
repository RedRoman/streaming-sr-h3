"""Where MTF conditioning works, where it fails, and why -- with the two readings it replaces.

The claim started as "an MTF is a better conditioning signal than a one-hot". Measurement has moved it
twice, and the second move produced a mechanism rather than another number.

READING 1, retired: "MTF beats one-hot". Not supported anywhere. On the five trained operators the deltas
are -0.010/-0.004/-0.015/-0.109/+0.008 dB at x2/x4/x8/deblur/denoise, i.e. mtf is behind or level
everywhere.

READING 2, retired as OVERSTATED: "MTF does not beat one-hot in vocabulary, proven". Two of its three
legs do not hold:
  - The end-to-end LPIPS leg is invalid. mtf-minus-onehot is +0.0008 LPIPS on 77 clips, but the
    mis-conditioning control measures that handing the arm a WRONG operator descriptor (x6 instead of x4)
    costs only +0.0009. A measurement with no resolving power for the conditioning cannot be used to
    declare two conditionings equivalent; the innerDC projection onto R_phi(y) absorbs the difference.
  - The data-size leg is confounded. Both arms read ~590 windows per operator. This repo's own DATA
    comment already records that data binds this family -- 8.7x moved the x4 estimate +0.457 dB -- and a
    17-dim vector is more data-hungry than a 6-dim one. A tie at 590 windows is not a tie in general.
  - What DOES hold is the sign analysis: x4/x8/denoise have mixed signs across 4 seeds (true noise),
    while deblur is 4/4 negative at -0.092 to -0.144, far outside the seed spread. Those are different
    findings and were wrongly pooled as "equivalent".

READING 3, current: MTF's value is EXPRESSIVENESS WITH A PREDICTABLE BOUNDARY.

  It works, and one-hot cannot run at all:
    x6   descriptor L2 0.2067 from the trained set, between x4 and x8   mtf +2.088 dB   onehot UNREPRESENTABLE
    x3   descriptor L2 0.2731, between x2 and x4                        mtf +7.586 dB   onehot UNREPRESENTABLE

  It fails, and worse than a wrong constant:
    deblur sigma 2.6-3.5   L2 0.03-0.21, nearest x8 / deblur:2.5        -0.038 dB vs onehot  (fine)
    deblur sigma 0.5-0.9   L2 0.14-1.05, between denoise and x2         -2.917 dB vs onehot  (collapse)

  The boundary is NOT distance. deblur:0.9 sits at L2 0.139, CLOSER than x6's 0.207, and still collapses.
  What separates them is whether the nearest trained operators share a behaviour. x6 is bracketed by x4
  and x8, which are the same operator at two strengths, so interpolating their behaviour is meaningful.
  A sigma-0.6 kernel is bracketed by denoise (D = identity, nothing to restore, MTF all ones) and x2
  (restore three quarters of the spectrum) -- two incompatible behaviours, and interpolating them is not.
  The collapse is also flat across the out-of-range bins (-2.74/-3.14/-2.87), not graded by distance,
  which is what "wrong behaviour selected" predicts and "far from training data" does not.

  The asymmetry has a cause in the descriptor's geometry: an MTF is bounded in [0,1], so shrinking sigma
  pushes it toward all-ones fast (L2 1.30 from sigma 1.0 to 0.5) while growing sigma saturates it toward
  all-zeros slowly (L2 0.24 from 2.5 to 3.5). Extrapolation is cheap in one direction and expensive in
  the other, and the physical parameter does not say which.

So the deployable statement is: an MTF buys operators a one-hot has no slot for, PROVIDED the descriptor
lands among trained operators that behave alike -- and that proviso is checkable before deployment, from
the operator alone, by taking the descriptor's distance to the trained set and reading off the nearest
neighbours' identities. Nothing about the image is needed, which keeps it in the non-blind setting.
"""
import json
import socket
from pathlib import Path

ROOT = Path('/cache/zihaohe/streaming_sr0')
OUT = ROOT / 'results/FROZEN_mtf_boundary.json'


def jl(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def main():
    o = {'date': '2026-11-02', 'host': socket.gethostname(), 'sections': {}}
    S = o['sections']

    S['in_vocabulary'] = {
        'measurement': 'R_phi val gain, 4 seeds per arm, identical data and capacity',
        'mtf_minus_onehot_dB': {'x2': -0.0103, 'x4': -0.0037, 'x8': -0.0151,
                                'deblur': -0.1091, 'denoise': +0.0082},
        'sign_across_4_seeds': {
            'x2': '4/4 negative but |delta| 0.000-0.030, at the edge of resolution',
            'x4': 'mixed (-0.032/+0.017/-0.006/+0.006) -> noise',
            'x8': 'mixed (-0.072/+0.054/-0.011/-0.032) -> noise',
            'deblur': '4/4 negative at -0.144/-0.099/-0.102/-0.092 -> a real, consistent deficit',
            'denoise': 'mixed -> noise',
        },
        'training_windows_per_operator': 590,
        'caveat': 'the big sets hold 10351-13672 per operator and were never used for this comparison; '
                  'a sweep is running because a 17-dim condition is more data-hungry than a 6-dim one',
    }

    S['end_to_end_lpips_cannot_decide_this'] = {
        'mtf_minus_onehot_lpips': 0.0008,
        'paired_clips': 77,
        'sem': 0.0004,
        'noise_floor': 0.0009,
        'mis_conditioning_control': {
            'setup': 'x4 input, x6 descriptor, same 53 clips, paired against the x4 descriptor',
            'wrong_minus_right_lpips': 0.0009,
            'sem': 0.0006,
            'reading': 'handing the arm the WRONG operator costs the same as the entire mtf-vs-onehot '
                       'difference, so this metric has no resolving power for the conditioning signal. '
                       'The innerDC projection onto R_phi(y) absorbs it. Conditioning must be judged at '
                       'the R_phi level, not after the DiT.',
        },
    }

    S['deblur_conditioning_was_never_implemented'] = {
        'bug': "mtf_cond caches on the operator STRING; every deblur sample passed the bare 'deblur' and "
               'operator_desc fell back to its default sigma=1.75, so one constant vector covered a '
               'kernel range of sigma 1.0-2.5',
        'discarded_signal': '||MTF(sigma=1.0) - MTF(sigma=2.5)|| = 0.8851',
        'cannot_be_fixed_at_inference': {
            'feeding_true_descriptors_to_the_old_weights': -0.4094,
            'worst_bin_furthest_from_1.75': -1.10,
            'reading': 'the network bound that constant as the identity of "deblur"; a retrain was '
                       'required to test the idea at all',
        },
        'after_retraining_with_per_sample_kernels': {
            'flag': '--mtf-persample (opt-in, so every earlier number stays reproducible)',
            'seeds': 4,
            'persample_minus_bare_dB': +0.0335,
            'seed_sd': 0.0394,
            'persample_minus_onehot_dB': -0.0756,
            'on_137_val_items': {'persample_minus_bare': +0.0536, 'persample_minus_onehot': -0.0406},
            'per_sigma_bin': 'one-hot leads in every bin; its lead shrinks as sigma grows '
                             '(0.079 -> 0.011)',
            'verdict': 'the fix is real and directionally right, but within the seed spread and still '
                       'behind one-hot. The deblur column is a negative result, not an implementation '
                       'note -- which is only knowable because the implementation was fixed first.',
        },
    }

    S['the_boundary'] = {
        'principle': 'MTF generalises to an operator it never trained on when the descriptor lands among '
                     'trained operators that share a behaviour; it collapses when the bracketing '
                     'operators behave incompatibly, and distance alone does not predict which',
        'works': {
            'x6': {'descriptor_L2': 0.2067, 'bracketed_by': ['x4', 'x8'],
                   'behaviours': 'one operator at two strengths',
                   'mtf_gain_dB': 2.088, 'mtf_persample_gain_dB': 1.997, 'onehot': 'UNREPRESENTABLE'},
            'x3': {'descriptor_L2': 0.2731, 'bracketed_by': ['x2', 'x4'],
                   'behaviours': 'one operator at two strengths',
                   'mtf_gain_dB': 7.586, 'mtf_persample_gain_dB': 7.446, 'onehot': 'UNREPRESENTABLE'},
            'deblur_sigma_2.6_3.5': {'descriptor_L2': [0.03, 0.21],
                                     'nearest': ['x8', 'deblur:2.5'],
                                     'vs_onehot_dB': -0.0380, 'reading': 'behaves like in-range'},
        },
        'fails': {
            'deblur_sigma_0.5_0.9': {
                'descriptor_L2': [0.14, 1.05],
                'bracketed_by': ['denoise (MTF all ones, nothing to restore)',
                                 'x2 (restore three quarters of the spectrum)'],
                'vs_onehot_dB': -2.9170,
                'vs_a_wrong_constant_descriptor_dB': -2.4725,
                'per_bin': [-2.736, -3.139, -2.872],
                'reading': 'flat across bins rather than graded by distance, which is what '
                           '"the condition selects an incompatible behaviour" predicts',
            },
        },
        'why_distance_is_not_the_criterion': 'deblur:0.9 sits at L2 0.139, closer than x6 at 0.207, and '
                                            'still collapses',
        'why_the_direction_is_asymmetric': 'an MTF is bounded in [0,1]: sigma 1.0->0.5 moves it 1.30 in '
                                           'L2 toward all-ones, while sigma 2.5->3.5 moves it only 0.24 '
                                           'toward all-zeros. The physical parameter does not reveal '
                                           'which extrapolation is cheap.',
        'checkable_before_deployment': 'compute the descriptor from the operator alone, take its distance '
                                       'to the trained set and read the nearest neighbours; no image is '
                                       'involved, so this stays inside the non-blind setting',
    }

    S['what_this_means_for_the_claim'] = {
        'do_not_write': 'MTF is a better conditioning signal (no measurement supports it)',
        'do_not_write': 'MTF handles arbitrary unseen operators (sigma 0.5-0.9 refutes it)',
        'do_write': 'a one-hot cannot represent an operator outside its vocabulary at all -- x6 and x3 '
                    'raise SystemExit rather than scoring badly -- while an MTF runs and delivers '
                    '+2.09 dB and +7.59 dB there, PROVIDED the descriptor is bracketed by trained '
                    'operators of the same behavioural family. The proviso is part of the contribution, '
                    'not a caveat on it, because it is decidable from the operator alone.',
    }

    S['sources'] = {k: ('present' if jl(v) else 'MISSING') for k, v in {
        'omni_mtf_k3': 'results/omni/omni_mtf_k3.json',
        'omni_mtf_ps_s0': 'results/omni/omni_mtf_ps_s0.json',
        'deblur_or_lo': 'results/omni/deblur_or_lo.json',
        'deblur_or_hi': 'results/omni/deblur_or_hi.json',
        'deblur_or_ctl': 'results/omni/deblur_or_ctl.json',
        'deblur_strata_ps': 'results/omni/deblur_strata_ps.json',
        'omni_holdout_hull': 'results/omni/omni_holdout_hull.json',
        'judge_cond_full': 'results/deploy/judge_cond_full.json',
    }.items()}

    S['limits'] = [
        'the boundary principle is supported by four operator families (x6, x3, two deblur ranges), not '
        'proven; it predicts the sign of every case measured so far but has not been tested on an '
        'operator bracketed by same-behaviour trainers at LARGE descriptor distance',
        'all conditioning comparisons are at 590 training windows per operator; the 3000-window sweep is '
        'running and the 10351-window point is not done',
        'the x6 and x3 numbers are R_phi-level PSNR; the pixel-level x6 table is being judged separately '
        'on data/real_eval/heldout_s6',
        'x3 has only 22 training windows available in its pair set, so its val n is 6 -- the +7.586 dB is '
        'indicative, not a measurement at the same weight as x6',
    ]

    OUT.write_text(json.dumps(o, indent=1))
    print(json.dumps({'wrote': str(OUT), 'sections': list(S)}, indent=1))


if __name__ == '__main__':
    main()
