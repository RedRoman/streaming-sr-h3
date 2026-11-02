"""Freeze every result that is currently believed, with the protocol that produced it.

Written because today a subset table was nearly reported as a full-set table: prepare_real_eval had each
of eight shards rebuild manifest.json from disk when it finished, so the last writer could be an early
finisher and the file listed 81 of 96 complete clips. The deploy read that manifest, sampled 81, and the
judge produced a table that looked exactly like the full one. Nothing in the numbers said "subset".

So every row here carries the surface it was measured on (clip set, n, sampling seed, judge file) and
every delta carries its size in units of the measured sampling-seed noise floor. A delta under about two
noise floors is not a result, whatever its bootstrap CI says -- that was established by sampling the same
arm three times, where the paired test called a +0.0035 seed difference "significant".

Run with --check to re-derive every number from the judge json on disk and fail if anything moved.
"""
import argparse
import json
from pathlib import Path

ROOT = Path('/cache/zihaohe/streaming_sr0')
R = ROOT / 'results/deploy'
OUT = ROOT / 'results/FROZEN.json'

# the one number that decides whether any other number means anything
# Two floors, one per step count. The shipped configuration is 2 steps, so NOISE2 is the one that
# applies to anything measured now; NOISE4 is kept because every arm up to 10-30 was measured at 4 steps
# and its deltas were judged against it. Halving the steps halved the floor, which is what fewer
# stochastic draws should do -- so the "2 steps beats 4" claim strengthened from 3.1 to 6.3 floors when
# the right denominator was used.
NOISE4 = {'source': 'judge_seeds.json, same arm (r6_xl_k3, dc=rphi, 4-step, merged Stage-1), seeds '
                    '101/202/303 over the full 96 held-out clips',
          'steps': 4, 'lpips_sd': 0.0018, 'lpips_spread': 0.0035,
          'psnr_sd': 0.0580, 'null_cos_sd': 0.0019, 'terr_null_sd': 0.0316}
NOISE2 = {'source': 'judge_noise2.json, same arm (r5_k3, dc=rphi, 2-step, merged Stage-1, no adapter), '
                    'seeds 101/202/303 over the full 96 held-out clips',
          'steps': 2, 'lpips_sd': 0.0009, 'lpips_spread': 0.0016,
          'psnr_sd': 0.0563, 'null_cos_sd': 0.0010, 'terr_null_sd': 0.0305,
          'lpips_three_seed_mean': 0.1939}
NOISE = {'shipped_floor': NOISE2, 'legacy_4step_floor': NOISE4,
         'lpips_sd': 0.0009, 'lpips_spread': 0.0016,
         'rule': ('a delta below 2 x lpips_sd is not interpretable; use the floor that matches the step '
                  'count the arms were sampled at, and say which one was used')}

MAIN = {'judge': 'judge_xl.json', 'group': 'realistic',
        'arms': ['bicubic', 'infvsr', 'flashvsr_plus', 'dck3', 'dcxl'],
        'note': ('dck3 = R_phi r5_k3 (4.7M, K=3); dcxl = r6_xl_k3 (13.3M, K=3). The width step between '
                 'them is +0.0020 LPIPS, 1.1 noise floors, so it is NOT established -- the shipped '
                 'configuration should be the 4.7M one. The headline LPIPS is the three-seed mean '
                 '0.1980 +- 0.0018, not the seed-101 draw of 0.1964.')}

# The shipped configuration changed on 10-31: two sampling steps, not four. Two steps is not a
# latency/quality trade -- it is better on LPIPS by 0.0056 (3.1 noise floors, 53/77 clips), better on
# lpips_gtrange and null_cos, and 38.7% faster on the deployed path. It loses 0.064 dB PSNR and 0.032
# terr_null, both small.
#
# The third-party step adapter is a separate and opposite finding: pdmd_2step at its own 2 steps costs
# +0.0518 LPIPS and at 4 steps +0.0933, with the damage in lpips_gtrange (0.1368 -> 0.2043 / 0.2482).
# plain2 beats it by 0.0573 (31.9 noise floors), so the step count was never the problem.
STEPS = {'judge': 'judge_plain2.json', 'group': 'realistic',
         'arms': ['flashvsr_plus', 'dck3', 'plain2', 'step2'],
         'shipped': ('R_phi r5_k3 (4.7M, K=3), 2 steps via turbo_sigmas(2, 12.0), innerDC onto '
                     'R_phi(y), no quota gate, no third-party adapter'),
         'deployed_s_per_frame_2step': 0.781, 'deployed_x_realtime_2step': 18.7,
         'two_step_floor_measured': ('sd 0.0009 over seeds 101/202/303 at 2 steps, half the 4-step '
                                     'floor. The -0.0056 is 6.3 of these, not 3.1 of the 4-step ones.'),
         'lpips_three_seed_mean': 0.1939}

ABL = {'judge': 'judge_abl.json', 'group': 'realistic',
       'arms': ['dcy', 'dcy_q', 'dcr', 'dcr_q'],
       'note': ('2x2 over innerDC target x quota gate. The gate is a net regression on held-out data '
                '(LPIPS +0.0085 = 4.7 noise floors, terr_null worse on 67 of 71 clips) and was dropped; '
                'its gain was tuned on the 24-clip const set and reversed on the 96-clip held-out set.')}

# CROSS_SCALE_CAVEAT: the three scale eval sets share the HR frames byte-for-byte and draw the
# degradation from the same seed, but they do NOT end up with the same degradation. degrade._noise calls
# rng.standard_normal(x.shape) (or rng.poisson over the array), which consumes a number of random values
# proportional to the LR pixel count, so the generator state diverges by scale and every later draw with
# it. Measured on clip 1jyM3quZ0h0: identical blur sigma 1.77 and identical interp, then x4 drew
# jpeg_q=61, x2 drew no compression, x8 drew h264_crf=26.
#
# This does not touch the omni result: within a scale both arms read byte-identical LR, so the paired
# delta is clean. It does mean the ABSOLUTE numbers are not comparable across scales -- bicubic on that
# clip is 15.685 / 15.256 / 13.919 dB at x2 / x4 / x8, monotone as it should be, but the gaps mix the
# scale factor with a different compression draw. Any cross-scale statement has to be about the deltas.
CROSS_SCALE_CAVEAT = True

OMNI = {'judges': {2: 'judge_omni_s2.json', 4: 'judge_omni_s4.json', 8: 'judge_omni_s8.json'},
        'group': 'realistic', 'arms': ['bicubic', 'single', 'omni'],
        'cross_scale_caveat': ('_noise consumes rng proportional to LR pixel count, so the three sets diverge in their compression draw after the noise stage; within-scale paired deltas are unaffected, absolute cross-scale numbers are not comparable'),
        'note': ('single = r5_big, omni = omni_m4; matched on width (4.73 vs 4.76M), windows (10351), '
                 'iters (24000), crop (112) and K (1 -- OmniRange has no causal stack). The x2 and x8 '
                 'held-out sets use the same 96 clips as x4 with byte-identical HR frames.')}

NEG = [
    ('Open-H3 2K refiner LoRA', 'judge_ref.json', 'dcref', 'dcxl',
     'lpips +0.0443 (25 noise floors), 6/77 clips better, and lpips_gtrange +0.0496 -- it degrades the '
     'generated detail itself. A strength sweep at 0.25/0.50/1.00 degrades monotonically from zero, so '
     'the scale assumption was right and the direction is wrong for degraded camera footage.'),
    ('pdmd 2-step adapter', 'judge_step2.json', 'step2', 'dck3',
     'lpips +0.0518 at its own 2 steps and +0.0933 at 4 steps, with lpips_gtrange 0.1368 -> 0.2043 / '
     '0.2482. Together with the Open-H3 refiner (+0.0443 lpips, +0.0496 gtrange) this is two independent '
     'third-party H3 adapters failing in the same band: both were distilled on H3\'s own generation '
     'distribution, not on a restoration pipeline with a guide latent and an innerDC projection. '
     'Reducing the step count WITHOUT an adapter is fine and in fact better -- see the steps section.'),
    ('post-hoc range substitution on other methods', 'judge_xbackbone.json', 'fvp_rs', 'flashvsr_plus',
     'lpips +0.0132 on FlashVSR+ and +0.0031 on InfVSR: the substitution buys PSNR and temporal error, '
     'not perceptual quality. lpips_gtrange moves +0.0004, which is the correctness check -- that metric '
     'replaces the range with truth, so substituting the range should not move it.'),
]

VID4 = {'files': {'r6_xl_k3 on BDx4': 'vid4_BDx4_r6_xl_k3.json',
                  'sp_bd on BDx4': 'vid4_BDx4_sp_bd.json',
                  'r6_xl_k3 on BIx4': 'vid4_BIx4.json'},
        'operator_fit': 'vid4_bd_fit.json',
        'note': ('Vid4 is out of the training degradation family both ways. BI is y = D(x) exactly, so '
                 'the identity is optimal and our estimator loses 12.84 dB. BD is blur-then-subsample '
                 'with no noise and no compression: +0.251 dB mean with 2 of 4 clips negative, i.e. '
                 'noise. A specialist trained on a GUESSED bd chain scored +15.174 dB on its own data '
                 'and -12.708 dB on Vid4, uniformly across clips, because the guessed chain sits 30.56 '
                 'dB from D(x) while Vid4 BD sits 38.17 dB away -- a severity mismatch. The operator was '
                 'then identified from Vid4 GT/BD pairs directly: INTER_AREA subsample, ksize 21, sigma '
                 '1.10 reproduces the released frames at 53.392 dB. Retraining on the fitted chain is '
                 'not done, and the pre-registered check is that the new pairs base must land near 38 dB.')}


def row(j, group, arm, keys=('lpips', 'psnr', 'lpips_gtrange', 'null_cos', 'p8.terr_null')):
    s = j['summary'][group]
    if arm not in s:
        return None
    return {k: round(s[arm][k]['mean'], 5) for k in keys if k in s[arm]}


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def delta(j, group, a, b, nf=None):
    nf = NOISE['lpips_sd'] if nf is None else nf
    pr = f'{a},{b}'
    if pr not in j.get('paired', {}):
        return None
    v = j['paired'][pr][group]['lpips']
    return {'lpips_mean': round(v['mean'], 5), 'ci95': [round(x, 5) for x in v['ci95']],
            'a_lower_on': f"{v['a_better']}/{v['n']}",
            'noise_floors': round(abs(v['mean']) / nf, 2),
            'interpretable': bool(abs(v['mean']) >= 2 * nf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='compare against the existing FROZEN.json')
    a = ap.parse_args()
    out = {'noise_floor': NOISE, 'main': None, 'steps': None, 'ablation': None, 'omni': {}, 'negative': [],
           'public_benchmark': VID4, 'latency': None}

    j = load(MAIN['judge'])
    if j:
        out['main'] = {'judge': MAIN['judge'], 'group': MAIN['group'], 'n': j['summary'][MAIN['group']]['n'],
                       'rows': {k: row(j, MAIN['group'], k) for k in MAIN['arms']},
                       'vs_flashvsr_plus': delta(j, MAIN['group'], 'dcxl', 'flashvsr_plus'),
                       'vs_infvsr': delta(j, MAIN['group'], 'dcxl', 'infvsr'),
                       'width_step_dcxl_vs_dck3': delta(j, MAIN['group'], 'dcxl', 'dck3'),
                       'note': MAIN['note']}
    j = load(STEPS['judge'])
    if j:
        out['steps'] = {'judge': STEPS['judge'], 'n': j['summary'][STEPS['group']]['n'],
                        'rows': {k: row(j, STEPS['group'], k) for k in STEPS['arms']},
                        'plain2_vs_dck3': delta(j, STEPS['group'], 'plain2', 'dck3'),
                        'plain2_vs_flashvsr_plus': delta(j, STEPS['group'], 'plain2', 'flashvsr_plus'),
                        'plain2_vs_step2': delta(j, STEPS['group'], 'plain2', 'step2'),
                        'shipped': STEPS['shipped'],
                        'deployed_s_per_frame': STEPS['deployed_s_per_frame_2step'],
                        'deployed_x_realtime': STEPS['deployed_x_realtime_2step'],
                        'two_step_floor_measured': STEPS['two_step_floor_measured'],
                        'lpips_three_seed_mean': STEPS['lpips_three_seed_mean']}
    j = load(ABL['judge'])
    if j:
        out['ablation'] = {'judge': ABL['judge'], 'group': ABL['group'], 'n': j['summary'][ABL['group']]['n'],
                           'rows': {k: row(j, ABL['group'], k) for k in ABL['arms']},
                           'gate_effect_at_dcr': delta(j, ABL['group'], 'dcr', 'dcr_q'),
                           'projection_target_effect': delta(j, ABL['group'], 'dcy', 'dcr'),
                           'note': ABL['note']}
    for op, f in OMNI['judges'].items():
        j = load(f)
        if j:
            out['omni'][f'x{op}'] = {'judge': f, 'n': j['summary'][OMNI['group']]['n'],
                                     'rows': {k: row(j, OMNI['group'], k) for k in OMNI['arms']},
                                     'omni_minus_single': delta(j, OMNI['group'], 'omni', 'single')}
    out['omni']['note'] = OMNI['note']
    for label, f, aa, bb, why in NEG:
        j = load(f)
        out['negative'].append({'what': label, 'judge': f,
                                'delta': delta(j, 'realistic', aa, bb) if j else None, 'why': why})
    lat = ROOT / 'logs/g4_e2e.log'
    if lat.exists():
        out['latency'] = {'source': 'logs/g4_fixed.log + logs/g4_e2e.log, warm, R_phi on the accelerator',
                          # Two protocols, never to be swapped silently. The harness figure is what
                          # every quality run was measured with; the deployed figure removes PNG
                          # encoding, which a stream does not do. The earlier 1.821 s/frame included
                          # PNG writes AND the sha256 second pass over the stitched stream.
                          'harness_s_per_frame': 1.4639, 'harness_x_realtime': 35.1,
                          'deployed_s_per_frame': 1.2156, 'deployed_x_realtime': 29.2,
                          'png_share_of_harness': 0.170, 'dit_share_of_deployed': 0.810,
                          'superseded_s_per_frame_1821': ('included PNG writes and the sha256 pass; '
                                                          'do not quote'),
                          'whole_clip_profile_s': {'sample': 126.54, 'png_write': 30.66, 'guide': 20.29,
                                                   'decode': 7.35, 'to_u8': 3.04, 'stream_write': 24.71},
                          'levers': {'drop_png': 0.26, 'steps_4_to_2': 0.40, 'pipelining': 0.18,
                                     'guide_reuse_dropped': 0.025, 'light_vae_dropped': 0.03},
                          'block_s': 32.103, 'dit_per_step_s': 3.446, 'vae_decode_s': 4.194,
                          'vae_encode_guide_s': 2.858, 'rphi_s_per_clip': 1.063,
                          'rphi_share_of_e2e': 0.0047,
                          'note': ('the block timer starts inside the block loop, so every latency figure '
                                   'reported before this one excluded R_phi; 1.821 is the end-to-end '
                                   'number. About 11 s of the 32.1 s block is still unaccounted for.')}

    if a.check and OUT.exists():
        prev = json.loads(OUT.read_text())
        diffs = [k for k in out if json.dumps(out[k], sort_keys=True) != json.dumps(prev.get(k), sort_keys=True)]
        print(json.dumps({'changed_sections': diffs}, indent=1))
        raise SystemExit(1 if diffs else 0)

    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print()
    print('wrote', OUT)


if __name__ == '__main__':
    main()
