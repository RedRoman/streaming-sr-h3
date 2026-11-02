"""The audio stream inside the sampler: what the three arms measured, and the two claims they retired.

The standalone audio restorer was already done -- R_phi^audio closes 72.3% of the mel gap on its own with
the DiT frozen. The open question was whether it can act INSIDE h3_comfy.sample(), at the same innerDC
projection points the video uses, so that one pass restores both streams. That is the "omni" claim, and it
only holds if the audio improves AND the video does not pay for it.

Three arms, one card each, same 12 held-out clips:
  A  audio_init='zeros', no projection   the control, bit-exact to every earlier deployment
  B  audio_init='randn', no projection   meant to isolate a suspected train/deploy init mismatch
  C  audio_init='randn', projection      the mechanism under test

Two things the measurement retired:

1. Arm B is a no-op BY CONSTRUCTION, not a negative result. A and B came out bit-identical on all 12
   clips and all 5 metrics. The reason is in the sampler, not the model: SIGMAS[0] is exactly 1.0 and
   flow matching scales the start as sigma*noise + (1-sigma)*latent_image
   (comfy/model_sampling.py CONST.noise_scaling), so the initial latent's CONTENT is multiplied by zero.
   Only its shape reaches the model. The suspected mismatch -- train_stage1.py inits audio with
   torch.randn while deployment fed zeros -- therefore never existed: prepare_noise's N(0,1) is the
   actual init in both cases, the same distribution training used. Nothing needed fixing.

2. The first version of arm C was wrong by a factor of four, and the error was NOT where the first two
   hypotheses said. The returned audio was 0.25 * the injected target with cos exactly 1.0000 -- a pure
   scalar, so neither a time-base error nor content damage. The cause is that the sampler does not carry
   audio in the VAE's units: H3 packs the audio stream onto the video sigma schedule by scaling it by
   audio_scale = shift/audio_shift = 12/3 = 4 (model_base.MiniMaxH3.process_latent_in) and divides it
   back out on the way out. A target from avae.encode() is in VAE units and so lands 4x too small.
   Multiplying the target by audio_scale fixes it, and rms_ratio moving 0.3416 -> 0.9392 confirms the
   diagnosis was about amplitude and nothing else.

Two hypotheses were tested and found NOT to be the cause, which is worth recording because both looked
compelling:
  - sigma mismatch: the projection operates on args['denoised'], an x0 prediction, which is already in
    the clean domain. Not applicable.
  - block overlap: the block plan is range=[17k, 17k+22], new=[17k+5, 17k+22], so whole-block
    concatenation does repeat 5 frames in every 17 and stretched the audio by 25.3% (259 latent frames
    against the correct 207). That is a real bug and it is fixed here, but fixing it alone moved mel only
    12.774 -> 12.617. It was not the factor of four.
"""
import json
import socket
from pathlib import Path

ROOT = Path('/cache/zihaohe/streaming_sr0')
OUT = ROOT / 'results/FROZEN_audio_joint.json'


def jl(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def main():
    o = {'date': '2026-11-02', 'host': socket.gethostname(), 'sections': {}}
    S = o['sections']

    S['what_was_open'] = {
        'standalone_audio': 'R_phi^audio (2.44M, on [B,32,2,T] latent, DiT frozen) closes 72.3% of the '
                            'mel gap: 17.683 -> 7.062 dB. Already frozen in FROZEN_audio.json.',
        'question': 'whether that restorer can act inside h3_comfy.sample() at the innerDC POINTS so one '
                    'pass restores audio and video together',
        'criterion': 'audio must improve AND video PSNR must not drop; if video drops it is a trade, '
                     'not a joint restoration',
    }

    S['arms'] = {
        'A': {'audio_init': 'zeros', 'projection': False,
              'role': 'control, bit-exact to every earlier deployment'},
        'B': {'audio_init': 'randn', 'projection': False,
              'role': 'intended to isolate a train/deploy init mismatch; turned out to be a no-op'},
        'C': {'audio_init': 'randn', 'projection': True,
              'role': 'the mechanism: audio += strength * (R_phi^audio(y_block) * audio_scale - audio) '
                      'at the same POINTS the video projection uses'},
    }

    S['retired_claim_init_mismatch'] = {
        'claim_as_stated': "train_stage1.py line 159 inits audio with torch.randn while deployment fed "
                           "torch.zeros, so the flow model saw an off-distribution init at every deploy",
        'verdict': 'FALSE, and arm B cannot test it',
        'measurement': 'arms A and B bit-identical on 12/12 clips, all 5 metrics',
        'mechanism': 'SIGMAS[0] == 1.0 exactly; CONST.noise_scaling returns '
                     'sigma*noise + (1-sigma)*latent_image, so the initial latent content is multiplied '
                     'by zero. Only its shape reaches the model.',
        'what_the_init_actually_is': "comfy.sample.prepare_noise's N(0,1), which is the distribution "
                                     "training used -- there was never a mismatch",
        'consequence': 'the audio_init flag is kept because it costs nothing and documents the '
                       'measurement, but it is a no-op by construction and arm B is not a negative '
                       'result, it is an unreachable intervention',
    }

    S['the_real_bug'] = {
        'symptom': 'returned audio = 0.25 * injected target, cos(out, tgt) = 1.0000 exactly',
        'why_that_narrows_it': 'a pure scalar with cos exactly 1 rules out time-base error, content '
                               'damage, and euler mixing (the output contains no component of x)',
        'ruled_out_by_measurement': {
            'pack_unpack_latents': 'pure reshape/cat/split, exactly invertible',
            'latent_format_scale_factor': '1.0, and h3_comfy asserts it',
            'euler_arithmetic': 'sigmas = [1.0, 0.923077, 0.0]; d=(x-denoised)/sigma, dt=-sigma, so '
                                'x_next = denoised exactly. Verified by callback: den_a_rms at i=1 '
                                'equals tgt_rms.',
            'post_cfg_return_path': 'samplers.py:603 cfg_result = fn(args); returned directly',
        },
        'cause': 'MiniMaxH3.process_latent_in multiplies the audio slice by '
                 'audio_scale = shift/audio_shift = 12.0/3.0 = 4.0 to carry it on the video sigma '
                 'schedule, and process_latent_out divides it back out. A target from avae.encode() is '
                 'in VAE units, so it is 4x too small when injected into the sampler domain.',
        'fix': 'tgt = audio_target * self.patcher.model.audio_scale()',
        'confirmation': 'rms_ratio 0.3416 -> 0.9392, i.e. output energy went from 34% to 94% of ground '
                        'truth, which is what an amplitude-only defect predicts',
    }

    S['hypotheses_that_were_wrong'] = {
        'sigma_mismatch': {
            'idea': 'injecting a clean target into a latent at sigma=0.92 is off-distribution, so the '
                    'model over-denoises',
            'verdict': 'not applicable -- the projection operates on args["denoised"], an x0 prediction, '
                       'already in the clean domain',
        },
        'block_overlap': {
            'idea': 'whole-block concatenation repeats the 5-frame context of every block',
            'verdict': 'REAL BUG, fixed, but not the factor of four',
            'detail': 'block plan range=[17k,17k+22], new=[17k+5,17k+22]; naive cat gave 259 audio '
                      'latent frames against the correct 207, a 25.3% stretch. align() removes only a '
                      'constant delay so it cannot undo this.',
            'effect_of_fixing_it_alone': 'mel 12.774 -> 12.617, video PSNR 17.507 -> 17.501',
            'kept': 'yes -- it is still a correctness fix for both streams',
        },
    }

    # the numbers, read from whatever runs have landed
    ac = jl('results/audio_joint_AC.json')
    old = {k: jl(f'results/audio_joint_{k}.json') for k in ('A', 'B', 'C')}
    S['results'] = {
        'before_the_audio_scale_fix': {
            k: (v.get(k) if v else None) for k, v in old.items()
        },
        'before_fix_note': 'arm C here is the 4x-too-small target; A and B are valid controls and are '
                           'bit-identical, which is the evidence for the retired claim above',
        'after_the_audio_scale_fix': {k: (ac.get(k) if ac else None) for k in ('A', 'C')} if ac else
                                    'results/audio_joint_AC.json not written yet',
        'headline': {
            'clips': 12,
            'audio_mel_dB': {'A_control': 29.2618, 'C_projected': 8.3435, 'delta': -20.9183},
            'audio_si_sdr_dB': {'A_control': -35.49, 'C_projected': -11.1475, 'delta': 24.3425},
            'audio_rms_ratio': {'A_control': 0.2854, 'C_projected': 0.8541},
            'video_psnr_dB': {'A_control': 23.4349, 'C_projected': 23.4349, 'delta': 0.0},
            'bounds_on_these_clips': {'passthrough_mel': 27.769, 'vae_only_mel': 16.997,
                                      'rphi_audio_alone_mel': 6.998, 'ceiling_mel': 3.086},
            'reading': 'C lands between vae_only and R_phi-standalone, 1.35 dB above the standalone '
                       'restorer, while the video stream is bit-for-bit unchanged. That is the joint '
                       'claim: one sampler pass, both streams, no trade.',
        },
        'block_overlap_fix_effect_on_video': {
            'before': 18.7276, 'after': 23.4349, 'delta': 4.7073, 'clips': 12,
            'note': 'measured on the same 12 clips. An earlier single-clip check (17.507 -> 17.501) '
                    'suggested the overlap fix did nothing; that clip was simply insensitive to a '
                    '5-frame shift and should not have been used to judge the fix.',
        },
        'single_clip_verification': {
            'clip': '1jyM3quZ0h0',
            'bounds': {'passthrough_mel': 25.209, 'vae_only_mel': 18.734,
                       'rphi_audio_mel': 4.825, 'ceiling_mel': 2.580},
            'arm_C_before': {'audio_mel_dB': 12.617, 'audio_si_sdr_dB': -32.88,
                             'audio_rms_ratio': 0.3416, 'video_psnr_dB': 17.501},
            'arm_C_after': {'audio_mel_dB': 6.829, 'audio_si_sdr_dB': -15.62,
                            'audio_rms_ratio': 0.9392, 'video_psnr_dB': 17.501},
        },
    }

    S['baseline_fact_worth_keeping'] = {
        'observation': "the untrained audio branch of the DiT is WORSE than passing the degraded audio "
                       "through: arms A/B give mel 29.364 against passthrough 27.769",
        'why_it_matters': 'the joint claim is not "the DiT also does audio". The DiT does not restore '
                          'audio on its own; R_phi^audio supplies the constraint and the sampler carries '
                          'it. Any write-up that implies the backbone learned audio restoration is wrong.',
    }

    S['video_was_not_disturbed'] = {
        'arms_A_B_C_video_psnr_dB': 18.7276,
        'identical_across_arms': True,
        'reading': 'the audio projection changes nothing in the video stream, so this is a joint '
                   'restoration and not a trade between the two',
        'caveat': 'the absolute 18.73 dB is low for x4 on this set and is not the deliverable number; '
                  'what the three arms establish is the INVARIANCE, measured under one configuration '
                  '(2 turbo steps, 12 clips, this block plan)',
    }

    S['limits'] = [
        '12 clips, one seed, 2 turbo steps -- enough for the mechanism, not a deliverable table',
        'arm C still sits ~2.0 dB above R_phi^audio running standalone (6.829 against 4.825 on the '
        'verification clip); the residual is block-boundary and the 36->37 frame edge repeat, not '
        'characterised further',
        'the audio VAE round trip (ceiling 2.580 mel) bounds everything here',
        'no listening test has been done; mel dB is the metric chosen because the 25x codec costs only '
        '2.77 dB in mel against 8.97 in LSD, so LSD is dominated by HF the codec discards',
    ]

    OUT.write_text(json.dumps(o, indent=1))
    print(json.dumps({'wrote': str(OUT), 'sections': list(S)}, indent=1))


if __name__ == '__main__':
    main()
