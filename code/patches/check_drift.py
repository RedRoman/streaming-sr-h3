"""Report which of this project's changes are present on THIS host, as a checklist.

Written because the same class of failure happened three times in one day and each time I patched
whichever host was in front of me:

  17:03  the x6 deploy died with "RangeBranch.__init__() got an unexpected keyword argument 'cond'"
         because run_deploy.py's is_omni detection existed only on 30778. It burned a ~3 hour card window.
  20:28  write_audio_lr.sh died with "No module named 'degrade_audio'" because that file existed only on
         30778.
         freeze_today.py likewise, so the day's frozen file covers only the half of the results that
         happened to live on the host it was written on.

The root cause is structural and not going away: there is no host-to-host route on port 22, and the
shared /temp mount refuses writes with "Disk quota exceeded" even for a 19 KB file. Code therefore moves
by re-running a deterministic patch script per host, or by gzip+base64 through the operator. So the fix
is not "remember to copy" -- it is a checklist that names what is missing before a card is spent.

Each entry is a marker whose ABSENCE has a known cost, not a version number.
"""
import json
import sys
from pathlib import Path

CODE = Path('/cache/zihaohe/streaming_sr0/code')

# (file, marker, what breaks without it)
CHECKS = [
    ('run_deploy.py', 'is_omni',
     'an mtf/K>1 omni checkpoint is built as a RangeBranch and dies on cfg key "cond" (killed x6 at 17:03)'),
    ('run_deploy.py', '_claim_device_early',
     'the card is idle for the 40 GB DiT load and a reservation holder can take it mid-load'),
    ('omni_stage1.py', 'COND_DIM',
     'no mtf conditioning at all; --cond mtf and --holdout-scales are unavailable'),
    ('omni_stage1.py', 'claim_card_early',
     'the card is idle while 31 GB of npz loads'),
    ('train_range2.py', 'claim_card_early',
     'same; an r7 run was killed at iteration 2000 by a holder taking the card during the load'),
    ('operator_desc.py', 'def operator_mtf',
     'the radial MTF descriptor is missing, so omni_stage1 cannot import it'),
    ('omni_holdout.py', 'mis_cond',
     'the x6 holdout and its mis-conditioning control cannot be run'),
    ('degrade_audio.py', 'def degrade_audio',
     'no audio degradation operator; audio pairs and audio_lr cannot be produced'),
    ('degrade_audio.py', 'def mel_dist',
     'the mel metric is missing and LSD would be used, which reads a 25x perceptual codec as capped'),
    ('audio_line1.py', 'def align',
     'waveform metrics are computed without the constant -266 sample codec delay, a ~30 dB artefact'),
    ('prepare_audio_pairs.py', 'def read_audio',
     'audio latent pairs cannot be generated'),
    ('train_audio_range.py', 'lsd_regressed_clips',
     'the audio verdict ignores LSD and reports mel-only wins as unqualified'),
    ('prepare_range_data.py', 'RANGE_SRC_PREFIX',
     'the manifests absolute paths do not resolve on a host where /temp is grafted differently'),
    ('band_energy.py', "'cos'",
     'band energy is reported without the cosine, which is the term that explained the sharpness failure'),
    ('bench_deploy_latency.py', 'attn_share',
     'attention share of the DiT cannot be measured, so a sparse-attention port is decided on extrapolation'),
    ('freeze_today.py', 'eval_set_fingerprint', 'the day-1 results cannot be frozen on this host'),
    ('freeze_today_b.py', 'conditioning_signal', 'the conditioning/holdout/sharpness half cannot be frozen'),
    ('freeze_audio.py', 'corr_headroom_vs_lsd_change', 'the audio line cannot be frozen'),
]


def main():
    import socket
    rows = []
    for f, marker, cost in CHECKS:
        p = CODE / f
        if not p.exists():
            rows.append({'file': f, 'marker': marker, 'state': 'FILE ABSENT', 'cost': cost})
            continue
        rows.append({'file': f, 'marker': marker,
                     'state': 'ok' if marker in p.read_text() else 'MARKER ABSENT',
                     'cost': cost})
    bad = [r for r in rows if r['state'] != 'ok']
    print('host %s  --  %d checks, %d missing' % (socket.gethostname(), len(rows), len(bad)))
    print()
    w = max(len(r['file']) for r in rows)
    for r in rows:
        print('%-4s %-*s %s' % ('' if r['state'] == 'ok' else '>>>', w, r['file'],
                                r['marker'] if r['state'] == 'ok' else
                                '%s: %s' % (r['state'], r['marker'])))
    if bad:
        print()
        print('MISSING HERE, and what it costs:')
        for r in bad:
            print('  %s / %s' % (r['file'], r['marker']))
            print('      %s' % r['cost'])
    (CODE.parent / 'results/code_drift.json').write_text(json.dumps(
        {'host': socket.gethostname(), 'checks': len(rows), 'missing': len(bad), 'rows': rows}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
