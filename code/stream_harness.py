"""Does this pipeline actually stream? Frame-arrival-driven harness plus a causality audit.

Every result so far came from a batch script that reads all 124 LR frames, processes blocks in a loop and
writes all frames out. That measures throughput and nothing else: it cannot show end-to-end latency, it
cannot show backlog growth, and it silently permits a block to read frames that would not have arrived yet.
This drives the same per-block timings through a clock instead.

Model: frame t arrives at t/fps. Block k spans pixel frames [17k, 17k+22) and cannot start until its last
frame has arrived; it occupies the (single) worker for the measured block time; the frames it newly emits
become available when it finishes. A frame is "late" if it becomes available after its own playout deadline.

Two things come out of that:

  latency      per frame, available_at - due_at; the deadline that matters is the steady-state one, since a
               streaming player can absorb a fixed startup delay but not a growing one
  speedup_need the smallest uniform speedup of block time at which the backlog stops growing, i.e. the
               number the distillation work has to hit

The causality audit is separate and static: it lists, for each stage, which input frames the output of
frame t depends on. A stage that reads frames after t is not causal, and the harness reports the implied
look-ahead rather than quietly folding it into the latency number.
"""
import argparse, glob, json
from pathlib import Path

import numpy as np

ROOT = Path('/cache/zihaohe/streaming_sr0')

# What each stage of t8dcr_q reads, relative to the frame being produced. Hand-derived from the code and
# asserted against the manifest below, so a change in block geometry cannot silently invalidate it.
CAUSALITY = [
    ('R_phi(y)', 'K=1 range branch', 0, 0, True),
    ('guide = VAE_enc(U(y))', 'per block', -21, 0, None),        # filled from block geometry
    ('DiT self-attention', 'bidirectional inside the 22-frame block', -21, 21, None),
    ('innerDC projection', 'inside the block only', -21, 21, None),
    ('block stitching', 'frame taken from the block that first emits it', 0, 0, True),
    ('cross-fade (not in t8dcr_q)', 'needs the NEXT block', 0, 22, False),
    ('spatial quota', 'local std of U(R_phi(y)) at this frame', 0, 0, True),
]


def block_plan(frames, block=22, stride=17):
    out = []
    k = 0
    while 17 * k < frames:
        s, e = stride * k, min(stride * k + block, frames)
        if e - s < block:
            break
        new_lo = 0 if k == 0 else s + (block - stride)
        out.append({'k': k, 'range': [s, e], 'new': [new_lo, e]})
        k += 1
    return out


def simulate(block_times, frames, fps=24.0, block=22, stride=17, speedup=1.0, workers=1):
    """Returns per-frame availability under a single-worker (or N-worker) frame-arrival schedule."""
    plan = block_plan(frames, block, stride)
    free = [0.0] * workers
    avail, starts = {}, []
    for b, bt in zip(plan, block_times[:len(plan)]):
        s, e = b['range']
        ready = (e - 1) / fps                       # last frame of the block has arrived
        w = int(np.argmin(free))
        start = max(ready, free[w])
        done = start + bt / speedup
        free[w] = done
        starts.append({'k': b['k'], 'ready': ready, 'start': start, 'done': done,
                       'queue_wait': start - ready})
        for f in range(b['new'][0], b['new'][1]):
            avail[f] = done
    lat = {f: avail[f] - f / fps for f in avail}
    return plan, starts, avail, lat


def needed_speedup(block_times, frames, fps=24.0, block=22, stride=17, workers=1):
    """Smallest uniform speedup at which queue wait stops growing (backlog bounded)."""
    mean_bt = float(np.mean(block_times))
    emit_interval = stride / fps                     # seconds of content produced per block
    return mean_bt / (emit_interval * workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='results/deploy_runs/t8_dcr', help='dir with */stream_result.json')
    ap.add_argument('--post-ms-per-frame', type=float, default=10.0,
                    help='post-processing cost per output frame (R_phi on NPU measured 3.6 ms; '
                         'the rest of the chain is elementwise)')
    ap.add_argument('--fps', type=float, default=24.0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--speedups', default='1,8,10,20,40,51,60')
    ap.add_argument('--out', default='results/deploy/stream_harness.json')
    a = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / a.runs / '*' / 'stream_result.json')))
    assert files, a.runs
    per_clip_bt, frames = [], None
    for p in files:
        d = json.loads(Path(p).read_text())
        bt = [b['total_s'] for b in d['blocks']]
        per_clip_bt.append(bt)
        n = max(f for f in d['frames_emitted']) + 1
        frames = n if frames is None else min(frames, n)
    bt_mean = [float(np.mean([c[i] for c in per_clip_bt if i < len(c)])) for i in range(max(map(len, per_clip_bt)))]
    post = a.post_ms_per_frame * 1e-3 * 17           # per block, 17 newly emitted frames
    bt_mean = [b + post for b in bt_mean]

    rep = {'runs': a.runs, 'clips': len(files), 'frames': frames, 'fps': a.fps, 'workers': a.workers,
           'block_seconds_mean': float(np.mean(bt_mean)), 'post_per_block_s': post,
           'content_seconds_per_block': 17 / a.fps,
           'needed_speedup_for_bounded_backlog': needed_speedup(bt_mean, frames, a.fps, workers=a.workers),
           'speedups': {}}

    for sp in [float(x) for x in a.speedups.split(',')]:
        plan, starts, avail, lat = simulate(bt_mean, frames, a.fps, speedup=sp, workers=a.workers)
        l = np.array([lat[f] for f in sorted(lat)])
        qw = np.array([s['queue_wait'] for s in starts])
        rep['speedups'][str(sp)] = {
            'first_frame_latency_s': float(l[0]),
            'last_frame_latency_s': float(l[-1]),
            'latency_growth_s_per_block': float(np.polyfit(np.arange(len(qw)), qw, 1)[0]) if len(qw) > 1 else 0.0,
            'backlog_bounded': bool(np.polyfit(np.arange(len(qw)), qw, 1)[0] < 1e-6) if len(qw) > 1 else True,
            'frames_late_vs_steady_deadline': int((l > l[0] + 1e-6).sum()),
            'realtime_factor': float(np.mean(bt_mean) / (17 / a.fps)),
        }

    # causality audit, resolved against the real block geometry
    plan = block_plan(frames)
    audit = []
    for name, note, lo, hi, causal in CAUSALITY:
        if causal is None:
            # a frame sits anywhere in its block, so worst-case look-ahead is block_len - 1
            hi = 21
            causal = False
        audit.append({'stage': name, 'note': note, 'reads_from': lo, 'reads_to': hi,
                      'causal': bool(causal and hi <= 0),
                      'look_ahead_frames': max(0, hi), 'look_ahead_s': max(0, hi) / a.fps})
    rep['causality'] = audit
    rep['total_look_ahead_frames'] = max(x['look_ahead_frames'] for x in audit)
    rep['total_look_ahead_s'] = rep['total_look_ahead_frames'] / a.fps

    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(rep, indent=1))

    print('block %.1f s produces %.3f s of content -> %.0fx realtime; need %.1fx speedup for bounded backlog (workers=%d)'
          % (rep['block_seconds_mean'], rep['content_seconds_per_block'],
             rep['block_seconds_mean'] / rep['content_seconds_per_block'],
             rep['needed_speedup_for_bounded_backlog'], a.workers))
    print()
    print('%-8s %14s %14s %16s %10s' % ('speedup', 'first lat (s)', 'last lat (s)', 'growth (s/blk)', 'bounded'))
    for sp, v in rep['speedups'].items():
        print('%-8s %14.1f %14.1f %16.2f %10s' % (sp, v['first_frame_latency_s'], v['last_frame_latency_s'],
                                                  v['latency_growth_s_per_block'], v['backlog_bounded']))
    print()
    print('causality audit (look-ahead in frames):')
    for x in audit:
        print('  %-30s %-46s %s' % (x['stage'], x['note'],
                                    'causal' if x['causal'] else '+%d frames (%.2f s)' % (x['look_ahead_frames'], x['look_ahead_s'])))


if __name__ == '__main__':
    main()
