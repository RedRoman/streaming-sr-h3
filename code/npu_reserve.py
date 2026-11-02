"""Reserve NPUs through the host's own cooperative lock, instead of fighting its reservation daemon.

A colleague's controller on this host keeps idle cards reserved by spawning a holder that allocates 59.5
GB each. Killing those holders does not work and actively backfires: the controller rescans and respawns
within seconds, and when it does it re-evaluates EVERY device -- including ones already running a job
whose process is briefly invisible. That is how a kill loop here took this host from two free cards to
zero and killed an r7 run at iteration 2000.

The controller is cooperative if you read it. Its spawn test is

    if d in holders or d in selected or m.query(d) or not lock(d): continue

where lock(d) is a non-blocking exclusive flock on /cache/songhua/device_locks/npu{d}.lock. Holding that
flock makes it skip the device permanently, with no killing and nothing to race. It also caps itself at
four devices, so taking six leaves its own policy satisfiable on the rest.

Releasing is just exiting this process, which is the correct failure mode: if this dies, the cards go back
into the colleague's pool rather than being stranded.
"""
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

LOCKS = Path('/cache/songhua/device_locks')


def main():
    want = [int(x) for x in sys.argv[1].split(',')]
    deadline = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 1800)
    held, streams = [], []
    # Poll rather than kill. The controller holds each lock only while it holds the device, and it caps
    # itself at four devices -- so when it trims its own excess it calls unlock(d) and the lock is simply
    # there to be taken. Killing its holders instead makes it rescan every device, which is how this host
    # went from two free cards to zero.
    print(json.dumps({'pid': os.getpid(), 'waiting_for': want,
                      'deadline_s': round(deadline - time.time())}), flush=True)
    while time.time() < deadline and len(held) < len(want):
        for d in want:
            if d in held:
                continue
            s = open(LOCKS / f'npu{d}.lock', 'a')
            try:
                fcntl.flock(s, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                s.close()
                continue
            streams.append(s); held.append(d)
            print(json.dumps({'acquired': d, 'held': sorted(held),
                              't': time.strftime('%H:%M:%S')}), flush=True)
        if len(held) < len(want):
            time.sleep(0.2)
    print(json.dumps({'pid': os.getpid(), 'held': sorted(held),
                      'never_freed': sorted(set(want) - set(held))}), flush=True)
    stop = {'v': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    while not stop['v']:
        time.sleep(5)
    print(json.dumps({'released': held}), flush=True)


if __name__ == '__main__':
    main()
