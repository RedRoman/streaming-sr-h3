"""Parallel HTTP range download with resume and SHA256 verification."""
import argparse, concurrent.futures as cf, hashlib, os, subprocess, sys, time
from pathlib import Path

def fetch(url, part, lo, hi):
    want = hi - lo + 1
    for attempt in range(30):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return
        if have > want:
            part.unlink(); have = 0
        r = subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '5', '--connect-timeout', '20',
                            '--speed-limit', '100000', '--speed-time', '60',
                            '-r', f'{lo + have}-{hi}', url], stdout=open(part, 'ab'), stderr=subprocess.PIPE)
        if r.returncode:
            print(f'retry {part.name} attempt {attempt}: {r.stderr.decode()[-200:]}', flush=True)
            time.sleep(5)
    raise RuntimeError(f'failed {part}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--size', type=int, required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--workers', type=int, default=16); p.add_argument('--chunk', type=int, default=512 << 20)
    a = p.parse_args()
    if a.out.exists() and a.out.stat().st_size == a.size:
        print('exists, verifying', flush=True)
    else:
        parts_dir = a.out.with_name(a.out.name + '.parts'); parts_dir.mkdir(parents=True, exist_ok=True)
        ranges = [(i, lo, min(lo + a.chunk, a.size) - 1) for i, lo in enumerate(range(0, a.size, a.chunk))]
        t0 = time.time()
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(fetch, a.url, parts_dir / f'{i:05d}', lo, hi): i for i, lo, hi in ranges}
            for n, f in enumerate(cf.as_completed(futs), 1):
                f.result()
                done = sum((parts_dir / f'{i:05d}').stat().st_size for i, _, _ in ranges if (parts_dir / f'{i:05d}').exists())
                print(f'{n}/{len(ranges)} parts, {done/1e9:.2f}/{a.size/1e9:.2f} GB, {done/1e6/(time.time()-t0):.1f} MB/s', flush=True)
        tmp = a.out.with_name(a.out.name + '.tmp')
        with open(tmp, 'wb') as w:
            for i, _, _ in ranges:
                with open(parts_dir / f'{i:05d}', 'rb') as r:
                    while b := r.read(64 << 20): w.write(b)
        assert tmp.stat().st_size == a.size
        tmp.replace(a.out)
        for i, _, _ in ranges: (parts_dir / f'{i:05d}').unlink()
        parts_dir.rmdir()
    h = hashlib.sha256()
    with open(a.out, 'rb') as r:
        while b := r.read(64 << 20): h.update(b)
    ok = h.hexdigest() == a.sha256
    print(f'SHA256 {"OK" if ok else "MISMATCH"} {h.hexdigest()}', flush=True)
    sys.exit(0 if ok else 2)

if __name__ == '__main__':
    main()
