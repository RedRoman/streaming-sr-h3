"""Scan LaSOT annotations (pinned hf-mirror, range requests only) for revisit events.

Selection uses annotations only (never model outputs). The criteria are frozen in
PROTOCOL and written to disk before any network access.
"""
import argparse, hashlib, json, struct, threading, time, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REV = 'a97464600e8c1ab91eb91a47332a20732830585e'
URL = 'https://hf-mirror.com/datasets/l-lt/LaSOT/resolve/{rev}/{cat}.zip'
PROTOCOL = {
    'version': 'revisit_scan_v1',
    'source': f'hf-mirror l-lt/LaSOT @ {REV} (third-party mirror; CRC-checked members)',
    'absent_definition': 'full_occlusion OR out_of_view',
    'min_absent_frames': 30,
    'pre_post_visible_frames': 22,
    'hr_rule': 'downscale so height=480 if source height>480 (never upscale), then center crop to multiples of 32',
    'size_rule_primary_px_at_hr': 96,
    'size_rule_secondary_px_at_hr': 64,
    'size_statistic': 'median of min(bbox_w,bbox_h) over the pre window and over the post window, both must pass',
    'bbox_inside_crop': 'median bbox center must lie inside the HR center crop in both windows',
    'model_outputs_used': False,
}
LOCK = threading.Lock()


def fetch(cat, size, lo, hi, cache):
    path = cache / cat / f'{lo}_{hi}.bin'
    if path.exists():
        return path.read_bytes()
    req = urllib.request.Request(URL.format(rev=REV, cat=cat), headers={'Range': f'bytes={lo}-{hi}'})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                assert r.status == 206 and r.headers.get('Content-Range') == f'bytes {lo}-{hi}/{size}', (r.status, dict(r.headers))
                raw = r.read()
            assert len(raw) == hi - lo + 1
            break
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(3 * (attempt + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def central(cat, size, cache):
    tail = fetch(cat, size, size - 65557, size - 1, cache)
    pos = tail.rfind(b'PK\x05\x06'); assert pos >= 0
    count, cdsize, offset = struct.unpack_from('<4s4H2IH', tail, pos)[4:7]
    if count == 65535 or cdsize == 0xffffffff or offset == 0xffffffff:
        pos64 = tail.rfind(b'PK\x06\x06'); assert pos64 >= 0
        count, cdsize, offset = struct.unpack_from('<4sQ2H2I4Q', tail, pos64)[7:10]
    blob = fetch(cat, size, offset, offset + cdsize - 1, cache)
    entries, pos = [], 0
    while pos < len(blob):
        h = struct.unpack_from('<4s6H3I5H2I', blob, pos); assert h[0] == b'PK\x01\x02'
        name = blob[pos + 46:pos + 46 + h[10]].decode('utf-8')
        extra = blob[pos + 46 + h[10]:pos + 46 + h[10] + h[11]]
        csize, usize, off = h[8], h[9], h[16]
        if 0xffffffff in (csize, usize, off):  # zip64 extra field
            q = 0
            while q < len(extra):
                tag, ln = struct.unpack_from('<2H', extra, q)
                if tag == 1:
                    vals = list(struct.unpack_from('<' + 'Q' * (ln // 8), extra, q + 4)); k = 0
                    if usize == 0xffffffff: usize = vals[k]; k += 1
                    if csize == 0xffffffff: csize = vals[k]; k += 1
                    if off == 0xffffffff: off = vals[k]; k += 1
                q += 4 + ln
        entries.append(dict(name=name, method=h[4], crc32=h[7], csize=csize, usize=usize, offset=off))
        pos += 46 + h[10] + h[11] + h[12]
    assert len(entries) == count
    return entries, hashlib.sha256(blob).hexdigest()


def member(cat, size, e, cache):
    lo = e['offset']; hi = min(size - 1, lo + 30 + len(e['name'].encode()) + 1024 + e['csize'])
    blob = fetch(cat, size, lo, hi, cache)
    h = struct.unpack_from('<4s5H3I2H', blob, 0); assert h[0] == b'PK\x03\x04'
    begin = 30 + h[9] + h[10]
    raw = blob[begin:begin + e['csize']]
    raw = zlib.decompress(raw, -15) if e['method'] == 8 else raw
    assert len(raw) == e['usize'] and zlib.crc32(raw) & 0xffffffff == e['crc32']
    return raw


def jpeg_size(raw):
    i = 2
    while i < len(raw):
        assert raw[i] == 0xFF
        marker = raw[i + 1]; ln = struct.unpack('>H', raw[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            hgt, wid = struct.unpack('>HH', raw[i + 5:i + 9]); return wid, hgt
        i += 2 + ln
    raise ValueError('no SOF')


def hr_geometry(w, h):
    s = 480 / h if h > 480 else 1.0
    sw, sh = round(w * s), round(h * s)
    cw, ch = sw // 32 * 32, sh // 32 * 32
    return s, (sw, sh), ((sw - cw) // 2, (sh - ch) // 2, cw, ch)


def median(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def events_for(flags_full, flags_out, boxes, geom):
    s, _, (cx, cy, cw, ch) = geom
    absent = [bool(a or b) for a, b in zip(flags_full, flags_out)]
    K, out, n, i = PROTOCOL['pre_post_visible_frames'], [], len(absent), 0
    while i < n:
        if not absent[i]:
            i += 1; continue
        j = i
        while j < n and absent[j]:
            j += 1
        L = j - i
        rec = {'absent_start': i, 'return': j, 'absent_frames': L}
        pre, post = list(range(i - K, i)), list(range(j, j + K))
        ok = L >= PROTOCOL['min_absent_frames'] and i - K >= 0 and j + K <= n
        ok = ok and not any(absent[t] for t in pre + post)
        ok = ok and all(boxes[t][2] > 0 and boxes[t][3] > 0 for t in pre + post)
        if ok:
            for tag, win in (('pre', pre), ('post', post)):
                side = median([min(boxes[t][2], boxes[t][3]) * s for t in win])
                mx = median([(boxes[t][0] + boxes[t][2] / 2) * s - cx for t in win])
                my = median([(boxes[t][1] + boxes[t][3] / 2) * s - cy for t in win])
                rec[f'{tag}_min_side_hr'] = round(side, 1)
                rec[f'{tag}_center_in_crop'] = bool(0 <= mx < cw and 0 <= my < ch)
            inside = rec['pre_center_in_crop'] and rec['post_center_in_crop']
            m = min(rec['pre_min_side_hr'], rec['post_min_side_hr'])
            rec['pass_primary'] = bool(inside and m >= PROTOCOL['size_rule_primary_px_at_hr'])
            rec['pass_secondary'] = bool(inside and m >= PROTOCOL['size_rule_secondary_px_at_hr'])
        else:
            rec['pass_primary'] = rec['pass_secondary'] = False
        rec['window_eligible'] = ok
        out.append(rec); i = j
    return out


def prefetch(cat, size, entries, cache, gap=2 << 20):
    """Fetch the needed members with coalesced ranges; return name -> raw bytes (CRC-checked)."""
    spans = sorted((e['offset'], e['offset'] + 30 + len(e['name'].encode()) + 1024 + e['csize'], e) for e in entries)
    groups, cur = [], None
    for lo, hi, e in spans:
        if cur and lo - cur[1] <= gap:
            cur[1] = max(cur[1], hi); cur[2].append(e)
        else:
            cur = [lo, hi, [e]]; groups.append(cur)
    out = {}
    for lo, hi, es in groups:
        hi = min(size - 1, hi)
        blob = fetch(cat, size, lo, hi, cache)
        for e in es:
            o = e['offset'] - lo
            h = struct.unpack_from('<4s5H3I2H', blob, o); assert h[0] == b'PK\x03\x04'
            begin = o + 30 + h[9] + h[10]
            raw = blob[begin:begin + e['csize']]
            raw = zlib.decompress(raw, -15) if e['method'] == 8 else raw
            assert len(raw) == e['usize'] and zlib.crc32(raw) & 0xffffffff == e['crc32'], e['name']
            out[e['name']] = raw
    return out


def scan(cat, size, cache, outdir):
    entries, csha = central(cat, size, cache)
    names = {e['name']: e for e in entries}
    seqs = sorted({e['name'].split('/')[0] for e in entries if e['name'].endswith('/full_occlusion.txt')})
    need = [e for e in entries if e['name'].split('/')[-1] in ('full_occlusion.txt', 'out_of_view.txt', 'groundtruth.txt', 'nlp.txt')]
    for seq in seqs:
        need.append(names[min(n for n in names if n.startswith(f'{seq}/img/') and n.endswith('.jpg'))])
    raw = prefetch(cat, size, need, cache)
    rows = []
    for seq in seqs:
        rd = lambda f: raw[f'{seq}/{f}'].decode()
        full = [int(x) for x in rd('full_occlusion.txt').replace('\n', ',').split(',') if x.strip()]
        outv = [int(x) for x in rd('out_of_view.txt').replace('\n', ',').split(',') if x.strip()]
        boxes = [tuple(float(v) for v in line.split(',')) for line in rd('groundtruth.txt').strip().splitlines()]
        imgs = sorted(n for n in names if n.startswith(f'{seq}/img/') and n.endswith('.jpg'))
        assert len(full) == len(outv) == len(boxes) == len(imgs), (seq, len(full), len(outv), len(boxes), len(imgs))
        w, h = jpeg_size(raw[imgs[0]])
        geom = hr_geometry(w, h)
        ev = events_for(full, outv, boxes, geom)
        rows.append({'sequence': seq, 'frames': len(imgs), 'source_wh': [w, h], 'hr_scale': geom[0],
                     'hr_crop_xywh': list(geom[2]), 'nlp': rd('nlp.txt').strip() if f'{seq}/nlp.txt' in names else '',
                     'events': ev})
    res = {'category': cat, 'archive_bytes': size, 'central_sha256': csha, 'sequences': rows,
           'n_primary': sum(e['pass_primary'] for r in rows for e in r['events']),
           'n_secondary': sum(e['pass_secondary'] for r in rows for e in r['events'])}
    (outdir / f'{cat}.json').write_text(json.dumps(res, indent=1))
    with LOCK:
        print(json.dumps({'category': cat, 'sequences': len(rows), 'primary': res['n_primary'], 'secondary': res['n_secondary']}), flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tree', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--categories', nargs='*')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    proto = a.out / 'protocol.json'
    if proto.exists():
        assert json.loads(proto.read_text()) == PROTOCOL, 'protocol drift'
    else:
        proto.write_text(json.dumps(PROTOCOL, indent=1))
    tree = {x['path'][:-4]: x['size'] for x in json.loads(a.tree.read_text()) if str(x.get('path', '')).endswith('.zip')}
    cats = a.categories or sorted(tree)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda c: scan(c, tree[c], a.cache, a.out), cats))
    summary = {'protocol_sha256': hashlib.sha256(proto.read_bytes()).hexdigest(), 'categories': len(res),
               'sequences': sum(len(r['sequences']) for r in res),
               'primary_events': sum(r['n_primary'] for r in res), 'secondary_events': sum(r['n_secondary'] for r in res),
               'per_category': {r['category']: [r['n_primary'], r['n_secondary']] for r in res}}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_category'}), flush=True)


if __name__ == '__main__':
    main()
