#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import av
import cv2
from PIL import Image, ImageDraw, ImageFont


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=False)
    started = time.monotonic()
    inp = json.loads(args.inputs.read_text())
    cv2.setNumThreads(1)
    canvas = Image.new('RGB', (768, 404), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    result = {'schema': 'openh3.targeted_source_pair_contact.v1',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'inputs_sha256': file_sha(args.inputs), 'script_sha256': file_sha(Path(__file__)),
              'parent_evidence_path': inp['parent_evidence_path'],
              'parent_evidence_sha256': inp['parent_evidence_sha256'],
              'source_order_rows': inp['source_ids'], 'columns': [0.1, 0.5, 0.9],
              'frame_index_contract': 'global_frame_index is unknown: backward-keyframe seeks do not establish a global ordinal. Actual selected PTS, timebase and ordinal within each seek decode are recorded. No nominal FPS multiplication is substituted.',
              'display_contract': 'RGB24, aspect-preserving INTER_AREA fit into 256x144, black letterbox; labels outside pixels. This contact sheet is a low-resolution viewing aid.',
              'source_records': [], 'exported_frames': 0, 'npu_calls': 0,
              'model_calls': 0, 'source_role_assignment': 'none', 'verdict': 'pending_manual_review',
              'software': {'pyav': av.__version__, 'opencv': cv2.__version__}}
    for row_index, expected in enumerate(inp['records']):
        path = Path(expected['path'])
        before = path.stat()
        sha = file_sha(path)
        if sha != expected['source_sha256']:
            raise ValueError('source SHA differs from frozen audit: ' + expected['source_id'])
        record = {'source_id': expected['source_id'], 'path': str(path), 'source_sha256': sha,
                  'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns, 'frames': []}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            record['time_base'] = str(stream.time_base)
            record['average_rate_metadata'] = str(stream.average_rate)
            for column, old in enumerate(expected['frames']):
                if time.monotonic() - started >= inp['budget_seconds'] - 2:
                    raise TimeoutError('targeted contact-sheet CPU budget exceeded')
                requested = old['requested_seconds']
                seek_timestamp = int(requested / stream.time_base)
                container.seek(seek_timestamp, stream=stream, backward=True, any_frame=False)
                for ordinal, frame in enumerate(container.decode(stream)):
                    if frame.pts is not None and float(frame.pts * stream.time_base) >= requested:
                        break
                else:
                    raise ValueError('requested presentation timestamp unavailable')
                rgb = frame.to_ndarray(format='rgb24')
                thumbnail = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
                thumbnail_sha = hashlib.sha256(thumbnail.tobytes()).hexdigest()
                if thumbnail_sha != old['thumbnail_rgb32_sha256'] or float(frame.pts * stream.time_base) != old['decoded_seconds']:
                    raise ValueError('selected frame does not reproduce frozen sparse evidence')
                height, width = rgb.shape[:2]
                ratio = min(256 / width, 144 / height)
                display = cv2.resize(rgb, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
                tile = Image.new('RGB', (256, 144), (0, 0, 0))
                tile.paste(Image.fromarray(display), ((256 - display.shape[1]) // 2, (144 - display.shape[0]) // 2))
                x, y = column * 256, row_index * 202
                canvas.paste(tile, (x, y))
                draw.text((x + 5, y + 148), expected['source_id'], fill=(0, 0, 0), font=font)
                draw.text((x + 5, y + 167), f"{old['fraction']:.0%} | t={float(frame.pts * stream.time_base):.6f}s", fill=(0, 0, 0), font=font)
                record['frames'].append({'fraction': old['fraction'], 'requested_seconds': requested,
                    'actual_pts': frame.pts, 'time_base': str(stream.time_base),
                    'actual_seconds': float(frame.pts * stream.time_base), 'global_frame_index': None,
                    'seek_timestamp_in_stream_timebase': seek_timestamp,
                    'decoded_ordinal_after_seek_zero_based': ordinal,
                    'incidental_frames_decoded_this_seek': ordinal + 1,
                    'rgb_shape_hwc': list(rgb.shape), 'rgb_dtype': str(rgb.dtype),
                    'full_decoded_rgb_pixel_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                    'thumbnail_rgb32_sha256': thumbnail_sha, 'reproduces_frozen_frame': True,
                    'display_rgb_shape_hwc': list(display.shape),
                    'display_content_rgb_pixel_sha256': hashlib.sha256(display.tobytes()).hexdigest()})
                result['exported_frames'] += 1
        after = path.stat()
        record['source_unchanged_during_read'] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        if not record['source_unchanged_during_read']:
            raise ValueError('source changed during reading')
        result['source_records'].append(record)
    contact = args.out_dir / 'contact_sheet.png'
    canvas.save(contact, optimize=True)
    result['contact_sheet'] = {'file': contact.name, 'sha256': file_sha(contact), 'bytes': contact.stat().st_size, 'size_wh': list(canvas.size)}
    if contact.stat().st_size > 1024 * 1024 or result['exported_frames'] != 6:
        raise ValueError('contact-sheet output budget mismatch')
    result['elapsed_seconds'] = time.monotonic() - started
    result['status'] = 'contact_sheet_prepared_no_source_decision'
    (args.out_dir / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'elapsed_seconds': result['elapsed_seconds'], 'contact_sheet': result['contact_sheet'], 'exported_frames': 6}))


if __name__ == '__main__':
    main()
