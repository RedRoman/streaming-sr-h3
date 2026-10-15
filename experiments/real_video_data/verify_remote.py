#!/usr/bin/env python3
"""CPU-only transfer audit: every original/HR/LR/mask/H3crop payload hash."""
import argparse,hashlib,json,socket,time,zlib
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[2]/'data'/'real_video');args=ap.parse_args()
    p=args.data;start=time.time();m=json.loads((p/'manifest.json').read_text());count=0;checks=[]
    for clip in m['videos']:
        for f in clip['frames_metadata']:
            for key,digest in [('source_jpeg','source_sha256'),('hr','hr_sha256'),('lr_x4','lr_sha256')]:
                data=(p/f[key]).read_bytes()
                assert hashlib.sha256(data).hexdigest()==f[digest],f[key]
                if key=='source_jpeg':assert zlib.crc32(data)&0xffffffff==int(f['zip_crc32'],16)
                count+=1
        for mask in clip['segmentation_masks']:
            assert hashlib.sha256((p/mask['path']).read_bytes()).hexdigest()==mask['sha256'];count+=1
        checks.append(dict(sequence=clip['sequence'],split=clip['project_split'],frames=clip['frames'],verified=True))
    windows=json.loads((p/'h3_windows_39.json').read_text())['windows']
    for v in windows:
        for f in v['frames_metadata']:
            for key,digest in [('hr','hr_sha256'),('lr_x4','lr_sha256')]:
                assert hashlib.sha256((p/f[key]).read_bytes()).hexdigest()==f[digest];count+=1
    report=dict(passed=True,host=socket.gethostname(),data=str(p.resolve()),clips=len(checks),source_frames=sum(v['frames'] for v in m['videos']),h3_windows=len(windows),h3_frames=sum(w['frames'] for w in windows),payload_hashes_verified=count,source_jpeg_crc32_verified=True,manifest_sha256=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),sequences=checks,elapsed_seconds=time.time()-start,gpu_used=False)
    (p/'remote_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
