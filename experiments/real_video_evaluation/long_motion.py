"""Fixed adjacent LR-only flow for evaluating real long-stream outputs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from video_correspondence import pair_correspondence, grid
DATA = ROOT / 'data/long_video'
OUT = ROOT / 'results/long_video_motion'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(video):
    folder = OUT / video['sequence']
    folder.mkdir(parents=True, exist_ok=True)
    records = video['frames_metadata']
    height, width = list(reversed(records[0]['lr_wh']))
    maps = np.lib.format.open_memmap(folder / 'current_to_previous.npy', mode='w+',
                                    dtype=np.float32, shape=(len(records), height, width, 2))
    confidence = np.lib.format.open_memmap(folder / 'pair_confidence.npy', mode='w+',
                                          dtype=np.float32, shape=(len(records), height, width))
    previous = None
    begin = time.perf_counter()
    for i, frame in enumerate(records):
        path = DATA / frame['lr_x4']
        assert sha(path) == frame['lr_sha256']
        current = np.asarray(Image.open(path).convert('RGB')).copy()
        if i == 0:
            maps[i] = grid(height, width)
            confidence[i] = 1
        else:
            maps[i], confidence[i], _ = pair_correspondence(previous, current)
        previous = current
    maps.flush(); confidence.flush()
    result = {'sequence': video['sequence'], 'frames': len(records), 'seconds': time.perf_counter() - begin,
              'observed_lr_sha256': [f['lr_sha256'] for f in records], 'hr_or_annotations_read': False,
              'protocol': 'Existing DIS-medium independent forward/backward LR-pair matcher, photo_sigma=.05 cycle_sigma=1',
              'opencv': cv2.__version__, 'script_sha256': sha(Path(__file__)),
              'matcher_sha256': sha(ROOT / 'src/video_correspondence.py'),
              'arrays_sha256': {p.name: sha(p) for p in folder.glob('*.npy')}}
    (folder / 'provenance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ['sequence', 'frames', 'seconds']}), flush=True)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    videos = json.loads((DATA / 'derived_h3_manifest.json').read_text())['videos']
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, videos))
