"""Cache the same fixed LR-only DIS correspondence for a declared input case."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from video_correspondence import LRCorrespondence

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    case=next(c for c in json.loads(a.manifest.read_text())['cases'] if c['id']==a.case)
    a.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(4)
    tracker=LRCorrespondence();rows=[];hashes=[]
    for i,path in enumerate(case['lr_paths']):
        path=ROOT/'data/real_video'/path;hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(tracker.push(np.asarray(Image.open(path).convert('RGB')),f'{case["sequence"]}:{i}',i))
    assert hashes==case['observed_lr_sha256']
    np.savez_compressed(a.out/'lr_correspondence.npz',**{key:np.stack([getattr(r,key) for r in rows]) for key in ['current_to_previous','current_to_canonical','pair_confidence','canonical_confidence']})
    (a.out/'provenance.json').write_text(json.dumps({'case':a.case,'lr_sha256':hashes,'inputs':'Observed LR only, no HR or annotation','parameters':{'direct_interval':17,'photo_sigma':.05,'cycle_sigma':1},'opencv':cv2.__version__,'source_sha256':hashlib.sha256((ROOT/'src/video_correspondence.py').read_bytes()).hexdigest()},indent=2))
    print(json.dumps({'case':a.case,'frames':len(rows),'shape_hw':list(rows[0].pair_confidence.shape)}))
