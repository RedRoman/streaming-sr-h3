"""Causal packets and supervised clips for the MiniMax-H3 SR experiments.

Natural photographs + known camera transforms form a controlled video dataset.
This is not a substitute for real-motion / estimated-correspondence evaluation.
Ground truth is kept outside the packet consumed by the restoration method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class LRPacket:
    index: int
    source_time: float
    rgb: np.ndarray
    world_rectangle: tuple[float, float, float, float]
    visible_mask: np.ndarray
    observation_id: str


class NaturalCameraSequence:
    """Render observed prefix; oracle geometry/visibility explicitly annotated."""
    def __init__(self, source, size=256, scale=4, frames=39, fps=24, protocol="zoom_revisit", seed=0):
        self.source = Path(source)
        self.image = Image.open(source).convert("RGB")
        self.size, self.scale, self.frames, self.fps = size, scale, frames, fps
        self.protocol, self.seed = protocol, seed
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        assert size % 32 == 0 and size % scale == 0
        assert frames % 17 == 5  # H3 native frame-count grid
        assert min(self.image.size) >= size

    def render(self, index):
        if not 0 <= index < self.frames:
            raise IndexError(index)
        progress = index / (self.frames-1)
        # Same initial/final field, with a genuine higher-information close view.
        zoom = 1 + max(0., 1-abs(progress-.5)/.30) if "zoom" in self.protocol else 1.
        field = self.size / zoom
        iw, ih = self.image.size
        move_x = 0.12*(iw-self.size)*np.sin(2*np.pi*progress)
        move_y = 0.12*(ih-self.size)*np.sin(2*np.pi*progress)
        cx, cy = iw/2 + move_x, ih/2 + move_y
        rect = (cx-field/2, cy-field/2, cx+field/2, cy+field/2)
        hr = self.image.resize((self.size,self.size), Image.Resampling.BICUBIC, box=rect)
        rgb = np.asarray(hr).astype(np.float32) / 255
        appearance = 1.
        if "lighting" in self.protocol:
            appearance = 1 + .15*np.sin(2*np.pi*progress)
            rgb = np.clip(rgb*appearance,0,1)
        visible = np.ones((self.size,self.size),np.uint8)*255
        if "revisit" in self.protocol and .62 <= progress <= .80:
            # Occluder is part of rendered GT; its texture must not enter the
            # background object's memory. The oracle mask is a controlled input.
            lo, hi = self.size//8, self.size*7//8
            rgb[lo:hi,:] = np.array([.23,.27,.31])
            visible[lo:hi,:] = 0
        hr = Image.fromarray(np.uint8(np.clip(rgb,0,1)*255))
        lr = hr.filter(ImageFilter.GaussianBlur(.6)).resize((self.size//self.scale,)*2,Image.Resampling.BICUBIC)
        rng = np.random.default_rng(self.seed*10000+index)
        lrf = np.asarray(lr).astype(np.float32)/255 + rng.normal(0,.003,(self.size//self.scale,self.size//self.scale,3))
        lr = np.uint8(np.clip(lrf,0,1)*255)
        mask = np.asarray(Image.fromarray(visible).resize(lr.shape[:2][::-1],Image.Resampling.NEAREST))
        packet = LRPacket(index,index/self.fps,lr,tuple(float(x) for x in rect),mask,f"{self.source_hash}:{self.seed}:{index}")
        return packet, np.asarray(hr), {"appearance_gain":float(appearance),"camera_zoom":float(zoom)}

    def packets(self):
        """Restoration input iterator exposes no high-resolution target."""
        for i in range(self.frames):
            yield self.render(i)[0]

    def save(self, root):
        root=Path(root)
        for name in ["lr","hr","visibility"]: (root/name).mkdir(parents=True,exist_ok=True)
        frames=[]
        for i in range(self.frames):
            packet,hr,extra=self.render(i)
            Image.fromarray(packet.rgb).save(root/"lr"/f"{i:04d}.png")
            Image.fromarray(hr).save(root/"hr"/f"{i:04d}.png")
            Image.fromarray(packet.visible_mask).save(root/"visibility"/f"{i:04d}.png")
            frames.append({"index":i,"source_time":packet.source_time,"world_rectangle":packet.world_rectangle,"observation_id":packet.observation_id,**extra})
        manifest={"source":str(self.source),"source_sha256":self.source_hash,"protocol":self.protocol,"seed":self.seed,"size":self.size,"scale":self.scale,"fps":self.fps,"frames":frames,"oracle_geometry":True,"oracle_visibility":True,"simulated_camera_not_real_motion":True,"hr_usage":"supervision/evaluation only; never part of LRPacket"}
        (root/"manifest.json").write_text(json.dumps(manifest,indent=2))
        return manifest


def make_manifest(root, output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    split_ids={s:sorted(p.name for p in (root/s).glob("*.jpg")) for s in ["train","val","test"]}
    assert [len(split_ids[s]) for s in ["train","val","test"]]==[200,100,200]
    assert not set(split_ids['train'])&set(split_ids['test'])
    assert not set(split_ids['train'])&set(split_ids['val'])
    result={"source":"BSDS500 official splits","root":str(root),"split_ids":split_ids,"target_frames":39,"output_size":256,"lr_size":64,"fps":24,"block_frames":22,"overlap_frames":5,"block_starts":[0,17],"wait_budget":"block-internal lookahead <=21 input frames, plus compute; no frames beyond emitted block endpoint","protocols":["zoom_revisit","zoom_lighting","revisit"],"primary_foundation":"MiniMax-H3 BF16 FL2VA preferred"}
    (output/"dataset_manifest.json").write_text(json.dumps(result,indent=2))
    for split,count in [('train',3),('val',1),('test',3)]:
        for i,name in enumerate(split_ids[split][:count]):
            protocol=result['protocols'][i%3]
            seq=NaturalCameraSequence(root/split/name,protocol=protocol,seed=17000+i)
            seq.save(output/f"{split}_{Path(name).stem}_{protocol}")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'data/BSR/BSDS500/data/images');p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'data/h3_camera');args=p.parse_args()
    r=make_manifest(args.source,args.output)
    print(json.dumps({"split_counts":{k:len(v) for k,v in r['split_ids'].items()},"saved_preview_sequences":7,"frames_per_sequence":39,"output":str(args.output)},indent=2))
