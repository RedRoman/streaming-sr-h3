"""Strict past-only writer / single-future reader API for the pixel codec pilot.

Labels, source identity, auxiliary construction phase and target tensors are
returned separately from eligible inputs. This module does not train a codec.
"""
from pathlib import Path
import json
import torch


class CodecCorpus:
    def __init__(self,root,split):
        self.root=Path(root);self.split=split
        manifest=json.loads((self.root/'manifest.json').read_text())
        self.records=[r for r in manifest['records'] if r['split']==split]
        assert len(self.records)==(32 if split=='train' else 8)
        self.items=[]
        for record in self.records:
            item=torch.load(self.root/split/(record['id']+'.pt'),map_location='cpu',weights_only=False)
            worlds=[dict(kind='primary_natural',**item['primary_natural']),*item['train_auxiliary']]
            if split=='val':assert len(worlds)==1
            else:assert worlds[1]['kind']=='fixed_range_interior_base' and len(worlds)==7
            self.items.append(worlds)

    def inputs_and_label(self,source,world,query,device='cpu'):
        value=self.items[source][world]
        # Only this tensor enters write(). No future, labels, source index,
        # donor, construction metadata or physical GT is a writer argument.
        past=value['writer_input']['past_lr_uint8'][None].to(device).float()/255
        future=value['reader_input']['future_lr_uint8'][query:query+1].to(device).float()/255
        phase=value['reader_input']['future_phase_nearLR'][query:query+1].to(device).float()
        target=value['supervision']['target_RGB'][None].to(device)
        return dict(past=past),dict(future=future,phase=phase),dict(target=target)

    def collision_pair(self,source,query,device='cpu'):
        assert self.split=='train'
        worlds=self.items[source];left=1
        right=next(i for i,w in enumerate(worlds) if w['kind']=='future_null_collision' and w['construction_phase']==query)
        wi,ri,yi=self.inputs_and_label(source,left,query,device)
        wj,rj,yj=self.inputs_and_label(source,right,query,device)
        assert torch.equal(ri['future'],rj['future']) and torch.equal(ri['phase'],rj['phase'])
        return (wi,ri,yi),(wj,rj,yj)

    def all_writer_patches(self):
        """TRAIN-only codebook fit data, with no side information or GT."""
        assert self.split=='train'
        return torch.stack([world['writer_input']['past_lr_uint8']
                            for worlds in self.items for world in worlds]).float()/255

    def check_future_payload_independence(self):
        count=0
        for si,worlds in enumerate(self.items):
            for wi in range(len(worlds)):
                first=self.inputs_and_label(si,wi,0)[0]['past']
                for q in range(1,4):assert torch.equal(first,self.inputs_and_label(si,wi,q)[0]['past'])
                count+=1
        return count
