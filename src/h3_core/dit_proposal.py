"""Immutable, real H3 proposal initialization and separate RGB audit.

Preparation exports only the first already-observed LR packet and its native
22-frame repeated-LR encoding. The external official sampler sees no teacher,
future packet, or HR. Collection happens after sampling and may read HR for
evaluation, never for choosing a sample, seed, or output frame.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from observation_quotient import ObservationQuotient


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_tensor(path):
    return torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2,0,1)[None].float()/255


def save_image(tensor,path):
    Image.fromarray((tensor[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)).save(path)


def prepare(args):
    item=torch.load(args.item,map_location='cpu',weights_only=False)
    packet=item['inputs']['packets'][0]
    assert packet['index']==0 and item['source']['context_frames']==22
    guide=item['inputs']['anchor'].contiguous()
    assert guide.shape==(1,24,7,16,16) and torch.isfinite(guide).all()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    first={'rgb':packet['rgb'],'visible_mask':packet['visible_mask'],
           'observation_id':packet['observation_id'],'index':0,
           'world_rectangle':packet['world_rectangle']}
    torch.save({'first_packet':first,'canonical_rectangle':item['source']['canonical_rectangle']},out/'observed_prefix.pt')
    observed=packet['rgb'].permute(2,0,1)[None].float()/255
    save_image(observed,out/'observed_lr.png')
    save_image(F.interpolate(observed,(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1),out/'bicubic.png')
    manifest={'scope':'Fixed actual H3 DiT-generated identity proposals; interface pilot, not restored-video efficacy',
              'source':item['source'],'observed_packet_ids':[packet['observation_id']],
              'latest_observed_index':0,'distinct_observations':1,
              'native_repeated_context_frames':22,'guide_shape':list(guide.shape),
              'guide_tokens':448,'resolved_frame_index':0,'canonical_readout_frame':21,
              'input_export':'Only inputs.anchor and first LR packet exported; original cache supervision is not exported or passed to the sampler.',
              'preprocessing':'Cached actual native H3 encode(2*bicubic(LR)-1), 22 copies of already-observed frame0; no official UI8bit resize truncation.',
              'seeds':args.seeds,'sampler':'Unmodified experiments/h3_backbone/staged_bf16_guide.py',
              'sample_protocol':'Full BF16 H3, neutral real Qwen cache, exact49Euler official grid; same guide and per-seed proposal for all later controls',
              'evaluation_selection':'Seed list and frame21 fixed before sampling; no HR selection.',
              'protected_identity':'Each seed uses its own fixed X0; unknown-region identity is never replaced by a common HR target.',
              'initial_rgb_storage':'Official native decoder output saved as8bit PNG; identical quantized X0 is shared by every control. Persistent DC identity is then stored float32.',
              'prefix_sha256':sha(out/'observed_prefix.pt')}
    for seed in args.seeds:
        folder=out/f'seed{seed}';folder.mkdir(exist_ok=True)
        destination=folder/'guide.safetensors'
        if destination.exists():raise FileExistsError(f'Refusing to overwrite immutable guide {destination}')
        save_file({'guide':guide},str(destination))
        (folder/'conditioning_manifest.json').write_text(json.dumps({**manifest,'seed':seed,'guide_sha256':sha(destination)},indent=2))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'event':'prepared','out':str(out),'seeds':args.seeds,'shape':list(guide.shape),'distinct_observations':1}),flush=True)


def collect(args):
    out=Path(args.out);manifest=json.loads((out/'manifest.json').read_text())
    prefix=torch.load(out/'observed_prefix.pt',weights_only=False,map_location='cpu');packet=prefix['first_packet']
    observed=packet['rgb'].permute(2,0,1)[None].float()/255
    q=ObservationQuotient(device='cpu');rows={};identities={}
    target=None
    if args.evaluation_item:
        evaluation=torch.load(args.evaluation_item,map_location='cpu',weights_only=False)
        assert evaluation['source']['source_hash']==manifest['source']['source_hash']
        target=evaluation['supervision']['canonical_hr'].permute(2,0,1)[None].float()/255
    def metrics(rgb):
        result={'first_lr_rmse':float((q.observe(rgb)-observed).square().mean().sqrt()),'gradient_energy':float((rgb[:,:,:,1:]-rgb[:,:,:,:-1]).square().mean()+(rgb[:,:,1:]-rgb[:,:,:-1]).square().mean()),'highfrequency_rms':float(q.null(rgb).square().mean().sqrt())}
        if target is not None:result['psnr_db']=float(-10*torch.log10((rgb-target).square().mean().clamp_min(1e-12)))
        return result
    for seed in manifest['seeds']:
        folder=out/f'seed{seed}'
        if not (folder/'output_21.png').exists():continue
        sampling=json.loads((folder/'sample_result.json').read_text());decoding=json.loads((folder/'decode_result.json').read_text())
        assert sampling['status']==decoding['status']=='complete'
        assert len(sampling['forward_events'])==49 and all(v['video_finite'] and v['audio_finite'] for v in sampling['forward_events'])
        raw=image_tensor(folder/'output_21.png');bounded=q.bounded(raw,observed)
        # Float32 identity is the persistent artifact; PNG is a preview only.
        identity_path=folder/'immutable_identity.pt'
        if identity_path.exists():
            saved=torch.load(identity_path,map_location='cpu',weights_only=False)
            assert torch.equal(saved['rgb'],bounded) and torch.equal(saved['raw_rgb'],raw),'Immutable reference changed'
        else:torch.save({'rgb':bounded,'raw_rgb':raw,'seed':seed,'observation_id':packet['observation_id'],'frame':21},identity_path)
        save_image(bounded,folder/'identity_dc.png');identities[seed]=bounded
        row={'raw':metrics(raw),'dc':metrics(bounded),'actual_full50block_forwards':len(sampling['forward_events']),
             'sampling_seconds':sampling['sampling_seconds_including_offload_and_observers'],
             'guide_sha256':sha(folder/'guide.safetensors'),'sample_sha256':sha(folder/'sampled_av_latents.safetensors'),
             'identity_sha256':sha(folder/'immutable_identity.pt'),'actual_latest_distinct_observation_index':0,
             'warning':'Sampler generic scope text may say5frames; actual tensor temporal7 and22 RGB outputs are recorded here.'}
        if (folder/'guide_roundtrip_21.png').exists():
            vae=image_tensor(folder/'guide_roundtrip_21.png');row['vae_guide_raw']=metrics(vae);row['vae_guide_dc']=metrics(q.bounded(vae,observed))
        rows[str(seed)]=row
    result={'manifest':manifest,'scope':'Post-sampling evaluation; no learned update trained yet. DC is classical and matched across controls.','items':rows,'bicubic':metrics(image_tensor(out/'bicubic.png')),'pairwise_identity_mse':{f'{a}-{b}':float((identities[a]-identities[b]).square().mean()) for a in identities for b in identities if a<b}}
    destination=out/('initial_proposal_evaluation.json' if args.evaluation_item else 'initial_proposal_audit.json')
    destination.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','collect']);p.add_argument('--item');p.add_argument('--evaluation-item');p.add_argument('--out',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[101,202,303]);args=p.parse_args();torch.set_num_threads(4)
    if args.phase=='prepare':
        if not args.item:raise ValueError('prepare requires --item')
        prepare(args)
    else:collect(args)


if __name__=='__main__':main()
