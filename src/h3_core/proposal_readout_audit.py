"""Equal49-call true-H3 readout audit after the five-step adapter smoke.

Reference and learned condition guides share the same actual initial X0,
sampling noise, neutral context, and448tokens. This is a TRAIN-source interface
audit. Shared classical physical protection is separately measured and is not
attributed to the learned adapter.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file,save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from proposal_condition import NativeProposalCondition
from observation_quotient import ObservationQuotient


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['export','collect']);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--item');args=p.parse_args();torch.set_num_threads(4)
    data=Path(args.data);out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    preparation=json.loads((data/'preparation.json').read_text())
    if args.phase=='export':
        values=load_file(str(data/'conditioning.safetensors'));adapter=NativeProposalCondition().eval();adapter.load_state_dict(load_file(str(data/'adapter.safetensors')))
        with torch.no_grad():state=adapter.transition(torch.zeros(1,32),values['reference'],values['innovation'],values['geometry'],True);learned=adapter.condition(values['reference'],values['current'],state)
        guides={'reference':values['reference'],'learned':learned}
        for method,guide in guides.items():
            folder=out/method;folder.mkdir(exist_ok=True)
            save_file({'guide':guide.contiguous()},str(folder/'guide.safetensors'))
        manifest={'scope':'TRAIN-only five-step adapter true49readout interface; no quality ornovelty claim','prefix_index':17,'observed_indices':[0,17],'canonical_query':'Static canonical memory at prefix17, native22 internalcontext; NOT actual futureframes0..21','resolved_frame_index':0,'conditioning_layout':'Whole synthetic canonical-memory22guide is anchored at nativequeryindex0; observednear timestamp17 remainsmetadata, not reindexed as physicaltime0','tokens_per_method':448,'methods':list(guides),'sampling_seed':101,'actual_initial_reference':'Same realBF16H3 generated X0seed101 forboth; referenceguide is E(X0), not E(upLR)','learned_condition_difference_rms':float((learned-values['reference']).square().mean().sqrt()),'state_norm':float(state.norm()),'preparation':preparation}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True);return
    if not args.item:raise ValueError('collect --item is evaluationonly and required')
    item=torch.load(args.item,map_location='cpu',weights_only=False);assert item['source']['split']=='train'
    rgb=torch.load(data/'rgb_audit_train_only.pt',map_location='cpu',weights_only=False);reference=rgb['reference_rgb'];target=item['supervision']['canonical_hr'].permute(2,0,1)[None].float()/255
    q=ObservationQuotient(device='cpu');packet=item['inputs']['packets'][17];near=packet['rgb'].permute(2,0,1)[None].float()/255;rectangle=item['source']['canonical_rectangle'];view=packet['world_rectangle']
    def axis(lo,hi,vlo,vhi):
        u=(torch.arange(256,dtype=torch.float64)+.5)/256*(vhi-vlo)+vlo;position=(u-lo)/(hi-lo)*256-.5;left=position.floor().long();fraction=position-left;matrix=torch.zeros(256,256,dtype=torch.float64);rows=torch.arange(256)
        for index,value in [(left,1-fraction),(left+1,fraction)]:
            valid=(index>=0)&(index<256);matrix[rows[valid],index[valid]]=value[valid]
        forward=q.matrix@matrix;right=forward.T@torch.linalg.solve(forward@forward.T,torch.eye(64,dtype=torch.float64));return forward,right
    wy,ry=axis(rectangle[1],rectangle[3],view[1],view[3]);wx,rx=axis(rectangle[0],rectangle[2],view[0],view[2])
    def observe(x):return (wy@x.double()@wx.T).float()
    def support(x):return q.null((ry@observe(q.null(x)).double()@rx.T).float())
    yy,xx=torch.meshgrid(torch.arange(256),torch.arange(256),indexing='ij');corners=((xx<32)|(xx>=224))&((yy<32)|(yy>=224));query=q.null(((xx+yy)%2*2-1).float()[None,None].expand(1,3,-1,-1)*corners);query=query/query.norm()
    query_far=float(q.observe(query).norm());query_near=float(observe(query).norm())
    first=item['inputs']['coarse'][0:1];result={'scope':'TRAIN-source posttraining readout audit; fivegradientsteps only. Sharedsupport/DC isclassiccontrol, not learnedbenefit.','manifest':json.loads((out/'manifest.json').read_text()),'query_measurement_norm':{'far':query_far,'near':query_near},'metrics':{}}
    def metrics(image):
        delta=image-reference
        return {'psnr_full_db':float(-10*torch.log10((image-target).square().mean().clamp_min(1e-12))),'near_real_LR_rmse':float((observe(image)-near).square().mean().sqrt()),'far_real_LR_rmse':float((q.observe(image)-first).square().mean().sqrt()),'outside_corners_revision_rms':float(delta[:,:,corners].square().mean().sqrt()),'jointly_unobserved_query_revision':float((delta*query).sum().abs()),'highfrequency_rms':float(q.null(image).square().mean().sqrt())}
    result['metrics']['immutable_X0']={'raw':metrics(reference)}
    result['metrics']['classical_near_innovation']={'common_far_box_dc':metrics(rgb['current_rgb'])}
    for method in ['reference','learned']:
        folder=out/method;record=json.loads((folder/'sample_result.json').read_text());assert record['status']=='complete' and len(record['forward_events'])==49
        decoded=torch.from_numpy(np.array(Image.open(folder/'output_21.png').convert('RGB'))).permute(2,0,1)[None].float()/255
        far=q.bounded(decoded,first);protected=q.bounded(reference+support(far-reference),first)
        result['metrics'][method]={'raw':metrics(decoded),'far_dc':metrics(far),'common_support_far_dc':metrics(protected),'actual_full50block_forwards':len(record['forward_events'])}
        torch.save({'rgb':protected},folder/'common_protected_rgb.pt');Image.fromarray((protected[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)).save(folder/'common_protected.png')
    result['training_target_rgb_range']=[float(rgb['target_rgb_train_only'].min()),float(rgb['target_rgb_train_only'].max())]
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
