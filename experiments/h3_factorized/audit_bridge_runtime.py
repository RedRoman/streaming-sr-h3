"""Actual prefix native ledgers+compressed stream, costs/withdrawal/readonly checks."""
import argparse,csv,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_bridge import build_bridge,BRIDGE_ARMS,attach_native,bridge_rgb,shared_bridge
from revision_runtime import RevisionRuntime
from revision_utils import batch,Readout
from native_vae import load_vae,decode_last


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--models',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());source=next(s for s in manifest['items'] if s['split']=='train');item=torch.load(cache/(source['id']+'.pt'),map_location='cpu',weights_only=False);inputs,_=batch([item],'cuda');attach_native(inputs,item);near=item['classical_inputs']['near_LR'].cuda().float();vae=load_vae(a.vae,a.comfy);read=Readout(vae);rows=[];timings=[];start=time.time();calls=0
    def timed(arm,step,op,fn):
        torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();timings.append(dict(arm=arm,step=step,operation=op,seconds=time.perf_counter()-t));return v
    with torch.no_grad():
        for arm in BRIDGE_ARMS:
            model=build_bridge(arm).cuda().eval();cp=torch.load(Path(a.models)/f'{arm}_seed17/last.pt',map_location='cpu',weights_only=False);assert cp['step']==672;model.load_state_dict(cp['model']);vector=model.rollout(inputs)[0];runtime=RevisionRuntime(model);native={};proxy={};maxparity=0;duplicate=False;readonly=True;query_count=0
            for step in range(18):
                for j,t in enumerate(inputs['packet_steps'][0].tolist()):
                    if t==step:
                        # Owned half buffers, not a view retaining future cache.
                        native[j]=item['classical_inputs']['packet_full_latents'][j:j+1].clone().cuda();proxy[j]=item['inputs']['packet_proxy'][j:j+1].clone().cuda();runtime.ingest(j,native[j].float()-proxy[j].float(),proxy[j].float(),inputs['packet_masks'][:,j],int(inputs['packet_groups'][0,j]))
                if step==9:
                    before=dict(runtime.payload());duplicate=not runtime.ingest(3,native[3].float()-proxy[3].float(),proxy[3].float(),inputs['packet_masks'][:,3],int(inputs['packet_groups'][0,3]));assert before==runtime.payload()
                weights={j:inputs['weights'][:,step,j].unsqueeze(2) for j in native};timed(arm,step,'whole_neural_state_revision_no_nativeD_controller',lambda:runtime.update(inputs['group_features'][:,step],weights))
                if step in inputs['query_steps'][0].tolist():
                    qi=inputs['query_steps'][0].tolist().index(step);qneutral=int(inputs['query_groups'][0,qi])<0;before={k:v.clone() for k,v in runtime.h.items()};z=timed(arm,step,'neural_query_only',lambda:runtime.query(inputs['query_latents'][:,qi],inputs['query_features'][:,qi],qneutral));maxparity=max(maxparity,float((z-vector[:,qi]).abs().max()));repeat=runtime.query(inputs['query_latents'][:,qi],inputs['query_features'][:,qi],qneutral);readonly&=torch.equal(z,repeat) and all(torch.equal(v,runtime.h[k]) for k,v in before.items());decoded=[None]*8
                    for j in native:decoded[j]=timed(arm,step,'common_phi_nativeD_one_factor',lambda j=j:decode_last(vae,native[j].float()));calls+=1
                    base=timed(arm,step,'common7_bridge_fusions_from_arrivedphi',lambda:shared_bridge(read,inputs,near,[v if v is not None else inputs['query_base_rgb'][:,qi]*0 for v in decoded]));# Only current qi is a valid prefix output; other indices are scratch.
                    pixel=timed(arm,step,'nativeD_current_correction_plus_H',lambda:bridge_rgb(read,inputs,z,qi,base,model.bridge_skip));calls+=1;query_count+=1
                    if step==17:
                        timed(arm,step,'shared_final_DC64',lambda:read.far.apply(pixel,item['classical_inputs']['query_LR'][qi:qi+1].cuda().float()))
                        payload=runtime.payload();ledger=dict(native_full_half=sum(v.numel()*v.element_size() for v in native.values()),native_proxy_half=sum(v.numel()*v.element_size() for v in proxy.values()));controller=49152+len(native)*(3*8*8*4)+2*18*4;common_current=49152+24*7*16*16*4+3*256*256*4;phi_scratch=sum(v.numel()*v.element_size() for v in decoded if v is not None);weights_revision={j:v.clone() for j,v in weights.items()};weights_revision[0].zero_();timed(arm,step,'one_factor_withdraw_neural_update',lambda:runtime.update(inputs['group_features'][:,step],weights_revision));withdraw=runtime.query(inputs['query_latents'][:,qi],inputs['query_features'][:,qi],qneutral);vi={k:v.clone() for k,v in inputs.items()};vi['weights'][:,step,0]=0;fresh=model.rollout(vi)[0][:,qi];withdraw_parity=float((withdraw-fresh).abs().max());timed(arm,step,'withdraw_restore_neural_update',lambda:runtime.update(inputs['group_features'][:,step],weights));restored=runtime.query(inputs['query_latents'][:,qi],inputs['query_features'][:,qi],qneutral);restore=float((restored-z).abs().max());assert withdraw_parity<2e-5 and restore<2e-5
                        # Full24/native archive is required for the physical
                        # factor's later reinterpretation, no replay of raw HR.
                        wb=timed(arm,step,'withdraw_common7_bridge_fusions_no_D_if_phi_cached',lambda:shared_bridge(read,vi,near,decoded));assert torch.isfinite(wb).all()
                        rows.append(dict(arm=arm,source=source['id'],N_packets=len(native),N_groups=2,compressed_runtime_payload=payload,native_ledger_bytes=ledger,controller_reference_descriptors_group_bytes=controller,currentLR_native_anchor_decoded_buffer_bytes=common_current,cameras_bytes=524288,adapter_FP32_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),frozen_VAE_bytes=sum(p.numel()*p.element_size() for p in vae.parameters()),persistent_total_excluding_weights_and_cameras=sum(payload.values())+sum(ledger.values())+controller+common_current,query_phi_RGB_scratch_bytes=phi_scratch,single_bridge_RGB_buffer_bytes=3*256*256*4,actual_reference_executor_all7bridge_stack_scratch_bytes=base.numel()*base.element_size(),stream_vectorized_maxabs=maxparity,withdraw_fresh_rollout_maxabs=withdraw_parity,withdraw_restore_maxabs=restore,duplicate_ID_idempotent=duplicate,query_readonly=readonly,prefix_queries=query_count))
            assert maxparity<2e-5 and duplicate and readonly;del model,runtime;torch.cuda.empty_cache();print(json.dumps(dict(arm=arm,status='B5_stream_and_cost_passed',seconds=time.time()-start)),flush=True)
    with (out/'operations.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(timings[0]));w.writeheader();w.writerows(timings)
    (out/'state_and_parity.json').write_text(json.dumps(dict(arms=rows,actual_D_calls=calls,actual_E_calls=0,optimizer_updates=0,seconds=time.time()-start,scope='Actual prefix stream, one fixedTRAINsource not samples; native/proxy archives owned FP16; h/momentsFP32. Shared phi charged per query here, no retained RGB archive. For equal compute S0 also computes bridge; minimalS0 could omit phiD. Camera/controller/frozenVAE/adapters reported separately; Python allocator overhead excluded. Sharedbridge helper builds all7 scratch rows; only currentqi contains valid prefix decoded packets, other rows not read. N growth: per-factor native+proxy172032B plush/weights/descriptor/ID~30520B; groupaction fixed2 vs fullrevision/phiD linearN, no O1 claim.'),indent=2))


if __name__=='__main__':main()
