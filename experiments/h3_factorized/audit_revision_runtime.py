"""Actual compressed streaming parity and packet-count complexity audit."""
import argparse,csv,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_model import build,KINDS
from revision_utils import batch
from revision_runtime import RevisionRuntime


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--models',required=True);p.add_argument('--checkpoint',choices=['best','last'],default='best');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=17);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;out=Path(args.out);out.mkdir(parents=True,exist_ok=True);manifest=json.loads((Path(args.cache)/'manifest.json').read_text());prov=next(x for x in manifest['items'] if x['split']=='train');item=torch.load(Path(args.cache)/(prov['id']+'.pt'),map_location='cpu',weights_only=False);inputs,_=batch([item],'cuda',rgb=False);rows=[];parity=[];start=time.time()
    def timecall(fn,repeats=5):
        samples=[]
        for _ in range(repeats):
            torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();samples.append(time.perf_counter()-t)
        return sorted(samples)[len(samples)//2]
    with torch.no_grad():
        for kind in KINDS:
            model=build(kind).cuda().eval();model.load_state_dict(torch.load(Path(args.models)/f'{kind}_seed{args.seed}'/(args.checkpoint+'.pt'),map_location='cpu',weights_only=False)['model']);expected,_,_=model.rollout(inputs);runtime=RevisionRuntime(model);maxerr=0;duplicate=True;readonly=True
            for step in range(18):
                for j,arrival in enumerate(inputs['packet_steps'][0].tolist()):
                    if arrival==step:
                        runtime.ingest(str(j),inputs['packet_delta'][:,j],inputs['packet_proxy'][:,j],inputs['packet_masks'][:,j],int(inputs['packet_groups'][0,j]));before=len(runtime.h);duplicate&=not runtime.ingest(str(j),inputs['packet_delta'][:,j]*99,inputs['packet_proxy'][:,j],inputs['packet_masks'][:,j],int(inputs['packet_groups'][0,j]));duplicate&=len(runtime.h)==before
                runtime.update(inputs['group_features'][:,step],{str(j):inputs['weights'][:,step,j].unsqueeze(2) for j in range(8) if str(j) in runtime.h})
                if step in inputs['query_steps'][0].tolist():
                    qi=inputs['query_steps'][0].tolist().index(step);before=runtime.canonical.clone();pred=runtime.query(inputs['query_latents'][:,qi],inputs['query_features'][:,qi],int(inputs['query_groups'][0,qi])<0);readonly&=torch.equal(before,runtime.canonical);maxerr=max(maxerr,float((pred-expected[:,qi]).abs().max()))
            assert maxerr<2e-4,(kind,maxerr);assert duplicate and readonly
            parity.append(dict(kind=kind,stream_vs_vectorized_maxabs=maxerr,duplicate_idempotent=duplicate,query_readonly=readonly,payload_bytes=runtime.payload(),native_codes_retained=False))
            feature=model.lift_packets(inputs)[:,0];q1=inputs['group_features'][:,3].clone();q2=inputs['group_features'][:,7].clone();q2[...,0]+=.01
            for n in [8,32,128,512]:
                rt=RevisionRuntime(model)
                for j in range(n):rt.ingest_feature(str(j),feature.clone(),j%2)
                weights={str(j):torch.ones_like(rt.weights[str(j)]) for j in range(n)};rt.update(q1,weights)
                toggle=[False]
                def revise():toggle[0]=not toggle[0];rt.update(q2 if toggle[0] else q1,weights)
                total=timecall(revise,3 if n>=128 else 5)
                if kind.startswith('closed'):
                    mm=rt.groups[0];ww=rt.ws[0];isolated=timecall(lambda:model.core.rebase(mm,ww,q1[:,0,:12],q2[:,0,:12]))
                elif kind=='nonlinear_group':isolated=timecall(lambda:model.core.canonical(rt.groups[0]/rt.ws[0],q2[:,0,:12]))
                elif kind=='factor_cache_mlp':
                    hh=torch.cat(list(rt.h.values()));qq=q2[:,0,:12].expand(n,-1);isolated=timecall(lambda:model.core.canonical(hh,qq),3)
                else:isolated=None
                payload=rt.payload();common=dict(reference_LR=49152,current_LR=49152,current_native_anchor_float32=172032,current_decoded_RGB_scratch=786432,coarse_descriptors=768*n,group_controller_features=2*18*4)
                rows.append(dict(kind=kind,N_packets=n,N_groups=2,actual_whole_state_revision_seconds=total,isolated_appearance_action_seconds=isolated,persistent_compressed_runtime_tensor_and_ID_bytes=sum(payload.values()),controller_reference_descriptor_group_bytes=common['reference_LR']+common['coarse_descriptors']+common['group_controller_features'],query_current_LR_anchor_decoded_scratch_bytes=common['current_LR']+common['current_native_anchor_float32']+common['current_decoded_RGB_scratch'],shared_camera_operators_bytes=524288,model_FP32_weights_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),raw_LR_factor_cache_if_retained_bytes=n*49152,raw_full_native_delta_proxy_cache_not_retained_bytes=n*2*172032,notes='N>8 repeats actual feature solely for complexity, not independent observations; controller inference and native E/D excluded from isolated cost; whole runtime update includes changed-ID scan and allocation; groupcount2fixed'))
            print(json.dumps(parity[-1]),flush=True)
        with (out/'state_cost_and_revision_runtime.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (out/'runtime_parity.json').write_text(json.dumps(dict(passed=True,parity=parity,seconds=time.time()-start,scope='One train source, cached current query code. Actual incremental h-only state vs vectorized training rollout; no test/no nativeD calls. Complexity repetitions are not new samples.'),indent=2))


if __name__=='__main__':main()
