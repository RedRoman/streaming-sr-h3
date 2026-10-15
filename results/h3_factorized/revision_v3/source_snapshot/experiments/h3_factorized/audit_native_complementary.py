"""Frozen native H3 E/D capacity audit of complementary natural LR packets.

No training, no test sources, no HR-derived runtime state. Neutral appearance
and known geometry deliberately isolate representation/fusion capacity.
"""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from cache_pairs import image,degrade
from native_vae import load_vae,encode_repeated,decode_last,CANONICAL_SHA
from pixel_dc import Projection
from probe_complementary import cg,FOOTPRINTS


def tensor_bytes(x):return x.numel()*x.element_size()


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--calibration',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=8);p.add_argument('--context',type=int,default=22);p.add_argument('--iterations',type=int,default=48);p.add_argument('--preprocessed-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;start=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'inputs').mkdir(exist_ok=True)
    def live(**fields):
        value=dict(command=' '.join(sys.argv),pid=os.getpid(),unix_time=time.time(),foreground=True,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),**fields);(out/'LIVE_STATE.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
    live(status='native_H3_complementary_audit_loading',step=0,steps=args.count)
    manifest=json.loads(Path(args.manifest).read_text());sources=[x for x in manifest['items'] if x['split']=='val'][:args.count];assert len(sources)==args.count and args.count<=8
    calibration=json.loads(Path(args.calibration).read_text());assert calibration['test_accessed'] is False;ridge=calibration['selected_ridge'];assert ridge==.01
    vae=load_vae(args.vae,args.comfy);torch.cuda.reset_peak_memory_stats()
    # CPU-built camera matrices preserve the exact previous natural-source
    # probe inputs; matrices are tiny/shared, not a decoder Jacobian cache.
    fcpu=Projection(256,4,device='cpu');ncpu=Projection(128,2,device='cpu');k0=fcpu.matrix.cuda().double();u0=fcpu.right.cuda().double();k=ncpu.matrix.cuda().double();u=ncpu.right.cuda().double();pn=u@k
    dc=Projection(256,4,device='cuda');dc.matrix=fcpu.matrix.cuda();dc.right=fcpu.right.cuda()
    def null(x):return x-u0@(k0@x@k0.T)@u0.T
    def crop(x,fp):top,left=fp;return x[...,top:top+128,left:left+128]
    def embed(x,fp):
        top,left=fp;z=torch.zeros((1,3,256,256),device='cuda',dtype=x.dtype);z[...,top:top+128,left:left+128]=x;return z
    def observe(x,fp):return k@crop(x,fp)@k.T
    def adjoint(y,fp):return embed(k.T@y@k,fp)
    def band(x,fp):return embed(pn@crop(x,fp)@pn.T,fp)
    def cband(x,fp):return null(band(null(x),fp))
    def history_band(x,history,count):
        # Same symmetric per-footprint row-band/current-null filter as the
        # decoded-atlas controls; isolates codec/fusion from free D high bands.
        return null(sum((band(null(x),fp) for fp,_ in history),torch.zeros_like(x))/count.clamp_min(1))
    def invert(base,packets):
        b=null(sum((adjoint(y-observe(base,fp),fp) for fp,y in packets),torch.zeros_like(base)))
        def normal(v):return null(sum((adjoint(observe(null(v),fp),fp) for fp,_ in packets),torch.zeros_like(v)))+ridge*v
        h,res=cg(normal,b,args.iterations);return base+null(h),res
    def pixel_atlas(base,packets):
        total=torch.zeros_like(base);count=torch.zeros((1,1,256,256),device='cuda',dtype=torch.float64)
        for fp,y in packets:
            top,left=fp;total+=embed(u@(y-observe(base,fp))@u.T,fp);count[...,top:top+128,left:left+128]+=1
        return base+null(total/count.clamp_min(1))
    timings=[];rows=[];diagnostics=[];packet_manifest=[];native_calls={'encode':0,'decode':0};state_rows=[];current_context='initial'
    def timed(label,fn):
        torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();elapsed=time.perf_counter()-t;timings.append(dict(source=current_source,context=current_context,operation=label,seconds=elapsed));return v,elapsed
    def enc(x):
        native_calls['encode']+=1;return timed('native_encode22',lambda:encode_repeated(vae,x.float().clamp(0,1),args.context))[0]
    def dec(z):
        native_calls['decode']+=1;return timed('native_decode22_lastframe',lambda:decode_last(vae,z))[0].double()
    previous={}
    def record(prov,order,prefix,method,pred,gt,regions,yfar,history,cgres=None):
        target=gt;current_error=float((k0@pred@k0.T-yfar).square().mean().sqrt())
        key=(order,method);prev=previous.get(key)
        for family,rgb in [('raw',pred.clamp(0,1)),('DC',dc.apply(pred.float(),yfar.float()).double())]:
            error=(rgb-target).square();entry=dict(source=prov['id'],order=order,prefix=prefix,method=method,family=family,rgb_psnr=float(-10*torch.log10(error.mean())),current_far_rmse=float((k0@rgb@k0.T-yfar).square().mean().sqrt()),prebox_current_far_rmse=current_error,past_near_rmse=float(torch.stack([(observe(rgb,fp)-y).square().mean() for fp,y in history]).mean().sqrt()),cg_relative_residual=cgres)
            for name,mask in regions.items():
                count=int(mask.sum());entry[name+'_pixels']=count;entry[name+'_psnr']=float(-10*torch.log10(error[mask.expand_as(error)].mean())) if count else None
            old=regions['old_exclusive'];entry['old_exclusive_mse_change_from_previous']=float((error-(prev[family]-target).square())[old.expand_as(error)].mean()) if prev is not None and bool(old.any()) else None
            rows.append(entry)
            if key not in previous:previous[key]={}
            previous[key][family]=rgb.clone()
        return previous[key]['DC']
    with torch.no_grad():
        for number,prov in enumerate(sources):
            current_source=prov['id'];source=Path(prov['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==prov['source_sha256'];gt_cpu=image(source,256,'cpu')[None].double();gt=gt_cpu.cuda();gen=torch.Generator().manual_seed(prov['seed']+3701)
            yf=degrade(gt_cpu.float(),4).double();yf=(yf+torch.randn(yf.shape,generator=gen,dtype=torch.float64)*.003).clamp(0,1);yfar=yf.cuda();up=F.interpolate(yfar,size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1);base=up+u0@(yfar-k0@up@k0.T)@u0.T
            packets=[];kc=ncpu.matrix.double()
            for fp in FOOTPRINTS:
                y=kc@crop(gt_cpu,fp)@kc.T;y=(y+torch.randn(y.shape,generator=gen,dtype=torch.float64)*.003).clamp(0,1);packets.append((fp,y.cuda()))
            # Runtime data contains observations/geometry only; target remains
            # outside this saved input packet and all reconstruction functions.
            inputs=dict(current_far_y=yf.float(),near_y=torch.stack([y.cpu().float() for _,y in packets]),footprints=FOOTPRINTS,packet_ids=[f'near{i}' for i in range(4)])
            path=out/'inputs'/(current_source+'.pt');torch.save(inputs,path);packet_manifest.append(dict(source=current_source,source_sha256=prov['source_sha256'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),HR_in_runtime_inputs=False))
            current_context='current_LR';z0=enc(up);base_h3=dec(z0)
            classical_base_h3=oracle=base_h3
            if not args.preprocessed_only:
                current_context='current_classical_affine_far';classical_base_h3=dec(enc(base))
                current_context='oracle_HR_evaluation_only';oracle=dec(enc(gt)) # separate evaluation-only capacity ceiling
            packet_codes=[];packet_full=[];packet_delta=[]
            for fp,y in packets:
                if args.preprocessed_only:
                    current_context=f'packet_{fp}_single_near_physical_preconditioner';zi=enc(pixel_atlas(base,[(fp,y)]));di=dec(zi);packet_codes.append((zi,zi));packet_full.append(di);packet_delta.append(di);continue
                top,left=fp;view=up.clone();view[...,top:top+128,left:left+128]=F.interpolate(y,size=(128,128),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
                proxy=view.clone();proxy[...,top:top+128,left:left+128]=F.interpolate(degrade(crop(view,fp).float(),4),size=(128,128),mode='bicubic',align_corners=False,antialias=True).double().clamp(0,1)
                current_context=f'packet_{fp}_full';zi=enc(view);packet_full.append(dec(zi))
                current_context=f'packet_{fp}_proxy_delta';zp=enc(proxy);packet_delta.append(dec(z0+zi-zp));packet_codes.append((zi,zp))
            previous={};source_start=time.time();gallery={};order_final={}
            for order,indices in [('forward',list(range(4))),('reverse',list(reversed(range(4))))]:
                history=[];seenmask=torch.zeros((1,1,256,256),device='cuda',dtype=torch.bool);total_full=torch.zeros_like(base);total_delta=torch.zeros_like(base);count=torch.zeros((1,1,256,256),device='cuda',dtype=torch.float64)
                for prefix,index in enumerate(indices,1):
                    fp,y=packets[index];history.append((fp,y));top,left=fp;currentmask=torch.zeros_like(seenmask);currentmask[...,top:top+128,left:left+128]=True
                    regions=dict(union=seenmask|currentmask,old_union=seenmask.clone(),old_exclusive=seenmask&~currentmask,new_only=currentmask&~seenmask,overlap=seenmask&currentmask,current=currentmask);seenmask|=currentmask
                    current_context=f'{order}_prefix{prefix}';classical_atlas,atlas_sec=timed('pixel_atlas_prefix',lambda:pixel_atlas(base,history));classical_inverse=classical_atlas;res=None
                    if not args.preprocessed_only:(classical_inverse,res),inverse_sec=timed('pixel_inverse_prefix',lambda:invert(base,history))
                    classical_last=pixel_atlas(base,history[-1:])
                    full_delta=band(null(packet_full[index]-base_h3),fp);delta_delta=band(null(packet_delta[index]-base_h3),fp);total_full+=full_delta;total_delta+=delta_delta;count[...,top:top+128,left:left+128]+=1
                    e_atlas=e_inverse=base_h3
                    if not args.preprocessed_only:
                        current_context=f'{order}_prefix{prefix}_ED_pixel_atlas';e_atlas=dec(enc(classical_atlas))
                        current_context=f'{order}_prefix{prefix}_ED_pixel_inverse';e_inverse=dec(enc(classical_inverse))
                    methods={'current_far_classical':base,'current_H3_LR':base_h3,'current_H3_classicalfar':classical_base_h3,'pixel_lastnear':classical_last,'pixel_atlas':classical_atlas,'pixel_inverse':classical_inverse,'H3_last_fullcode':base_h3+cband(packet_full[index]-base_h3,fp),'H3_last_deltacode':base_h3+cband(packet_delta[index]-base_h3,fp),'H3_decoded_atlas_full':base_h3+null(total_full/count.clamp_min(1)),'H3_decoded_atlas_delta':base_h3+null(total_delta/count.clamp_min(1)),'H3_ED_pixel_atlas':e_atlas,'H3_ED_pixel_inverse':e_inverse,'H3_ED_pixel_atlas_anchored':base_h3+null(e_atlas-base_h3),'H3_ED_pixel_inverse_anchored':base_h3+null(e_inverse-base_h3),'H3_ED_pixel_atlas_matched_support':base_h3+history_band(e_atlas-base_h3,history,count),'H3_ED_pixel_inverse_matched_support':base_h3+history_band(e_inverse-base_h3,history,count),'oracle_H3_HR':oracle}
                    if args.preprocessed_only:methods={'current_H3_LR':base_h3,'H3_preconditioned_last_fullcode':base_h3+cband(packet_full[index]-base_h3,fp),'H3_preconditioned_last_ED':packet_full[index],'H3_preconditioned_decoded_atlas_full':base_h3+null(total_full/count.clamp_min(1))}
                    for method,pred in methods.items():
                        dc_rgb=record(prov,order,prefix,method,pred,gt,regions,yfar,history,res if method in ['pixel_inverse','H3_ED_pixel_inverse','H3_ED_pixel_inverse_anchored'] else None)
                        if order=='forward' and prefix==4 and number<2:gallery[method]=dc_rgb
                    # These are exact before boxing; generated values are never
                    # appended to the LR packet ledger or treated as evidence.
                    for method in [m for m in methods if m in ['H3_last_fullcode','H3_last_deltacode','H3_decoded_atlas_full','H3_decoded_atlas_delta','H3_ED_pixel_atlas_anchored','H3_ED_pixel_inverse_anchored','H3_ED_pixel_atlas_matched_support','H3_ED_pixel_inverse_matched_support','H3_preconditioned_last_fullcode','H3_preconditioned_decoded_atlas_full']]:
                        diagnostics.append(dict(source=current_source,order=order,prefix=prefix,method=method,prebox_anchor_difference_rms=float((k0@(methods[method]-base_h3)@k0.T).square().mean().sqrt()),future_packets_used=False))
                    state_specs={name:(0,'No past detail state') for name in ['current_far_classical','current_H3_LR','current_H3_classicalfar']}
                    state_specs.update({name:(tensor_bytes(packet_codes[index][0]),'One native code; current anchor is transient') for name in ['H3_last_fullcode','H3_last_deltacode']})
                    state_specs['pixel_lastnear']=(3*64*64*4,'Last actual LR packet; recompute RGB residual on query')
                    state_specs.update({name:(3*256*256*4+256*256*4,'RGB residual sum+single-channel counts, or retain arrived LR packets and recompute deterministic D at higher query cost') for name in ['H3_decoded_atlas_full','H3_decoded_atlas_delta','pixel_atlas','H3_ED_pixel_atlas','H3_ED_pixel_atlas_anchored','H3_ED_pixel_atlas_matched_support']})
                    state_specs.update({name:(prefix*3*64*64*4,'Actual LR ledger; bounded per-fixed-footprint sum/count sufficient statistics for independent repeat noise') for name in ['pixel_inverse','H3_ED_pixel_inverse','H3_ED_pixel_inverse_anchored','H3_ED_pixel_inverse_matched_support']})
                    state_specs.update({name:(tensor_bytes(packet_codes[index][0]),'One full native code after single-packet classical preconditioning') for name in ['H3_preconditioned_last_fullcode','H3_preconditioned_last_ED']});state_specs['H3_preconditioned_decoded_atlas_full']=(3*256*256*4+256*256*4,'Decoded preconditioned RGB sum+count, no new learning')
                    for method,(state,note) in state_specs.items():
                        if method not in methods:continue
                        nativecode=method in ['H3_last_fullcode','H3_last_deltacode','H3_preconditioned_last_fullcode','H3_preconditioned_last_ED'];state_rows.append(dict(source=current_source,order=order,prefix=prefix,method=method,float32_method_state_payload_bytes=state,audit_operator_dtype_state_payload_bytes=state if nativecode else 2*state,audit_operator_state_dtype='float32' if nativecode else 'float64',common_current_LR_payload_float32_bytes=tensor_bytes(yfar.float()),audit_current_LR_float64_bytes=tensor_bytes(yfar),geometry_packetID_payload_bytes=prefix*16,native_current_latent_transient_bytes=tensor_bytes(z0),float32_LR_ledger_recompute_alternative_bytes=prefix*3*64*64*4,notes=note+'; Python container overhead and audit memoization excluded here, full actual GPU peak separately recorded'))
                    if prefix==4:order_final[order]={name:value.detach().clone() for name,value in methods.items() if not name.startswith('oracle')}
                    write_csv(out/'rgb_metrics.csv',rows);write_csv(out/'state_cost.csv',state_rows);write_csv(out/'runtime_operations.csv',timings);(out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));live(status='native_H3_complementary_capacity_audit_running',step=number,steps=len(sources),source=current_source,order=order,prefix=prefix,rows=len(rows),seconds=time.time()-start)
            commutator={method:float((order_final['forward'][method]-order_final['reverse'][method]).square().mean().sqrt()) for method in order_final['forward']};(out/(current_source+'_order_difference.json')).write_text(json.dumps(commutator,indent=2))
            if gallery:
                entries=[('GT',gt)]+list(gallery.items());canvas=Image.new('RGB',(4*256,((len(entries)+3)//4)*280),'white');draw=ImageDraw.Draw(canvas)
                for i,(name,rgb) in enumerate(entries):canvas.paste(Image.fromarray((rgb[0].clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype('uint8')),((i%4)*256,(i//4)*280));draw.text(((i%4)*256+3,(i//4)*280+258),name,fill='black')
                canvas.save(out/(current_source+'_final_DC.png'))
            live(status='native_H3_complementary_source_complete',step=number+1,steps=len(sources),source=current_source,seconds=time.time()-start)
    config=dict(**vars(args),scope='Frozen actual H3 VAE E/D capacity only; neutral photometry/no occlusion, exact known geometry, natural source photos under simulated camera, no training/test/DiT or actual video claim',sources=[{k:v for k,v in x.items() if k in ['id','source','source_sha256']} for x in sources],ridge=ridge,ridge_selection=calibration['selection'],prefix_protocol='Same 4 measured near footprints and two orders as CPU probe; far return reuses fixed actual far packet to isolate representation. Cache computation is stateless; only arrived prefix packet indices enter each readout.',copied_context='22 copies supply native temporal context only, not22 measurements',method_contract='H3 full/delta nativeD followed by each measured footprint row-band and current-far null; decoded atlas precision-average of such near residuals. ED pixel baselines encode/decode LR-only classical reconstruction; anchored variants remove only nativeD coarse drift. All share actual current-far box/affine DC64.',oracle='oracle_H3_HR uses HR only for a separately named representation ceiling, never state or valid-method inputs',precision=dict(native_E_D='float16',camera_math='float64',TF32=False,DC='float32'),vae_sha256=CANONICAL_SHA,native_calls=native_calls,native_timings_seconds={op:sum(t['seconds'] for t in timings if t['operation']==op) for op in sorted({t['operation'] for t in timings})},peak_GPU_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start,rows=len(rows),state_cost='state_cost.csv includes actual float32 persistent representation, common current LR and metadata; deterministic decoder-query memoization in audit is evaluation scratch, not falsely claimed runtime state. Inverse also has bounded per-footprint sum/count sufficient statistics, not inherently growing with duplicate/noisy packets.',shared_camera_matrix_bytes=sum(tensor_bytes(x) for x in [k0,u0,k,u,pn,dc.matrix,dc.right]),packets=packet_manifest,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'complete.json').write_text(json.dumps(config,indent=2));live(status='native_H3_complementary_capacity_audit_complete',step=len(sources),steps=len(sources),seconds=config['seconds'],rows=len(rows),peak_GPU_bytes=config['peak_GPU_bytes'])


if __name__=='__main__':main()
