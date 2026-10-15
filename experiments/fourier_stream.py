"""Natural-image global-operator experiment; not a real-motion video benchmark."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fourier_ledger import FourierLedger


def crop(path, n):
    x = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255
    h, w = x.shape; return x[(h-n)//2:(h+n)//2, (w-n)//2:(w+n)//2].copy()


def psnr(x, y):
    # Evaluate the central area so periodic simulation boundaries are excluded.
    sl = np.s_[8:-8, 8:-8]
    return float(-10*np.log10(np.mean((x[sl]-y[sl])**2)+1e-12))


def sqrt_psd(a):
    w, v = np.linalg.eigh(a)
    return (v * np.sqrt(np.maximum(w, 1e-12))[..., None, :]) @ v.swapaxes(-1, -2).conj()


def main(args):
    t0 = time.time(); out = ROOT / "results" / "fourier_stream"; out.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data" / "BSR" / "BSDS500" / "data" / "images"
    train = sorted((data / "train").glob("*.jpg")); test = sorted((data / "test").glob("*.jpg"))[:args.images]
    assert len(train) == 200 and len(test) == args.images
    spectra = [np.abs(np.fft.fft2(crop(p,args.size)-0.5,norm="ortho"))**2 for p in train]
    psd = np.mean(spectra,axis=0) + 1e-5
    precision = args.noise**2 / psd
    mean_prior = np.full((args.size,args.size), 0.5)
    phases = [(i//4,i%4) for i in [0,10,5,15,2,8,7,13,1,4,11,14,3,12,6,9]]
    rows = []; panels = []; max_imag = 0.; readonly = True
    protocols = ["complementary", "repeat_packet", "independent_repeat", "revisit", "kernel_mismatch", "scene_change"]
    for image_index, path in enumerate(test):
        x0 = crop(path,args.size)
        x1 = crop(test[(image_index+1)%len(test)],args.size)
        for pi,protocol in enumerate(protocols):
            rng = np.random.default_rng(10000 + image_index*100+pi)
            ledger = FourierLedger(x0.shape,4,mean_prior,precision)
            history = []; initial = None; previous = {}; z = ledger.pack(np.fft.fft2(rng.normal(size=x0.shape), norm="ortho"))
            old_mean = mean_prior.copy(); old_sample = mean_prior + np.fft.ifft2(np.fft.fft2(rng.normal(size=x0.shape),norm="ortho") * np.sqrt(psd),norm="ortho").real
            old_cov = np.eye(16)[None,None,:,:] / ledger.j0[...,None,:]
            first_y = None
            for ti in range(16):
                truth = x1 if protocol == "scene_change" and ti >= 8 else x0
                shift = (0,0) if "repeat" in protocol else phases[ti]
                actual_blur = 0.85 if protocol == "kernel_mismatch" else 0.6
                y = ledger.observe(truth,shift,actual_blur) + rng.normal(0,args.noise,(args.size//4,args.size//4))
                if first_y is None: first_y = y.copy()
                if protocol == "repeat_packet": y = first_y.copy()
                visible = not(protocol == "revisit" and 4 <= ti < 12)
                if visible:
                    packet_id = 0 if protocol == "repeat_packet" else ti
                    ledger.add(y,shift,0.6,observation_id=packet_id)
                    # Repeated arrival may refresh the finite window without
                    # being counted twice as independent evidence inside it.
                    history.append((ti,y.copy(),shift,packet_id))
                last = history[-1]
                current = FourierLedger(x0.shape,4,mean_prior,precision); current.add(last[1],last[2],0.6)
                short = FourierLedger(x0.shape,4,mean_prior,precision)
                for ht,hy,hs,hid in history:
                    if ht >= ti-3: short.add(hy,hs,0.6,observation_id=hid)
                mean = ledger.solve(); cov = np.linalg.inv(ledger.system())
                c_sqrt = sqrt_psd(cov)
                fixed = mean + args.noise*np.fft.ifft2(ledger.unpack((c_sqrt @ z[...,None])[...,0]),norm="ortho").real
                fresh_z = ledger.pack(np.fft.fft2(rng.normal(size=x0.shape),norm="ortho"))
                fresh = mean + args.noise*np.fft.ifft2(ledger.unpack((c_sqrt @ fresh_z[...,None])[...,0]),norm="ortho").real
                bures,old_mean,old_cov = ledger.posterior_transport(old_sample,old_mean,old_cov)
                old_sample = bures
                if initial is None: initial = fixed.copy()
                outputs = {"current_gaussian_map":current.solve(),"window4_gaussian_map":short.solve(),"all_history_gaussian_map":mean,"fixed_reference_sample":initial,"fixed_noise_posterior":fixed,"fresh_noise_posterior":fresh,"bures_posterior_transport":bures}
                for method,pred in outputs.items():
                    drift = float(np.mean((pred-previous[method])**2)) if method in previous else 0.
                    rows.append({"image":path.stem,"protocol":protocol,"frame":ti,"method":method,"psnr":psnr(pred,truth),"temporal_mse":drift,"measurement_mse":float(np.mean((ledger.observe(pred,shift,0.6)-y)**2)) if visible else float("nan"),"observations":ledger.count,"state_bytes":ledger.nbytes})
                    previous[method] = pred.copy()
                # Query scale/ROI operations must never mutate the evidence state.
                before = hashlib.sha256(ledger.gram.tobytes()+ledger.rhs.tobytes()).hexdigest()
                im=Image.fromarray(np.uint8(np.clip(mean,0,1)*255))
                for scale in [2,4,8,2]: im.resize((args.size*scale//4,)*2,Image.Resampling.LANCZOS)
                readonly &= before == hashlib.sha256(ledger.gram.tobytes()+ledger.rhs.tobytes()).hexdigest()
                if image_index == 0 and protocol == "complementary" and ti in [0,3,7,15]:
                    panels.append(np.concatenate([truth,outputs["current_gaussian_map"],outputs["window4_gaussian_map"],mean,fixed,bures],axis=1))
            print(f"image={image_index+1}/{len(test)} protocol={protocol} done",flush=True)
    with (out/"per_frame.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for protocol in protocols:
        for method in sorted({r['method'] for r in rows}):
            rs=[r for r in rows if r['protocol']==protocol and r['method']==method]
            final=[r['psnr'] for r in rs if r['frame']==15]
            summary.append({"protocol":protocol,"method":method,"final_psnr":float(np.mean(final)),"mean_temporal_mse":float(np.mean([r['temporal_mse'] for r in rs if r['frame']>0]))})
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    manifest={"train_images":len(train),"test_images":len(test),"test_ids":[p.stem for p in test],"size":args.size,"noise":args.noise,"periodic_boundary":True,"oracle_global_translation":True,"hr_only_for_simulation_prior_fitting_and_evaluation":True,"query_readonly":bool(readonly),"seconds":time.time()-t0,"protocols":protocols,"row_count":len(rows),"note":"Classical Gaussian baseline and posterior couplings; not a trained novel method or real-motion video experiment."}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    if panels: Image.fromarray(np.uint8(np.clip(np.concatenate(panels,axis=0),0,1)*255)).resize((args.size*6*2,args.size*4*2)).save(out/"comparison.png")
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--images",type=int,default=24);p.add_argument("--size",type=int,default=64);p.add_argument("--noise",type=float,default=0.01);main(p.parse_args())
