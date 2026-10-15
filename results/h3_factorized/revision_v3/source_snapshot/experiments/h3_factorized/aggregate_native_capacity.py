"""Merge fixed native capacity audits after matching single-near preprocessing."""
import argparse,csv,json,shutil
from pathlib import Path


def read(path):return list(csv.DictReader(path.open()))
def save(path,rows):
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--matched',required=True);p.add_argument('--preconditioned',required=True);p.add_argument('--out',required=True);args=p.parse_args();a,b,out=map(Path,[args.matched,args.preconditioned,args.out]);out.mkdir(parents=True,exist_ok=True)
    ar=read(a/'rgb_metrics.csv');br=read(b/'rgb_metrics.csv');base={(x['source'],x['order'],x['prefix'],x['family']):float(x['union_psnr']) for x in ar if x['method']=='current_H3_LR'};err=max(abs(float(x['union_psnr'])-base[(x['source'],x['order'],x['prefix'],x['family'])]) for x in br if x['method']=='current_H3_LR');assert err<1e-6,('Changed native current baseline',err)
    # For prefix1 the exact same LR-only single-near preconditioner must equal
    # batch-atlas native E/D under the same support readout, source by source.
    first={(x['source'],x['order'],x['family']):float(x['union_psnr']) for x in ar if x['method']=='H3_ED_pixel_atlas_matched_support' and x['prefix']=='1'};firsterr=max(abs(float(x['union_psnr'])-first[(x['source'],x['order'],x['family'])]) for x in br if x['method']=='H3_preconditioned_last_fullcode' and x['prefix']=='1');assert firsterr<1e-6,('Single-packet preconditioner mismatch',firsterr)
    save(out/'rgb_metrics.csv',ar+[x for x in br if x['method']!='current_H3_LR'])
    save(out/'runtime_operations.csv',read(a/'runtime_operations.csv')+read(b/'runtime_operations.csv'))
    save(out/'state_cost.csv',read(a/'state_cost.csv')+[x for x in read(b/'state_cost.csv') if x['method']!='current_H3_LR'])
    diag=json.loads((a/'diagnostics.json').read_text())+json.loads((b/'diagnostics.json').read_text());(out/'diagnostics.json').write_text(json.dumps(diag,indent=2))
    ca=json.loads((a/'complete.json').read_text());cb=json.loads((b/'complete.json').read_text());meta=dict(ca);meta.update(component_runs=[dict(path=str(a),seconds=ca['seconds'],script_sha256=ca['script_sha256'],native_calls=ca['native_calls']),dict(path=str(b),seconds=cb['seconds'],script_sha256=cb['script_sha256'],native_calls=cb['native_calls'])],seconds=ca['seconds']+cb['seconds'],native_calls={k:ca['native_calls'][k]+cb['native_calls'][k] for k in ['encode','decode']},native_timings_seconds={k:ca['native_timings_seconds'].get(k,0)+cb['native_timings_seconds'].get(k,0) for k in set(ca['native_timings_seconds'])|set(cb['native_timings_seconds'])},peak_GPU_bytes=max(ca['peak_GPU_bytes'],cb['peak_GPU_bytes']),rows=len(ar)+sum(x['method']!='current_H3_LR' for x in br),current_baseline_max_delta_dB=err,prefix1_matched_preconditioner_max_delta_dB=firsterr,state_cost='Explicit float32 deployment payload and actual float64 camera/atlas audit payload; full measured CUDA peak includes evaluation memoization. Fixed-footprint LR sum/count sufficient-stat baseline retained.',initial_unmatched_audit_seconds=52.15317130088806,total_physical_native_audit_seconds=52.15317130088806+ca['seconds']+cb['seconds'],fairness_correction='Single-near physical preconditioning explains the apparent ~1.1dB extra fusion/codec gap; retain strong preconditioned last-near and decoded-atlas controls.')
    (out/'complete.json').write_text(json.dumps(meta,indent=2))
    for file in b.glob('*final_DC.png'):shutil.copy2(file,out/(file.stem+'_preconditioned.png'))
    print(json.dumps(dict(rows=meta['rows'],native_calls=meta['native_calls'],same_baseline_max_delta=err,matched_prefix1_max_delta=firsterr)))


if __name__=='__main__':main()
