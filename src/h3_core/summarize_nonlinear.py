"""Summarize completed fixed runs without selecting models or hyperparameters."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);args=p.parse_args();root=Path(args.suite)
    rows=[];runs={}
    for path in sorted(root.glob('*/results.json')):
        result=json.loads(path.read_text());name=path.parent.name;runs[name]=result
        for source,methods in result['evaluation'].items():
            for method,metrics in methods.items():
                for audit in metrics['audits']:
                    rows.append({'run':name,'basis':result['config']['basis_mode'],'seed':result['config']['seed'],'source':source,'split':source.split('/')[0],'protocol':result['source_protocols'][source],'method':method,**audit,'rgb_revision_17_38':metrics['actual_canonical_rgb_drift_17_38'],'code_drift_occluded24_30':metrics['code_drift_occluded24_30']})
    if not rows:raise SystemExit('No evaluation rows yet')
    with (root/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'runs':{k:{'config':r['config'],'seconds':r.get('seconds'),'evaluated_sources':len(r['evaluation']),'parameters':r['parameters']} for k,r in runs.items()},'means':{},'paired_test_effects':{},'limits':['These runs are frozen-H3-VAE mechanism experiments, not H3 DiT restoration.','39 frames is a short controlled static-natural camera, with oracle geometry/visibility.','RGB revision measures state changes during new evidence and is not by itself a stability score.','HR-code transfer diagnostics are excluded from ordinary method comparisons and are not RGB upper bounds.']}
    for basis in sorted({v['basis'] for v in rows}):
        for split in ['val','test']:
            for method in sorted({v['method'] for v in rows}):
                selected=[v for v in rows if v['basis']==basis and v['split']==split and v['method']==method and v['frame']==38]
                if not selected:continue
                summary['means'][f'{basis}/{split}/{method}']={metric:float(np.mean([v[metric] for v in selected])) for metric in ['psnr_full_db','psnr_nominal_roi_db','psnr_teacher90_support_db','gradient_l1_full','lr_witness_rmse','rgb_revision_17_38','code_drift_occluded24_30']}
    generator=np.random.default_rng(1729)
    for basis in sorted({v['basis'] for v in rows}):
        source_values={}
        for v in rows:
            if v['basis']==basis and v['split']=='test' and v['frame']==38:
                source_values.setdefault(v['source'],{}).setdefault(v['method'],[]).append(v['psnr_nominal_roi_db'])
        for control in ['map','stateless','gru','current_only','anchor_only']:
            differences=[np.mean(x['transport'])-np.mean(x[control]) for x in source_values.values() if 'transport' in x and control in x]
            if not differences:continue
            delta=np.asarray(differences);sample=generator.choice(delta,(5000,len(delta)),replace=True).mean(1)
            summary['paired_test_effects'][f'{basis}/transport-minus-{control}/nominal_roi_db']={'mean':float(delta.mean()),'image_bootstrap95':np.quantile(sample,[.025,.975]).tolist(),'independent_sources':len(delta),'seeds_averaged_per_source':len({v['seed'] for v in rows if v['basis']==basis})}
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
