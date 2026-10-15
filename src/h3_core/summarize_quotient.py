"""Summarize complete projected validation runs; exclude invalid current control."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);args=p.parse_args();root=Path(args.suite);rows=[];runs={}
    for folder in sorted(root.glob('seed*')):
        corrected=folder/'results_corrected_controls.json';path=corrected if corrected.exists() else folder/'results.json'
        if not path.exists():continue
        data=json.loads(path.read_text())
        if 'seconds' not in data:continue
        runs[folder.name]={'seed':data['config']['seed'],'seconds':data['seconds'],'current_control_corrected':corrected.exists(),'head_parameters':data['head_parameters'],'prior_parameters':data['prior_parameters']}
        for source,methods in data['evaluation'].items():
            for method,metrics in methods.items():
                if method=='current_only' and not corrected.exists():continue
                for audit in metrics['audits']:
                    for readout in ['affine','bounded']:
                        rows.append({'run':folder.name,'seed':data['config']['seed'],'source':source,'method':method,'frame':audit['frame'],'readout':readout,**audit[readout],'initialfootprint_lr_rmse':audit['initialfootprint_lr_rmse'],'past_witness_clean_rmse':audit['past_witness_clean_rmse'],'past_witness_raw_rmse':audit['past_witness_raw_rmse'],'rgb_revision_17_38':metrics['rgb_revision_17_38']})
    if not rows:raise SystemExit('No completed projected runs')
    with (root/'metrics_corrected.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'runs':runs,'scope':'Validation-only structural development; NOT a fresh heldouttest. Initialcurrentcontrolinvalidandexcludeduntilrepairartifactexists.','means':{},'paired_effects':{}}
    for method in sorted({r['method'] for r in rows}):
        selected=[r for r in rows if r['method']==method and r['frame']==38 and r['readout']=='bounded']
        result['means'][method]={key:float(np.mean([r[key] for r in selected])) for key in ['psnr_full_db','psnr_nominal_roi_db','psnr_teacher90_support_db','gradient_l1_full','initialfootprint_lr_rmse','past_witness_clean_rmse','past_witness_raw_rmse','rgb_revision_17_38']}
    generator=np.random.default_rng(1729);values={}
    for row in rows:
        if row['frame']==38 and row['readout']=='bounded':values.setdefault(row['source'],{}).setdefault(row['method'],[]).append(row['psnr_nominal_roi_db'])
    for control in ['map','stateless','gru','current_only','anchor_only','fixed_prior']:
        differences=[np.mean(v['transport'])-np.mean(v[control]) for v in values.values() if control in v]
        if differences:
            difference=np.array(differences);sample=generator.choice(difference,(5000,len(difference)),replace=True).mean(1)
            result['paired_effects'][f'transport-minus-{control}/bounded_roi_db']={'mean':float(difference.mean()),'validation_image_bootstrap95':np.quantile(sample,[.025,.975]).tolist(),'sources':len(difference),'note':'Development validation, not fresh confirmation'}
    (root/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
