"""Aggregate fixed-run evidence without selecting models or seeds."""
import argparse
import json
from pathlib import Path
import numpy as np


def aggregate(records):
    grouped={}
    for seed,conditions in records.items():
        for condition,models in conditions.items():
            for model,metrics in models.items():
                grouped.setdefault(condition,{}).setdefault(model,[]).append(metrics)
    result={}
    for condition,models in grouped.items():
        result[condition]={}
        for model,values in models.items():
            fields={k:{'mean':float(np.mean([v[k] for v in values])),
                       'std_seed':float(np.std([v[k] for v in values],ddof=1)) if len(values)>1 else 0.}
                    for k in values[0] if not isinstance(values[0][k],list)}
            result[condition][model]=fields
        if 'map_transport' in models and 'gaussian_all_history_map' in models:
            a=np.array([v['sample_mse_final'] for v in models['map_transport']])
            b=np.array([v['sample_mse_final'] for v in models['gaussian_all_history_map']])
            per_image=(-10*np.log10(a)+10*np.log10(b)).mean(0)
            rng=np.random.default_rng(909)
            boots=per_image[rng.integers(0,len(per_image),size=(2000,len(per_image)))].mean(1)
            result[condition]['paired_transport_minus_gaussian_db']={'mean':float(per_image.mean()),
                'bootstrap_image_ci95':np.quantile(boots,[.025,.975]).tolist(),
                'source_images':len(per_image)}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/learned_core');args=p.parse_args();root=Path(args.root)
    raw=json.loads((root/'main/all_results.json').read_text())
    main_records={seed:v['evaluation'] for seed,v in raw.items()}
    result={'seeds':list(raw),'total_optimizer_steps':sum(v['train_steps']*len(v['parameters']) for v in raw.values()),
            'train_seconds':sum(v['train_seconds'] for v in raw.values()),'parameters':next(iter(raw.values()))['parameters'],
            'main':aggregate(main_records),
            'label_corrections':{'texture_ood/new':'Independent draws from the same sinusoidal texture generator as training; this is synthetic ID, NOT OOD.'}}
    extended_path=root/'extended/all_results.json'
    if extended_path.exists():result['extended']=aggregate(json.loads(extended_path.read_text()))
    calibrated_path=root/'calibrated_new/all_results.json'
    if calibrated_path.exists():result['calibrated_new']=aggregate(json.loads(calibrated_path.read_text()))
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Learned core: fixed-budget evidence','',
           'Protocol: grayscale 4× known canonical point footprints, BSDS200 train / 100 validation / 200 test, one 48×48 held-out patch per source image. This is a mechanism experiment, not general video SR.',
           '',f"Three training seeds {list(raw)}, six models, {result['total_optimizer_steps']} actual optimizer steps; total training time {result['train_seconds']:.1f}s on one H100. Each run uses 1000 steps × batch 24; the Gaussian prior sees training images only.",
           '', 'Test final PSNR, mean ± standard deviation over training seeds:',
           '', '| condition | Gaussian MAP | stateless learned MAP residual | persistent learned MAP residual | raw GRU |',
           '|---|---:|---:|---:|---:|']
    for condition in ['test/repeat','test/mixed_repeat','test/new','test/reveal','test/wrong']:
        parts=[]
        for model in ['gaussian_all_history_map','map_residual','map_transport','gru']:
            x=result['main'][condition][model]['psnr_final_db'];parts.append(f"{x['mean']:.3f} ± {x['std_seed']:.3f}")
        lines.append('| '+condition+' | '+' | '.join(parts)+' |')
    lines += ['', 'Interpretation:', '',
              '- Learned non-Gaussian detail improves strongly under one or four observed phases. This reproduces in all three seeds.',
              '- At eight observed phases, the fitted all-history Gaussian MAP has higher PSNR. No superiority claim over this strong baseline is supported there.',
              '- Persistent learned residual is more accurate than stateless residual in some conditions but its independent benefit is not stable across seeds.',
              '- Exact packet duplicates must not increase precision. Independent noisy packets at the same footprint must increase precision; the extended evaluation passes explicit fresh-packet masks.',
              '- All-history ledger and nullspace projection retain a complete reveal and prevent future LR-only frames from rewriting observed detail. They do not solve correspondence errors.',
              '- Main texture_ood labels are corrected to synthetic ID; extended checker_ood uses discontinuous multiplicative patterns absent from training.',
              '', 'Actionable direction for a large H3 adapter: preserve a persistent detail state in weak-evidence directions, project new detail away from constrained directions, and estimate residual risk before injecting it when evidence is sufficient. A validation-only risk control is evaluated separately and is not presented as a new trained architecture.']
    if 'extended' in result:
        lines += ['', 'Additional fixed-checkpoint tests (no extra training):', '',
                  '| condition | Gaussian MAP | learned transport | stateless residual |', '|---|---:|---:|---:|']
        for condition in ['independent_noise_repeat','high_noise_new','checker_ood','reveal_then_repeat']:
            values=[]
            for model in ['gaussian_all_history_map','map_transport','map_residual']:
                x=result['extended'][condition][model]['psnr_final_db'];values.append(f"{x['mean']:.3f} ± {x['std_seed']:.3f}")
            lines.append('| '+condition+' | '+' | '.join(values)+' |')
        projected=result['extended']['reveal_then_repeat']['map_transport']['psnr_final_db']['mean']
        unprojected=result['extended']['reveal_then_repeat']['map_transport/no_projection']['psnr_final_db']['mean']
        lines += ['',f'Removing the observed-direction projection after a complete reveal reduces PSNR from {projected:.3f} to {unprojected:.3f} dB. Freezing the prior after the first frame also hurts novel-evidence uptake. These are mechanism ablations of the fixed model, not independently optimized alternative architectures.',
                  '', 'The raw ledger CNN fails the unseen precision-count regime (independent same-footprint noise packets). A precision scalar presented out of its training support is insufficient. The analytic posterior plus persistent residual remains usable; this motivates an actual spectral information transition for H3, rather than merely feeding counts to a CNN.']
    if 'calibrated_new' in result:
        condition=result['calibrated_new']['eight_new']
        base=condition['gaussian_all_history_map']['psnr_final_db']['mean']
        calibrated=condition['val_risk_calibrated_transport']['psnr_final_db']['mean']
        raw_value=condition['map_transport']['psnr_final_db']['mean']
        lines += ['', 'Exploratory risk control:', '',
                  'Use R(a)-R(0)=2a E[e·d]+a²E[d²], with e the MAP error and d the proposed residual, estimated on validation images only. Apply a*=clip(-E[e·d]/E[d²],0,1). No test target enters this coefficient.',
                  '',f'For a fresh held-out acquisition schedule at eight phases, raw transport is {raw_value:.3f} dB, Gaussian MAP {base:.3f} dB, and the validation-calibrated control {calibrated:.3f} dB. This follows the initial experiment and uses the same test source images, so it is an exploratory diagnostic rather than a preregistered confirmation. The H3 core must learn an evidence-conditioned transition instead of treating this scalar postprocessing as the final method.']
    (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'summary':str(root/'summary.json'),'report':str(root/'FINDINGS.md'),'optimizer_steps':result['total_optimizer_steps']}))


if __name__=='__main__':main()
