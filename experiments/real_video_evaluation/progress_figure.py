"""Source-level evidence overview from completed result JSON, without fitting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/progress_overview'


def main():
    OUT.mkdir(exist_ok=True)
    names = ['book-20', 'hat-3', 'book-19', 'hat-1']
    sources = [json.loads((ROOT / f'results/long_video_vae/{n}/evaluation.json').read_text()) for n in names]
    pilot = json.loads((ROOT / 'results/h3_setup/native_sr_pilot400_native_io_v2/evaluation/aggregate.json').read_text())['aggregate']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    fig.subplots_adjust(top=.86, bottom=.17, hspace=.48, wspace=.30)
    fig.suptitle('H3 experiments: foundation gains; adaptation shows a quality tradeoff',
                 x=.06, ha='left', y=.97, fontsize=16, fontweight='bold')
    fig.text(.06, .92, 'All displayed methods use the same output observation-consistency projection (DC25).', color='#444')
    payload = {'long_source_deltas': [], 'pilot': pilot, 'pilot_version': 'official_native_io_v2'}
    for ax, metric, title in zip(axes[0], ['psnr', 'lpips'],
                                  ['Long streams: PSNR gain (higher is better)', 'Long streams: LPIPS change (lower is better)']):
        delta = [s['methods']['h3_vae_dc25'][metric] - s['methods']['bicubic_dc25'][metric] for s in sources]
        ax.bar(np.arange(4), delta, color='#246c9c', width=.58)
        ax.axhline(0, color='#444', linewidth=.8)
        ax.set_xticks(np.arange(4), [f"{s['sequence']}\n{s['split']}" for s in sources])
        ax.set_title(title, loc='left', fontsize=12)
        ax.set_ylabel('H3 VAE minus bicubic' + (' (dB)' if metric == 'psnr' else ''))
        span = max(abs(v) for v in delta)
        ax.set_ylim((0, span * 1.28) if metric == 'psnr' else (-span * 1.28, .003))
        for i, v in enumerate(delta):
            ax.text(i, v + (span*.045 if metric == 'psnr' else -span*.045),
                    f'{v:+.3f}', ha='center', va='bottom' if metric == 'psnr' else 'top')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        payload['long_source_deltas'].append({'metric': metric, 'names': names, 'deltas': delta})
    styles = {
        'bicubic': ('Bicubic', '#999999', 's'), 'vae_only': ('VAE', '#246c9c', 'D'),
        'unadapted': ('H3 base', '#222222', 'o'), 'lora_only': ('LoRA400', '#d79b25', '^'),
        'guide_only': ('Guide400', '#a54b72', 'v'), 'joint': ('Joint400', '#b64e37', 'P')}
    offsets = {
        '256': {'bicubic': (5, 4), 'vae_only': (6, 3), 'unadapted': (2, -18),
                'lora_only': (8, 10), 'guide_only': (-50, 10), 'joint': (-35, -20)},
        'native480': {'bicubic': (5, 4), 'vae_only': (8, 3), 'unadapted': (8, -1),
                      'lora_only': (8, -8), 'guide_only': (8, 4), 'joint': (8, -8)}}
    for ax, resolution in zip(axes[1], ['256', 'native480']):
        rows = {r['method']: r for r in pilot if r['resolution'] == resolution}
        for key, (label, color, marker) in styles.items():
            r = rows[key + '_dc25']; x = r['mean_frame_psnr_db']; y = r['mean_lpips_alex_v01']
            ax.scatter(x, y, color=color, marker=marker, s=65, zorder=3)
            ax.annotate(label, (x, y), xytext=offsets[resolution][key], textcoords='offset points',
                        fontsize=9, color=color)
        ax.set_title(f"Native SR pilot: {'256 ROI' if resolution == '256' else '480-high full view'}", loc='left', fontsize=12)
        ax.set_xlabel('PSNR (dB)  → better'); ax.set_ylabel('LPIPS  ← lower is better')
        ax.margins(x=.25, y=.30); ax.grid(alpha=.2)
    fig.text(.06, .045,
             'Top: 4 independent sources × 600 frames; frozen VAE, no persistent memory.\n'
             'Bottom: v2 pilot, 4 training / 2 validation sources, 400 updates per control; training IO matches the official BF16 inference wrapper.\n'
             'Spatial views differ; not a resolution ablation. This pilot does not establish an independent learned-memory benefit.',
             fontsize=9, color='#444')
    fig.savefig(OUT / 'evidence_overview.png', dpi=180)
    fig.savefig(OUT / 'evidence_overview.pdf')
    (OUT / 'plot_data.json').write_text(json.dumps(payload, indent=2))
    print(OUT)


def sampling_dc_sources():
    data = json.loads((ROOT/'results/h3_setup/sampling_dc_davis8/sourcewise_metrics.json').read_text())
    rows = data['case_deltas']
    names = ['bear','bus','car-turn','flamingo','cows','dance-twirl','bmx-trees','soapbox']
    labels = [n+'\n'+('train' if i<4 else 'val' if i<6 else 'project-test') for i,n in enumerate(names)]
    fig, axes = plt.subplots(2,1,figsize=(11,7.2),layout='constrained')
    for ax,metric,title in zip(axes,['dPSNR_vs_vae_only_dc25','dLPIPS_vs_vae_only_dc25'],
                               ['PSNR gain (higher is better)','LPIPS change (lower is better)']):
        for res,offset,color in [('256',-.18,'#bd8651'),('native480',.18,'#237fa0')]:
            values = [next(r[metric] for r in rows if r['sequence']==n and r['resolution']==res) for n in names]
            ax.bar(np.arange(8)+offset,values,width=.34,label=res,color=color)
        ax.axhline(0,color='0.3',lw=.9); ax.set_xticks(np.arange(8),labels,fontsize=9)
        ax.set_title(title,loc='left',fontsize=12); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(frameon=False,ncol=2)
    fig.suptitle('Frozen H3: sampling-time observation constraints versus matched VAE\n8 independent videos, same final DC25; no learned memory or parameter selection',fontsize=13)
    fig.savefig(OUT/'sampling_dc_source_paired.png',dpi=180)
    fig.savefig(OUT/'sampling_dc_source_paired.pdf')
    (OUT/'sampling_dc_plot_data.json').write_text(json.dumps({'case_deltas':rows,'sources':names,
        'note':'Different fields of view, not a pure resolution ablation. Project-test already used by other project baselines.'},indent=2))
    plt.close(fig)


if __name__ == '__main__':
    main()
    sampling_dc_sources()
