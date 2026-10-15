"""Source is the sampling unit; epochs and training seeds are not videos."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);args=p.parse_args();root=Path(args.results);families={'raw':root,'DC':root/'projected'};groups={'return':['return_A0'],'lighting':['appearance_A1','appearance_B2'],'true_replacement':['content_replace_B0']};controls=['factorized_no_swap','noninvertible_swap','mixed_memory','H3_current_LR'];rng=np.random.default_rng(19417);results=[];csvrows=[]
    for family,directory in families.items():
        rows=list(csv.DictReader((directory/'rgb_metrics.csv').open()));lookup={(r['method'],int(r['seed']),r['source'],r['scenario']):float(r['rgb_psnr']) for r in rows};seeds=sorted({int(r['seed']) for r in rows if int(r['seed'])>=0});sources=sorted({r['source'] for r in rows})
        assert len(sources)==16 and len(seeds)==3
        for group,scenarios in groups.items():
            for control in controls:
                matrix=np.array([[np.mean([lookup[('factorized_swap',seed,source,scenario)]-lookup[(control,-1 if control=='H3_current_LR' else seed,source,scenario)] for scenario in scenarios]) for source in sources] for seed in seeds])
                # First reduce event repetitions within each source/checkpoint,
                # then average training seeds within that same source. The
                # confidence interval resamples16 source blocks, not48 seeds.
                source_delta=matrix.mean(0);draws=[]
                for _ in range(5000):
                    starts=rng.integers(len(sources),size=(len(sources)+1)//2);ii=((starts[:,None]+np.arange(2))%len(sources)).ravel()[:len(sources)];draws.append(source_delta[ii].mean())
                lo,hi=np.percentile(draws,[2.5,97.5]);result=dict(family=family,group=group,comparison='factorized_swap minus '+control,n_sources=len(sources),n_training_seeds=len(seeds),mean_difference_dB=float(source_delta.mean()),source_wins=int((source_delta>0).sum()),source_win_rate=float((source_delta>0).mean()),source_bootstrap_CI95=[float(lo),float(hi)],per_seed={str(seed):dict(mean_difference_dB=float(matrix[i].mean()),source_wins=int((matrix[i]>0).sum()),n_sources=len(sources)) for i,seed in enumerate(seeds)});results.append(result)
                for j,source in enumerate(sources):csvrows.append(dict(family=family,group=group,control=control,source=source,mean_three_seed_delta_dB=float(source_delta[j]),**{f'seed{seed}_delta_dB':float(matrix[i,j]) for i,seed in enumerate(seeds)}))
    payload=dict(unit='16 distinct natural-photo canonical source episodes, not videos; two lighting events averaged inside each source/seed, then three trained seeds averaged within source',CI='5000 circular source-block(length2) draws, conditional on these three trained checkpoints; donor reuse means remaining correlation and exploratory interval',training_seed_interpretation='Seed changes optimization only; never count as an independent source or video',results=results)
    (root/'source_paired_outcomes.json').write_text(json.dumps(payload,indent=2))
    with (root/'source_paired_differences.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
