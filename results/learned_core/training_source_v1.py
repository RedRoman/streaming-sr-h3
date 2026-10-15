"""Causal canonical-footprint SR experiment. No future frame or HR at inference.

This deliberately controlled protocol assumes known correspondence: a frame is one
or more of 16 point-sampling footprints within a 4x4 canonical HR cell. It is not a
claim of general video SR. Natural source images are split by BSDS source split.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pack(x):
    return F.pixel_unshuffle(x, 4)


def unpack(x):
    return F.pixel_shuffle(x, 4)


class Sources:
    def __init__(self, root, split, synthetic=False):
        self.split = split
        self.paths = sorted((Path(root) / split).glob('*.jpg'))
        if not self.paths and not synthetic:
            raise RuntimeError(f'No {split} source images at {root}')
        self.images = [np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255 for p in self.paths]

    def batch(self, rng, n, side, texture_fraction=0.2, image_indices=None):
        result = []
        for index in range(n):
            if self.images and rng.random() >= texture_fraction:
                image_index = int(rng.integers(len(self.images))) if image_indices is None else int(image_indices[index])
                a = self.images[image_index]
                y, x = int(rng.integers(a.shape[0]-side+1)), int(rng.integers(a.shape[1]-side+1))
                a = a[y:y+side, x:x+side].copy()
                a = np.rot90(a, int(rng.integers(4))).copy()
                if rng.random() < .5:
                    a = a[:, ::-1].copy()
            else:
                yy, xx = np.mgrid[:side, :side]
                a = np.full((side, side), rng.uniform(.3, .7), dtype=np.float32)
                for _ in range(3):
                    angle, freq, phase = rng.uniform(0, 2*np.pi), rng.uniform(.04, .46), rng.uniform(0, 2*np.pi)
                    a += rng.uniform(.06, .16)*np.sin(2*np.pi*freq*(xx*np.cos(angle)+yy*np.sin(angle))+phase)
                a = np.clip(a, 0, 1).astype(np.float32)
            result.append(a)
        return torch.from_numpy(np.stack(result)[:, None])

    def evaluation(self, count, side, seed, texture_fraction=0):
        rng = np.random.default_rng(seed)
        if texture_fraction == 0 and self.images:
            count = min(count,len(self.images))
            indices = rng.permutation(len(self.images))[:count]
            return self.batch(rng,count,side,0,indices)
        return self.batch(rng,count,side,texture_fraction)


class ConvCore(nn.Module):
    def __init__(self, inputs=32, outputs=16, width=62):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(inputs, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, outputs, 3, padding=1))

    def forward(self, z):
        return self.net(z)


class CoreModel(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind in ('current', 'ledger', 'map_residual'):
            self.net = ConvCore()
        elif kind in ('transport', 'map_transport'):
            self.net = ConvCore(64, 32, 47)
        elif kind == 'gru':
            self.gates = nn.Conv2d(64, 64, 3, padding=1)
            self.candidate = nn.Conv2d(64, 32, 3, padding=1)
            self.out = nn.Conv2d(32, 16, 3, padding=1)
        else:
            raise ValueError(kind)

    def forward(self, ys, masks, map_bases=None):
        shape = ys[0].shape
        sums = torch.zeros_like(ys[0])
        counts = torch.zeros_like(sums)
        prior = torch.full_like(sums, .5)
        detail = torch.zeros_like(sums)
        h = torch.zeros((shape[0], 32, *shape[2:]), device=sums.device)
        outputs = []
        for y, mask in zip(ys, masks):
            old_mean = sums / counts.clamp_min(1)
            new_mask = mask * (counts == 0)
            innovation = mask * (y - old_mean)
            # Simulator repeats reuse the SAME noisy sample, so count each
            # footprint once rather than pretending copies are independent.
            sums = sums + y*new_mask
            counts = counts + new_mask
            mean = sums / counts.clamp_min(1)
            if self.kind == 'current':
                prior = self.net(torch.cat([y*mask, mask], 1)) + .5
                estimate = .98*y*mask + (1-.98*mask)*prior
            elif self.kind == 'ledger':
                prior = self.net(torch.cat([mean, counts.clamp_max(4)/4], 1)) + .5
                weight = counts / (counts + .02)
                estimate = weight*mean+(1-weight)*prior
            elif self.kind == 'gru':
                inp = torch.cat([y*mask, mask], 1)
                reset, update = self.gates(torch.cat([inp, h], 1)).sigmoid().chunk(2, 1)
                candidate = self.candidate(torch.cat([inp, reset*h], 1)).tanh()
                h = (1-update)*h+update*candidate
                prior = self.out(h)+.5
                estimate = .98*y*mask+(1-.98*mask)*prior
            elif self.kind in ('map_residual', 'map_transport'):
                base = map_bases[len(outputs)]
                if self.kind == 'map_residual':
                    detail = self.net(torch.cat([base, counts.clamp_max(4)/4], 1))
                else:
                    proposal, gain = self.net(torch.cat([base, counts.clamp_max(4)/4, detail, innovation], 1)).chunk(2, 1)
                    novelty = new_mask.flatten(1).amax(1)[:, None, None, None]
                    detail = detail + gain.sigmoid()*novelty*(proposal-detail)
                # Learned non-Gaussian detail can only modify nullspace
                # directions; observed directions use the Gaussian posterior.
                estimate = base+(counts == 0)*detail
            else:
                proposal, gain = self.net(torch.cat([mean, counts.clamp_max(4)/4, prior, innovation], 1)).chunk(2, 1)
                # Exact repetitions carry no novel rank and cannot rewrite the
                # persistent prior. With new directions, learn the revisability.
                novelty = new_mask.flatten(1).amax(1)[:, None, None, None]
                gain = gain.sigmoid()*novelty
                prior = prior+gain*(proposal+.5-prior)
                weight = counts/(counts+.02)
                estimate = weight*mean+(1-weight)*prior
            outputs.append(estimate)
        return outputs


def observations(target, seed, steps=8, mode='new', noise=.01, shared_order=False):
    """All frames depend only on canonical target and present footprint.

    Target is used by the simulator only, never supplied to a network.
    Repeats reuse the exact noisy frame, so identity stability is measurable.
    """
    g = torch.Generator(device=target.device).manual_seed(seed)
    n, c, h, w = target.shape
    order = torch.stack([torch.randperm(16, generator=g, device=target.device) for _ in range(n)])
    if shared_order:
        order = order[:1].expand(n, -1)
    eps = torch.randn(target.shape, generator=g, device=target.device)*noise
    noisy = target+eps
    ys, masks = [], []
    for t in range(steps):
        mask = torch.zeros((n, 16, 1, 1), device=target.device)
        phase = order[:, t % 16]
        if mode == 'repeat':
            phase = order[:, 0]
        elif mode == 'mixed_repeat':
            phase = order[:, min(t, 3)]
        mask[torch.arange(n, device=target.device), phase] = 1
        if mode == 'reveal' and t == 4:
            mask[:] = 1
        values = noisy
        if mode == 'wrong' and t % 4 == 3:
            values = torch.roll(noisy, 1, dims=1)
        ys.append(values*mask)
        masks.append(mask.expand(-1, -1, h, w))
    return ys, masks


def gaussian_covariance(source, side, device, samples=512):
    """Fit a 3x3 canonical-cell Gaussian prior on TRAIN images only.

    The 144x144 covariance yields a joint all-history local Gaussian MAP
    predictor, including neighboring LR cells. This is stronger than an
    independent per-pixel ridge with a flat prior.
    """
    rng = np.random.default_rng(8821)
    total, second, number = None, None, 0
    for start in range(0, samples, 32):
        x = pack(source.batch(rng, min(32, samples-start), side, 0).to(device))
        z = F.unfold(x, 3, padding=1).transpose(1, 2).reshape(-1, 144).double()
        if total is None:
            total, second = z.sum(0), z.T@z
        else:
            total += z.sum(0)
            second += z.T@z
        number += len(z)
    mean = total/number
    cov = second/number-mean[:, None]*mean[None, :]
    cov = cov.float()+torch.eye(144, device=device)*1e-5
    center = torch.arange(16, device=device)*9+4
    return mean.float(), cov, center


def gaussian_map(ys, masks, stats, ridge=.0001):
    mean, cov, center = stats
    sums, counts = torch.zeros_like(ys[0]), torch.zeros_like(masks[0])
    out = []
    for y, mask in zip(ys, masks):
        new_mask = mask*(counts == 0)
        sums += y*new_mask
        counts += new_mask
        n, _, h, w = sums.shape
        values = F.unfold(sums/counts.clamp_min(1), 3, padding=1)
        mask_vec = (counts[:, :, 0, 0] > 0).repeat_interleave(9, dim=1)
        prediction = []
        if torch.equal(counts[:, :, 0, 0], counts[:1, :, 0, 0].expand(n,-1)):
            ix = mask_vec[0].nonzero().flatten()
            c = counts[0, :, 0, 0].repeat_interleave(9)[ix]
            k = cov[ix[:, None], ix[None, :]]+torch.diag(ridge/c)
            gain = torch.linalg.solve(k, cov[center][:, ix].T).T
            pred = mean[center][None,:,None]+torch.einsum('ij,bjk->bik',gain,values[:,ix]-mean[ix][None,:,None])
            out.append(pred.reshape(n,16,h,w))
            continue
        for b in range(n):
            ix = mask_vec[b].nonzero().flatten()
            # Noise variance decreases with repeated measurements.
            c = counts[b, :, 0, 0].repeat_interleave(9)[ix]
            k = cov[ix[:, None], ix[None, :]]+torch.diag(ridge/c)
            gain = torch.linalg.solve(k, cov[center][:, ix].T).T
            pred = mean[center, None]+gain@(values[b, ix]-mean[ix, None])
            prediction.append(pred.reshape(16, h, w))
        out.append(torch.stack(prediction))
    return out


def simple_map(ys, masks, prior=None, ridge=.02):
    sums, counts = torch.zeros_like(ys[0]), torch.zeros_like(masks[0])
    outputs = []
    for t, (y, mask) in enumerate(zip(ys, masks)):
        new_mask = mask*(counts == 0)
        sums += y*new_mask
        counts += new_mask
        mean = sums/counts.clamp_min(1)
        if prior is None:
            p = sums.sum(1, keepdim=True)/counts.sum(1, keepdim=True).clamp_min(1)
        else:
            p = prior[t]
        weight = counts/(counts+ridge)
        outputs.append(weight*mean+(1-weight)*p)
    return outputs


def score(outputs, target, ys, masks):
    error = torch.stack([((p-target)**2).flatten(1).mean(1) for p in outputs], 1)
    drift = torch.stack([((outputs[t]-outputs[t-1])**2).flatten(1).mean(1) for t in range(1,len(outputs))], 1)
    hf = lambda x: (unpack(x)[:, :, :, 1:]-unpack(x)[:, :, :, :-1]).abs().flatten(1).mean(1)
    gt_hf = hf(target).clamp_min(1e-6)
    return {'psnr_mean_db': float((-10*torch.log10(error.clamp_min(1e-10))).mean()),
            'psnr_final_db': float((-10*torch.log10(error[:, -1].clamp_min(1e-10))).mean()),
            'mse_final': float(error[:, -1].mean()),
            'drift_last3_mse': float(drift[:, -3:].mean()),
            'hf_ratio_final': float((hf(outputs[-1])/gt_hf).mean()),
            'psnr_per_t': (-10*torch.log10(error.mean(0).clamp_min(1e-10))).tolist(),
            'sample_mse_final': error[:, -1].tolist()}


def evaluate(models, target, stats, seed, mode, noise):
    ys, masks = observations(target, seed, mode=mode, noise=noise)
    result = {}
    cache = {}
    map_bases = gaussian_map(ys, masks, stats, noise**2)
    for kind, model in models.items():
        model.eval()
        cache[kind] = model(ys, masks, map_bases)
        result[kind] = score(cache[kind], target, ys, masks)
    outputs = map_bases
    result['gaussian_all_history_map'] = score(outputs, target, ys, masks)
    result['flat_all_history_map'] = score(simple_map(ys, masks), target, ys, masks)
    result['current_prior_all_history_map'] = score(simple_map(ys, masks, cache['current']), target, ys, masks)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/BSR/BSDS500/data/images')
    p.add_argument('--out', default='results/learned_core')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--side', type=int, default=48)
    p.add_argument('--eval-count', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--synthetic-smoke', action='store_true')
    args = p.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = {s: Sources(args.data, s, args.synthetic_smoke) for s in ['train', 'val', 'test']}
    manifest = {'config': vars(args), 'source_images': {s: [str(x) for x in v.paths] for s,v in sources.items()},
                'protocol': '4x known canonical point footprints; grayscale; 8 causal frames; 0.01 Gaussian noise fixed across repeats',
                'limitations': ['Known exact canonical correspondence, not estimated optical flow.',
                                '4x4 point footprints, not standard bicubic degradation.',
                                'Synthetic texture mixed into train at 20%; natural test is held-out BSDS test images.']}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'event':'sources','counts':{s:len(v.paths) for s,v in sources.items()}}), flush=True)
    stats = gaussian_covariance(sources['train'], args.side, device, samples=128 if args.synthetic_smoke else 512)
    torch.save([v.cpu() for v in stats], out/'gaussian_prior.pt')
    evaluation = {('val','new'): pack(sources['val'].evaluation(args.eval_count,args.side,198)).to(device)}
    for mode in ['new','repeat','mixed_repeat','reveal','wrong']:
        evaluation[('test',mode)] = pack(sources['test'].evaluation(args.eval_count,args.side,269)).to(device)
    evaluation[('texture_ood','new')] = pack(sources['test'].evaluation(args.eval_count,args.side,397,1)).to(device)
    summary, rows = {}, []
    for seed in args.seeds:
        seed_all(seed)
        models = {kind:CoreModel(kind).to(device) for kind in ['current','gru','ledger','transport','map_residual','map_transport']}
        optimizers = {kind: torch.optim.Adam(m.parameters(),lr=.0007) for kind,m in models.items()}
        params = {kind:sum(p.numel() for p in model.parameters()) for kind,model in models.items()}
        print(json.dumps({'event':'start','seed':seed,'parameters':params}), flush=True)
        rng = np.random.default_rng(seed)
        history, start = [], time.time()
        for step in range(1,args.steps+1):
            target = pack(sources['train'].batch(rng,args.batch,args.side).to(device))
            mode = ['new','new','mixed_repeat','reveal'][step%4]
            ys, masks = observations(target,seed*100000+step,steps=8,mode=mode,shared_order=True)
            with torch.no_grad():
                map_bases = gaussian_map(ys,masks,stats,.0001)
            losses = {}
            for kind,model in models.items():
                model.train()
                optimizers[kind].zero_grad(set_to_none=True)
                outputs = model(ys,masks,map_bases)
                loss = sum(F.mse_loss(x,target) for x in outputs)/len(outputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizers[kind].step()
                losses[kind] = float(loss.detach())
            if step%100 == 0 or step == args.steps:
                entry = {'step':step,'seconds':time.time()-start,'loss':losses}
                history.append(entry)
                print(json.dumps({'event':'train','seed':seed,**entry}),flush=True)
        result = {'seed':seed,'parameters':params,'train_steps':args.steps,'train_seconds':time.time()-start,'history':history,'evaluation':{}}
        with torch.no_grad():
            for (split,mode),target in evaluation.items():
                metrics = evaluate(models,target,stats,50199,mode,.01)
                result['evaluation'][split+'/'+mode] = metrics
                print(json.dumps({'event':'eval','seed':seed,'split':split,'mode':mode,'final_psnr':{k:round(v['psnr_final_db'],4) for k,v in metrics.items()}}),flush=True)
                for model,values in metrics.items():
                    rows.append({'seed':seed,'split':split,'condition':mode,'model':model,**{k:v for k,v in values.items() if not isinstance(v,list)}})
        torch.save({'models':{k:m.state_dict() for k,m in models.items()},'seed':seed,'steps':args.steps,'config':vars(args)},out/f'checkpoint_seed{seed}.pt')
        (out/f'seed{seed}.json').write_text(json.dumps(result,indent=2))
        summary[str(seed)] = result
        (out/'all_results.json').write_text(json.dumps(summary,indent=2))
        with (out/'metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'event':'complete','out':str(out)}),flush=True)


if __name__ == '__main__':
    main()
