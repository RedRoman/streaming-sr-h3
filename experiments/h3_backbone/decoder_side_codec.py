"""Frozen-book decoder-side codec pilot, with a real seven-bit wire format.

The only candidate change is the fine-to-bin assignment. Balanced assignment
is a DSC-inspired baseline candidate, not a claim of a new coding principle.
This module has no data paths, source dictionary, image encoder or GPU policy.
Writer takes past features only; independent reader takes bytes and one cue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def pack7(indices: torch.Tensor) -> torch.Tensor:
    """[B,...] indices 0..127 -> [B,ceil(7*N/8)] uint8, LSB first.

    Symbols are flattened in row-major H,W,G order. Bit k of each symbol is
    stored at bit (7*n+k)%8 of byte floor((7*n+k)/8). Final padding is zero.
    """
    if indices.ndim < 2 or bool(((indices < 0) | (indices > 127)).any()):
        raise ValueError('Expected batched seven-bit symbols')
    v = indices.detach().long().reshape(indices.shape[0], -1)
    bits = ((v[..., None] >> torch.arange(7, device=v.device)) & 1).flatten(1)
    pad = (-bits.shape[1]) % 8
    bits = F.pad(bits, (0, pad))
    return (bits.reshape(v.shape[0], -1, 8) *
            (1 << torch.arange(8, device=v.device))).sum(-1).to(torch.uint8)


def unpack7(blob: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if blob.dtype != torch.uint8 or blob.ndim != 2:
        raise ValueError('Wire packet must be [B,bytes] uint8')
    n = math.prod(shape)
    if blob.shape[1] != (7 * n + 7) // 8:
        raise ValueError('Unexpected packet length')
    bits = ((blob.long()[..., None] >> torch.arange(8, device=blob.device)) & 1).flatten(1)
    if bool(bits[:, 7*n:].any()):
        raise ValueError('Nonzero padding')
    v = (bits[:, :7*n].reshape(blob.shape[0], n, 7) *
         (1 << torch.arange(7, device=blob.device))).sum(-1)
    return v.reshape(blob.shape[0], *shape)


def _lookup(book: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    # book G,K,D; index B,H,W,G -> B,H,W,G,D
    groups = torch.arange(book.shape[0], device=book.device)
    return book[groups, index]


def _nearest(features: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    return (features.unsqueeze(-2) - book).square().sum(-1).argmin(-1)


def _nearest_st(features: torch.Tensor, book: torch.Tensor,
                temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    distance = (features.unsqueeze(-2) - book).square().sum(-1)
    index = distance.argmin(-1)
    hard = F.one_hot(index, book.shape[1]).to(features.dtype)
    soft = (-distance / temperature).softmax(-1)
    return hard + (soft - soft.detach()), index


class DecoderSideCodec(nn.Module):
    """Fine256->bin128 codec, or nearest conditional-VQ128 rate control.

    API shapes: features[B,H,W,G,D], future_logits[B,H,W,G,256]. The latter
    MUST be computed using exactly one arrived future LR and its known phase.
    It must not depend on past features, fine indices, targets, source IDs or
    the other three possible future observations. Both books are frozen.

    modes='cvq'|'categorical'|'balanced'. The last two have identical trainable
    Theta[G,256,128]; balanced constrains each bin to exactly two prototypes.
    All arms train a writer encoder through nearest-assignment ST, the same
    future-selector CE, and a future-conditioned RGB reader. Only the books
    are frozen. Thus the same pair loss can change transmitted VQ128 indices.
    """
    def __init__(self, fine_book: torch.Tensor, coarse_book: torch.Tensor,
                 mode: str, temperature: float = .5, sinkhorn_steps: int = 32):
        super().__init__()
        if mode not in ('cvq', 'categorical', 'balanced'):
            raise ValueError(mode)
        if fine_book.ndim != 3 or fine_book.shape[1] != 256:
            raise ValueError('fine_book must be G,256,D')
        if coarse_book.shape != (fine_book.shape[0], 128, fine_book.shape[2]):
            raise ValueError('coarse_book must be G,128,D')
        self.mode, self.temperature, self.sinkhorn_steps = mode, temperature, sinkhorn_steps
        self.register_buffer('fine_book', fine_book.detach().float().clone())
        self.register_buffer('coarse_book', coarse_book.detach().float().clone())
        if mode != 'cvq':
            # Identical deterministic initialization for balanced/categorical.
            # Euclidean pairing is an initialization, never a held-source fit.
            distance = (fine_book[:, :, None] - coarse_book[:, None]).square().sum(-1)
            scale = distance.mean((-1, -2), keepdim=True).clamp_min(1e-6)
            self.theta = nn.Parameter((-distance / scale).float())
        else:
            self.register_parameter('theta', None)

    def assignment(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard-forward/soft-backward P[G,256,128], hard P."""
        if self.theta is None:
            raise RuntimeError('Direct VQ has no fine-to-bin assignment')
        logits = self.theta / self.temperature
        if self.mode == 'categorical':
            soft = logits.softmax(-1)
            hard = F.one_hot(logits.argmax(-1), 128).to(logits.dtype)
        else:
            logp = logits
            for _ in range(self.sinkhorn_steps):
                logp = logp - logp.logsumexp(-1, keepdim=True)
                logp = logp + math.log(2.) - logp.logsumexp(-2, keepdim=True)
            soft = logp.softmax(-1)
            # Two identical slots per bin: bijection onto 256 slots gives
            # exactly two fine prototypes in every bin, without greedy bias.
            from scipy.optimize import linear_sum_assignment
            hard = torch.zeros_like(logits)
            for g in range(logits.shape[0]):
                score = logits[g].detach().cpu().double().repeat_interleave(2, -1).numpy()
                rows, slots = linear_sum_assignment(-score)
                hard[g, torch.as_tensor(rows, device=logits.device),
                     torch.as_tensor(slots // 2, device=logits.device)] = 1
        # This form is bit-exact hard in forward, unlike soft+(hard-soft).detach.
        return hard + (soft - soft.detach()), hard

    def nearest_fine(self, features: torch.Tensor) -> torch.Tensor:
        return _nearest(features, self.fine_book)

    def _posterior(self, emission: torch.Tensor, assignment: torch.Tensor,
                   future_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membership = torch.einsum('bhwgc,gjc->bhwgj', emission, assignment)
        # Normalize inside the actual received bin, never global-softmax then
        # clamp its tiny mass. This remains exact even when its members have
        # logits hundreds below an unrelated prototype.
        in_bin = membership.detach() > 0
        selected = future_logits.masked_fill(~in_bin, -torch.inf)
        weights = selected.softmax(-1)
        # The hard membership comparison cannot carry assignment gradients.
        # Add its zero-valued ST differential analytically. For real members
        # this is the exact conditional derivative. For absent members its
        # likelihood-ratio derivative is capped at1e6 to avoid overflow under
        # extreme, incompatible cues. It changes backward only, not the wire
        # codec, the exact within-bin probabilities, or cue-logit gradients.
        log_z = selected.logsumexp(-1, keepdim=True)
        ratio = (future_logits-log_z).clamp(max=math.log(1e6)).exp()
        differential = (membership-membership.detach()) * ratio
        weights = weights + differential - weights*differential.sum(-1, keepdim=True)
        return torch.einsum('bhwgj,gjd->bhwgd', weights, self.fine_book), weights

    def forward(self, features: torch.Tensor, future_logits: torch.Tensor) -> dict:
        fine_emission, fine = _nearest_st(features, self.fine_book, self.temperature)
        fine_feature = _lookup(self.fine_book, fine)
        if self.mode == 'cvq':
            emission, index = _nearest_st(features, self.coarse_book, self.temperature)
            reconstructed = torch.einsum('bhwgc,gcd->bhwgd', emission, self.coarse_book)
            posterior = None
        else:
            assignment, hard = self.assignment()
            emission = torch.einsum('bhwgj,gjc->bhwgc', fine_emission, assignment)
            index = emission.argmax(-1)
            reconstructed, posterior = self._posterior(emission, assignment, future_logits)
        # Identical fine-book commitment trains every arm's writer. Books are
        # buffers; fine indices serve only as a detached selector target.
        commitment = F.mse_loss(features, fine_feature)
        selector_ce = F.cross_entropy(future_logits.reshape(-1, 256), fine.reshape(-1))
        side_feature = torch.einsum('bhwgj,gjd->bhwgd', future_logits.softmax(-1), self.fine_book)
        return dict(feature=reconstructed, side_feature=side_feature, indices=index,
                    nearest_fine=fine, commitment=commitment,
                    selector_ce=selector_ce, posterior=posterior)

    @torch.no_grad()
    def write(self, features: torch.Tensor) -> torch.Tensor:
        """No future argument exists; returned bytes are the only message."""
        if self.mode == 'cvq':
            indices = _nearest(features, self.coarse_book)
        else:
            fine = self.nearest_fine(features)
            _, hard = self.assignment()
            indices = hard.argmax(-1)[torch.arange(self.fine_book.shape[0], device=features.device), fine]
        return pack7(indices)

    def read(self, blob: torch.Tensor, future_logits: torch.Tensor) -> dict:
        """Independent reader: no writer tensors/fine indices are accepted."""
        shape = (*future_logits.shape[1:3], self.fine_book.shape[0])
        indices = unpack7(blob, shape)
        if self.mode == 'cvq':
            feature = _lookup(self.coarse_book, indices)
        else:
            _, hard = self.assignment()
            emission = F.one_hot(indices, 128).to(future_logits.dtype)
            feature, _ = self._posterior(emission, hard, future_logits)
        side_feature = torch.einsum('bhwgj,gjd->bhwgd', future_logits.softmax(-1), self.fine_book)
        return dict(feature=feature, side_feature=side_feature, indices=indices)


def selftest(out: Path) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(17)
    out.mkdir(parents=True, exist_ok=True)
    fine = torch.randn(4, 256, 8)
    coarse = fine[:, ::2].clone()
    features_value = torch.randn(2, 16, 16, 4, 8)
    records = []
    for mode in ('cvq', 'categorical', 'balanced'):
        model = DecoderSideCodec(fine, coarse, mode)
        features = features_value.clone().requires_grad_()
        cue = torch.randn(2, 16, 16, 4, 256, requires_grad=True)
        result = model(features, cue)
        packet = model.write(features)
        decoded = model.read(packet, cue)
        error = (decoded['feature'] - result['feature']).abs().max().item()
        assert error == 0, (mode, error)
        assert packet.shape == (2, 896)
        assert torch.equal(unpack7(packet, (16, 16, 4)), result['indices'])
        assert torch.equal(packet, model.write(features))
        # Nonconstant reconstruction proxy plus shared selector CE, not a test
        # that merely mirrors assignment arithmetic.
        target = torch.randn_like(result['feature'])
        loss = F.mse_loss(result['feature'] + result['side_feature'], target) + .1 * result['selector_ce']
        loss.backward()
        theta_grad = None if model.theta is None else float(model.theta.grad.norm())
        assert theta_grad is None or theta_grad > 0
        assert float(cue.grad.norm()) > 0
        assert float(features.grad.norm()) > 0
        assert model.fine_book.grad is None and model.coarse_book.grad is None
        extreme_error = None
        if mode != 'cvq':
            # Every actual bin member has logit-1000, an absent member+1000.
            # A global-softmax/clamped denominator would collapse toward0.
            _, p_hard = model.assignment()
            emitted = result['indices']
            mask = p_hard.transpose(-1,-2)[torch.arange(4),emitted].bool()
            extreme = torch.where(mask, -1000., 1000.).requires_grad_()
            robust = model.read(packet, extreme)['feature']
            expected = torch.einsum('bhwgj,gjd->bhwgd',
                                    mask.float()/mask.sum(-1,keepdim=True), fine)
            extreme_error = float((robust-expected).abs().max())
            assert torch.allclose(robust, expected, atol=1e-6, rtol=1e-6)
            extreme_result = model(features, extreme)['feature']
            assert torch.equal(extreme_result, robust)
            extreme_grad = torch.autograd.grad(extreme_result.square().mean(),
                                                [model.theta, extreme], retain_graph=False)
            assert all(bool(torch.isfinite(g).all()) for g in extreme_grad)
        occupancy = None
        if model.theta is not None:
            _, hard = model.assignment()
            occupancy = hard.sum(-2).detach().tolist()
            if mode == 'balanced':
                assert bool((hard.sum(-2) == 2).all())
        # A newly instantiated independent reader receives only shared state,
        # packet bytes and cue. Neither features nor fine indices cross over.
        reader = DecoderSideCodec(fine, coarse, mode)
        reader.load_state_dict(model.state_dict())
        independent = reader.read(packet.clone(), cue.detach())
        assert torch.equal(independent['feature'], decoded['feature'].detach())
        (out/(mode+'_example0.bin')).write_bytes(packet[0].cpu().numpy().tobytes())
        records.append(dict(mode=mode, bytes_per_example=896, message_symbols=1024,
                            trainable_quantizer_parameters=sum(p.numel() for p in model.parameters()),
                            forward_wire_maxabs=error, theta_gradient_norm=theta_grad,
                            writer_feature_gradient_norm=float(features.grad.norm()),
                            extreme_cue_conditional_maxabs=extreme_error,
                            future_gradient_norm=float(cue.grad.norm()), frozen_books=True,
                            packet_sha256=hashlib.sha256(packet.cpu().numpy().tobytes()).hexdigest(),
                            balanced_occupancy=occupancy if mode == 'balanced' else None))
    # Non-byte-aligned generic length checks exercise padding, not just896B.
    for n in (1, 7, 8, 13, 1024):
        v = torch.randint(128, (3, n))
        assert torch.equal(unpack7(pack7(v), (n,)), v)
    report = dict(status='pass',device='CPU',GPU_used=False,training_updates=0,
                  H3_calls=0,torch_version=torch.__version__,records=records,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Wire, assignment and gradient API audit only; no codec efficacy claim')
    (out/'api_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest-out', type=Path, required=True)
    selftest(parser.parse_args().selftest_out)
