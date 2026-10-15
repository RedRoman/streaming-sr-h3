"""Exact information-form SR for periodic blur/translation/decimation.

This is a classical inverse-problem baseline, not a claimed novel model.
Unlike independent local footprint toys, translation mixes neighbouring pixels.
The Fourier alias groups make the complete information matrix block diagonal.
Each group has scale**2 coefficients; memory is independent of stream length.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


class FourierLedger:
    def __init__(self, shape, scale=4, prior=None, precision=0.01):
        self.shape = tuple(shape)
        self.scale = int(scale)
        self.h, self.w = (n // scale for n in self.shape)
        assert self.shape == (self.h * scale, self.w * scale)
        self.k = scale * scale
        self.fy, self.fx = np.meshgrid(
            np.fft.fftfreq(self.shape[0]), np.fft.fftfreq(self.shape[1]), indexing="ij"
        )
        self.prior = np.zeros(self.shape) if prior is None else np.asarray(prior).copy()
        assert self.prior.shape == self.shape
        self.precision = np.broadcast_to(np.asarray(precision), self.shape).copy()
        if np.min(self.precision) <= 0:
            raise ValueError("Prior precision must be positive")
        self.j0 = self.pack(self.precision)
        self.gram = np.zeros((self.h, self.w, self.k, self.k), dtype=np.complex128)
        self.rhs = np.zeros((self.h, self.w, self.k), dtype=np.complex128)
        self.count = 0
        self.observation_ids = set()

    def pack(self, x):
        return np.asarray(x).reshape(self.scale, self.h, self.scale, self.w).transpose(1, 3, 0, 2).reshape(self.h, self.w, self.k)

    def unpack(self, x):
        return np.asarray(x).reshape(self.h, self.w, self.scale, self.scale).transpose(2, 0, 3, 1).reshape(self.shape)

    def design(self, shift=(0, 0), blur=0.7):
        transfer = np.exp(-2 * np.pi**2 * blur**2 * (self.fy**2 + self.fx**2))
        phase = np.exp(-2j * np.pi * (self.fy * shift[0] + self.fx * shift[1]))
        return self.pack(transfer * phase) / self.scale

    def observe(self, x, shift=(0, 0), blur=0.7):
        """Forward operator in real space, computed independently of alias design."""
        transfer = np.exp(-2 * np.pi**2 * blur**2 * (self.fy**2 + self.fx**2))
        phase = np.exp(-2j * np.pi * (self.fy * shift[0] + self.fx * shift[1]))
        y = np.fft.ifft2(np.fft.fft2(x, norm="ortho") * transfer * phase, norm="ortho").real
        return y[::self.scale, ::self.scale]

    def add(self, y, shift=(0, 0), blur=0.7, weight=1.0, observation_id=None):
        """Only real input measurements write evidence. Copied packets deduplicate."""
        if observation_id is not None and observation_id in self.observation_ids:
            return False
        a = self.design(shift, blur)
        yf = np.fft.fft2(y, norm="ortho")
        self.gram += weight * a.conj()[..., :, None] * a[..., None, :]
        self.rhs += weight * a.conj() * yf[..., None]
        self.count += 1
        if observation_id is not None:
            self.observation_ids.add(observation_id)
        return True

    def system(self):
        return self.gram + np.eye(self.k) * self.j0[..., None, :]

    def solve(self, prior=None):
        """Posterior mean / ridge MAP; optional prior must not be counted as data."""
        prior = self.prior if prior is None else np.asarray(prior)
        pf = self.pack(np.fft.fft2(prior, norm="ortho"))
        xf = np.linalg.solve(self.system(), (self.rhs + self.j0 * pf)[..., None])[..., 0]
        return np.fft.ifft2(self.unpack(xf), norm="ortho").real

    def posterior_transport(self, previous, previous_mean, previous_covariance):
        """Classical Gaussian Bures transport, useful as an exact small teacher.

        Returns a sample, mean and covariance. No assertion that a Gaussian
        posterior is an adequate model for natural high-frequency appearance.
        """
        mean = self.solve()
        cov = np.linalg.inv(self.system())
        def sqrt_psd(a, inverse=False):
            w, v = np.linalg.eigh(a)
            d = np.maximum(w, 1e-10) ** (-0.5 if inverse else 0.5)
            return (v * d[..., None, :]) @ v.swapaxes(-1, -2).conj()
        old_sqrt = sqrt_psd(previous_covariance)
        old_inv = sqrt_psd(previous_covariance, inverse=True)
        transport = old_inv @ sqrt_psd(old_sqrt @ cov @ old_sqrt) @ old_inv
        residual = self.pack(np.fft.fft2(previous - previous_mean, norm="ortho"))
        new_residual = (transport @ residual[..., None])[..., 0]
        sample = mean + np.fft.ifft2(self.unpack(new_residual), norm="ortho").real
        return sample, mean, cov

    @property
    def nbytes(self):
        return self.gram.nbytes + self.rhs.nbytes + self.j0.nbytes + self.prior.nbytes


def self_test():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(16, 16))
    m = FourierLedger(x.shape, 4, precision=0.005)
    max_forward = 0.0
    for shift in [(0, 0), (1, 2), (3, 1)]:
        y = m.observe(x, shift, 0.6)
        yf_alias = np.sum(m.design(shift, 0.6) * m.pack(np.fft.fft2(x, norm="ortho")), axis=-1)
        max_forward = max(max_forward, float(np.max(np.abs(yf_alias - np.fft.fft2(y, norm="ortho")))))
        m.add(y, shift, 0.6)
    # Independently construct the full real-space operator and solve dense MAP.
    a = []
    for shift in [(0, 0), (1, 2), (3, 1)]:
        cols = []
        for i in range(x.size):
            e = np.zeros_like(x); e.flat[i] = 1
            cols.append(m.observe(e, shift, 0.6).ravel())
        a.append(np.stack(cols, axis=1))
    a = np.concatenate(a, axis=0)
    dense = np.linalg.solve(a.T @ a + 0.005 * np.eye(x.size), a.T @ (a @ x.ravel())).reshape(x.shape)
    dense_error = float(np.max(np.abs(m.solve() - dense)))
    m.add(m.observe(x), observation_id="same_packet")
    before = m.rhs.copy()
    assert not m.add(m.observe(x), observation_id="same_packet")
    dedup_error = float(np.max(np.abs(before - m.rhs)))
    old_hash = (m.gram.copy(), m.rhs.copy(), m.count)
    m.solve(); m.solve(prior=np.ones_like(x))
    readonly = np.array_equal(old_hash[0], m.gram) and np.array_equal(old_hash[1], m.rhs) and old_hash[2] == m.count
    assert max_forward < 1e-10, max_forward
    assert dense_error < 1e-10, dense_error
    assert readonly and dedup_error == 0
    return {"max_forward_dft_error": max_forward, "max_dense_map_error": dense_error,
            "duplicate_packet_delta": dedup_error, "query_readonly": readonly,
            "state_bytes_16x16": m.nbytes, "passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    t = time.time(); result = self_test(); result["seconds"] = time.time() - t
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
