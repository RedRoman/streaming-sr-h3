"""Prefix-only LR correspondence for H3 memory experiments.

This is a classical DIS optical-flow baseline, not the proposed learned core.
Forward/backward and photometric scores indicate local matching consistency;
they are not probabilities of object identity. No HR image or annotation is an
argument to the estimator. Direct anchor matches allow recovery from chain
drift but do not establish long-gap semantic re-identification.
"""
from dataclasses import dataclass
import hashlib
import cv2
import numpy as np


def grid(height, width):
    yy, xx = np.mgrid[:height, :width]
    return np.stack([xx, yy], -1).astype(np.float32)


def sample(image, coordinates):
    return cv2.remap(image, coordinates[..., 0].astype(np.float32),
                     coordinates[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def inside(coordinates, shape):
    h, w = shape[:2]
    return ((coordinates[..., 0] >= 0) & (coordinates[..., 0] <= w - 1) &
            (coordinates[..., 1] >= 0) & (coordinates[..., 1] <= h - 1))


def warp_reference(reference, current_to_reference):
    """Warp an s-times larger RGB/feature reference using an LR absolute map.

    Both images occupy the same pixel footprints. Interpolate displacement,
    not absolute positions, preserving the half-pixel convention and identity.
    The result has reference's spatial resolution, with constant-zero borders.
    """
    lh, lw = current_to_reference.shape[:2]
    hh, hw = reference.shape[:2]
    displacement = current_to_reference - grid(lh, lw)
    resized = cv2.resize(displacement, (hw, hh), interpolation=cv2.INTER_LINEAR)
    resized *= np.array([hw / lw, hh / lh], dtype=np.float32)
    return sample(reference, grid(hh, hw) + resized)


def _gray(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected an observed LR uint8 RGB image')
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)


def pair_correspondence(reference, current, photo_sigma=.05, cycle_sigma=1.):
    """Backward map on the current grid, with both directions estimated in LR."""
    if reference.shape != current.shape:
        raise ValueError('Reference/current LR grids must match')
    first, second = _gray(reference), _gray(current)
    # Separate estimator instances avoid implicit flow reuse across pairs.
    backward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(second, first, None)
    forward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(first, second, None)
    coordinates = grid(*first.shape) + backward
    cycle = backward + sample(forward, coordinates)
    cycle_squared = np.sum(cycle * cycle, axis=-1)
    warped = sample(reference.astype(np.float32) / 255, coordinates)
    photo_squared = np.mean((warped - current.astype(np.float32) / 255) ** 2, axis=-1)
    valid = inside(coordinates, reference.shape)
    confidence = np.exp(-.5 * (cycle_squared / cycle_sigma ** 2 + photo_squared / photo_sigma ** 2)) * valid
    return coordinates, confidence.astype(np.float32), {
        'cycle_error_px': np.sqrt(cycle_squared),
        'photometric_rmse': np.sqrt(photo_squared),
        'in_bounds': valid,
    }


@dataclass(frozen=True)
class Correspondence:
    index: int
    observation_id: str
    current_to_previous: np.ndarray
    current_to_canonical: np.ndarray
    pair_confidence: np.ndarray
    canonical_confidence: np.ndarray
    fresh: bool
    direct_fraction: float


class LRCorrespondence:
    def __init__(self, direct_interval=17, photo_sigma=.05, cycle_sigma=1.):
        self.direct_interval = direct_interval
        self.photo_sigma = photo_sigma
        self.cycle_sigma = cycle_sigma
        self.anchor = self.previous = self.result = None
        self.ids = set()

    def push(self, rgb, observation_id, index):
        _gray(rgb)
        if observation_id in self.ids:
            # Duplicates are not new observations; no geometry/state mutation.
            r = self.result
            return Correspondence(r.index, r.observation_id,
                                  r.current_to_previous.copy(), r.current_to_canonical.copy(),
                                  r.pair_confidence.copy(), r.canonical_confidence.copy(), False, r.direct_fraction)
        if self.result is not None and index <= self.result.index:
            raise ValueError('Fresh frames must arrive in strictly increasing order')
        if self.anchor is None:
            self.anchor = rgb.copy()
            coordinates = grid(*rgb.shape[:2])
            confidence = np.ones(rgb.shape[:2], np.float32)
            result = Correspondence(index, observation_id, coordinates, coordinates.copy(),
                                    confidence, confidence.copy(), True, 0.)
        else:
            pair, pair_confidence, _ = pair_correspondence(self.previous, rgb, self.photo_sigma, self.cycle_sigma)
            chained = sample(self.result.current_to_canonical, pair)
            inherited = sample(self.result.canonical_confidence, pair)
            # Historical certainty cannot recover solely through chaining.
            confidence = np.minimum(inherited, pair_confidence)
            anchor_error = np.mean((sample(self.anchor.astype(np.float32) / 255, chained) - rgb.astype(np.float32) / 255) ** 2, -1)
            confidence *= np.exp(-.5 * anchor_error / self.photo_sigma ** 2) * inside(chained, self.anchor.shape)
            fraction = 0.
            if index % self.direct_interval == 0:
                direct, direct_confidence, _ = pair_correspondence(self.anchor, rgb, self.photo_sigma, self.cycle_sigma)
                use_direct = direct_confidence > confidence
                chained = np.where(use_direct[..., None], direct, chained)
                confidence = np.maximum(confidence, direct_confidence)
                fraction = float(use_direct.mean())
            result = Correspondence(index, observation_id, pair, chained, pair_confidence,
                                    confidence.astype(np.float32), True, fraction)
        self.ids.add(observation_id)
        self.previous = rgb.copy()
        # Protect internal state from accidental mutation through returned arrays.
        for value in result.__dict__.values():
            if isinstance(value, np.ndarray): value.setflags(write=False)
        self.result = result
        return result

    def state_digest(self):
        digest = hashlib.sha256()
        digest.update(repr(sorted(self.ids)).encode())
        if self.result is not None:
            digest.update(str(self.result.index).encode())
            for value in [self.anchor, self.previous, self.result.current_to_canonical,
                          self.result.canonical_confidence]:
                digest.update(value.tobytes())
        return digest.hexdigest()
