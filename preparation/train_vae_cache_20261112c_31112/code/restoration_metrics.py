"""CPU metric primitives shared by the versioned restoration evaluators.

No model, torch, accelerator, or project runtime is imported here. Historical
unversioned judge files are v1 and must not be pooled with these v2 statistics.
"""
import json
from pathlib import Path

import cv2
import numpy as np

METRIC_SCHEMA_VERSION = "restoration_metrics.v2"
FB_ALPHA = 0.01
FB_BETA = 0.5  # squared pixels, not pixels

SCALAR_ALIASES = {
    "psnr_gtrange": "psnr_gt_lowband_substitution",
    "lpips_gtrange": "lpips_gt_lowband_substitution",
    "null_energy": "detail_energy_ratio",
    "null_cos": "detail_cosine",
}
TEMPORAL_ALIASES = {
    "flicker": "detail_temporal_energy_ratio",
    "warp_range": "warp_lowband_ratio",
    "warp_null": "warp_detail_ratio",
    "terr_range": "temporal_lowband_error_x1000",
    "terr_null": "temporal_detail_error_x1000",
}


def metric_metadata():
    return {
        "version": METRIC_SCHEMA_VERSION,
        "compatible_with_unversioned_results": False,
        "changes_from_v1": [
            "Forward GT flow samples frame 1 onto frame 0, not the reverse.",
            "Warp errors exclude out-of-bounds and forward/backward-inconsistent pixels.",
            "Absolute temporal errors include static clips; undefined ratios are null.",
            "Paired better counts use metric direction; amplitude ratios have no better direction.",
        ],
        "band_definition": "lowband=U(D(x)); detail=x-U(D(x)); D=antialiased bicubic downsample, U=bicubic upsample",
        "band_interpretation": "A complementary frequency-band diagnostic, not an orthogonal projection or a physical observation/null-space decomposition. GT substitution is an oracle diagnostic, not achievable recovery.",
        "legacy_aliases": {**SCALAR_ALIASES, **TEMPORAL_ALIASES},
        "flow": {"source": "GT only, Farneback in both directions",
                 "comparison": "abs(frame0(x)-frame1(x+flow01(x)))",
                 "visibility": "in bounds and ||f01+f10(x+f01)||^2 <= alpha*(||f01||^2+||f10(x+f01)||^2)+beta",
                 "fb_alpha": FB_ALPHA, "fb_beta_squared_pixels": FB_BETA,
                 "limitation": "Forward/backward consistency is an occlusion heuristic, not ground-truth visibility."},
        "paired_difference": "a-b; ci95 is bootstrap CI of the mean, median is median of paired differences",
        "comparability": "Recompute all compared arms under v2; do not relabel old JSON as v2 or mix v1/v2 warp and aggregate statistics.",
    }


def add_aliases(record, aliases):
    """Keep legacy consumers working while exposing accurate band names."""
    record.update({new: record[old] for old, new in aliases.items() if old in record})
    return record


def assert_compatible_output(path):
    """Never overwrite a historical result with changed metric definitions."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
            raise ValueError(f"{path} is not {METRIC_SCHEMA_VERSION}; use a new output name. Historical metrics cannot be upgraded without rescoring.")


def remap_target_to_source(target, forward_flow):
    """Pull target-frame values onto source coordinates using source→target flow."""
    h, w = forward_flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    coords = np.asarray(forward_flow, dtype=np.float32)
    mx, my = gx + coords[..., 0], gy + coords[..., 1]
    # Invalid coordinates are excluded by the visibility mask. Keep remap itself finite.
    mx, my = np.nan_to_num(mx, nan=-1, posinf=-1, neginf=-1), np.nan_to_num(my, nan=-1, posinf=-1, neginf=-1)
    return cv2.remap(np.ascontiguousarray(target), mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def flow_visibility_mask(forward_flow, backward_flow, alpha=FB_ALPHA, beta=FB_BETA):
    """Shared GT-derived support, independent of restoration quality."""
    f, b = np.asarray(forward_flow), np.asarray(backward_flow)
    if f.shape != b.shape or f.ndim != 3 or f.shape[-1] != 2:
        raise ValueError("Flows must have matching [H,W,2] shapes")
    h, w = f.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mx, my = gx + f[..., 0], gy + f[..., 1]
    inside = np.isfinite(f).all(-1) & (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    bw = remap_target_to_source(b, f)
    consistent = np.square(f + bw).sum(-1) <= alpha * (np.square(f).sum(-1) + np.square(bw).sum(-1)) + beta
    return inside & np.isfinite(bw).all(-1) & consistent


def masked_warp_mae(frame0, frame1, forward_flow, mask):
    """Return None if no visible pixel exists; never substitute replicated borders."""
    if not np.any(mask):
        return None
    err = np.abs(np.asarray(frame0) - remap_target_to_source(np.asarray(frame1), forward_flow))
    return float(err[mask].mean())


def metric_direction(key):
    key = key.rsplit(".", 1)[-1]
    key = SCALAR_ALIASES.get(key, TEMPORAL_ALIASES.get(key, key))
    if key.startswith("psnr") or key == "detail_cosine":
        return "higher"
    if key.startswith("lpips") or key in {"temporal_lowband_error_x1000", "temporal_detail_error_x1000"}:
        return "lower"
    # Energy ratios, raw warp residuals and their GT ratios can reward smoothing.
    # They are diagnostics: neither globally higher nor globally lower is better.
    return "diagnostic"


def paired_counts(differences, key):
    values = np.asarray(differences, dtype=float)
    lower, higher = int((values < 0).sum()), int((values > 0).sum())
    direction = metric_direction(key)
    return {"direction": direction,
            "a_better": higher if direction == "higher" else lower if direction == "lower" else None,
            "a_lower": lower, "a_higher": higher, "ties": int((values == 0).sum())}


def psnr_from_mse(mse):
    return float(-10 * np.log10(max(float(mse), 1e-12)))


def band_error_terms(target, prediction, target_lowband, prediction_lowband):
    """Exact error identity for a complementary, generally nonorthogonal split."""
    gt, pred, gl, pl = [np.asarray(v, dtype=np.float64) for v in
                        (target, prediction, target_lowband, prediction_lowband)]
    low_error = pl - gl
    detail_error = (pred - pl) - (gt - gl)
    mse = float(np.square(pred - gt).mean())
    low = float(np.square(low_error).mean())
    detail = float(np.square(detail_error).mean())
    cross = float(2 * (low_error * detail_error).mean())
    substituted = gl + (pred - pl)
    sub_mse = float(np.square(substituted - gt).mean())
    clipped_mse = float(np.square(np.clip(substituted, 0, 1) - gt).mean())
    return {"mse_actual": mse, "mse_lowband": low, "mse_detail": detail,
            "mse_cross_term": cross, "mse_reconstructed": low + detail + cross,
            "mse_identity_abs_error": abs(mse - low - detail - cross),
            "mse_gt_lowband_substitution": sub_mse,
            "mse_gt_lowband_substitution_clipped": clipped_mse}
