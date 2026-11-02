import numpy as np, cv2, revisit_metrics as R
R.lpips_dist = lambda x, y: float('nan')
rng = np.random.default_rng(0)
H, W = 240, 320
base = cv2.GaussianBlur(rng.random((H, W, 3)).astype(np.float32), (0, 0), 1.0)
shift = lambda x, dx: np.roll(x, dx, axis=1)
ga, gb = base, shift(base, 6)
box_a, box_b = [100, 60, 120, 100], [106, 60, 120, 100]
noise = (rng.standard_normal((H, W, 3)) * .03).astype(np.float32)
noise2 = (rng.standard_normal((H, W, 3)) * .03).astype(np.float32)
pr = R.Pair(ga, gb, box_a, box_b); m = pr.mask
print('scale', round(pr.scale, 3), 'coverage', round(pr.coverage, 3))
def icr(ya, yb):
    nga, ngb = R.unobserved(ga), R.unobserved(gb)
    ea = pr.crop_a(R.unobserved(ya) - nga)[m]; eb = pr.warp_b(R.unobserved(yb) - ngb)[m]
    return ((ea - eb) ** 2).sum() / ((ea ** 2).sum() + (eb ** 2).sum())
print('same hallucination   :', round(icr(ga + noise, gb + shift(noise, 6)), 3))
print('independent          :', round(icr(ga + noise, gb + noise2), 3))
print('opposite             :', round(icr(ga + noise, gb - shift(noise, 6)), 3))
