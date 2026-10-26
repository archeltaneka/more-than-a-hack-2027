"""Collective team proximity, without persistent slot/player identities."""
import numpy as np


def proximity_evidence(distances, scale_m, nearest_weight):
    """Mixture of nearest-player and top-k soft team support, each normalised.

    distances: (frames, teams H/A, k), NaN padded. Each player's contribution is
    exp(-distance/scale), bounded by one; absent team => unknown, not zero control.
    """
    finite = np.isfinite(distances)
    available = finite.any(axis=2).all(axis=1)
    safe = np.where(finite, distances, np.inf)
    nearest = safe.min(axis=2)
    # Subtract common minimum to avoid underflow when the ball is far from everyone.
    minimum = np.min(nearest, axis=1, keepdims=True)
    minimum = np.where(np.isfinite(minimum), minimum, 0)
    support = np.exp(-(safe-minimum[:, :, None])/scale_m).sum(axis=2)
    nearest_support = np.exp(-(nearest-minimum)/scale_m)
    with np.errstate(invalid='ignore', divide='ignore'):
        collective = support/support.sum(axis=1, keepdims=True)
        nearest_prob = nearest_support/nearest_support.sum(axis=1, keepdims=True)
    out = nearest_weight*nearest_prob+(1-nearest_weight)*collective
    out[~available] = np.nan
    return out


def smooth_evidence(p, quality, runs, radius):
    """Weighted centred evidence mean; no half crossing and no invented confidence."""
    out = np.full_like(p, np.nan)
    for lo, hi in runs:
        q = quality[lo:hi]*np.isfinite(p[lo:hi]).all(axis=1)
        weighted = np.nan_to_num(p[lo:hi])*q[:, None]
        cs = np.vstack((np.zeros((1, 2)), weighted.cumsum(axis=0)))
        cq = np.r_[0., q.cumsum()]
        index = np.arange(hi-lo)
        left, right = np.maximum(0, index-radius), np.minimum(hi-lo, index+radius+1)
        denominator = cq[right]-cq[left]
        with np.errstate(invalid='ignore', divide='ignore'):
            out[lo:hi] = (cs[right]-cs[left])/denominator[:, None]
    return out
