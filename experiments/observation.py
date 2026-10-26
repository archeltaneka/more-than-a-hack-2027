"""Reliability and offline gap context. Missing geometry is never dead-ball evidence."""
import numpy as np


def continuity_baseline(baseline, reliable, runs, bridge_frames):
    """E1: preserve supported baseline H/A, replace unsupported periods with continuity.

    Short gaps use the nearer endpoint (ties prefer past). Long gaps carry past
    team indefinitely, explicitly without renewed evidence. Long leading gaps and
    wholly unobserved halves remain X. No information crosses a half boundary.
    """
    out = np.full(len(baseline), 3, dtype=np.int8)
    for lo, hi in runs:
        good = np.flatnonzero(reliable[lo:hi] & np.isin(baseline[lo:hi], [0, 1]))+lo
        if not len(good):
            continue
        out[good] = baseline[good]
        endpoints = np.r_[lo-1, good, hi]
        for left, right in zip(endpoints[:-1], endpoints[1:]):
            a, b = left+1, right
            if a == b:
                continue
            has_left, has_right = left >= lo, right < hi
            if b-a <= bridge_frames and has_right:
                if has_left:
                    frames = np.arange(a, b)
                    out[a:b] = np.where(frames-left <= right-frames, baseline[left], baseline[right])
                else:
                    out[a:b] = baseline[right]
            elif has_left:
                out[a:b] = baseline[left]
    return out


def reliability(ball_m, team_counts, runs, fps, jump_limit_mps):
    """Finite ball XY and both teams observed; reject both ends of implausible jumps.

    Speed is derived only between adjacent observed frames within the same half.
    A missing measurement is NaN, never a stationary ball (zero speed).
    """
    visible = np.isfinite(ball_m).all(axis=1)
    speed = np.full(len(ball_m), np.nan)
    quality = np.zeros(len(ball_m))
    for lo, hi in runs:
        quality[lo:hi] = visible[lo:hi]
        valid = visible[lo+1:hi] & visible[lo:hi-1]
        steps = np.linalg.norm(np.diff(ball_m[lo:hi], axis=0), axis=1)*fps
        speed[lo+1:hi] = np.where(valid, steps, np.nan)
        bad = np.flatnonzero(valid & (steps > jump_limit_mps))+lo
        quality[bad] = quality[bad+1] = 0
    return dict(visible=visible, ball_quality=quality, speed_mps=speed,
                team_quality=quality*(np.asarray(team_counts) > 0).all(axis=1))


def bridge_evidence(probabilities, quality, runs, max_gap, weight):
    """Offline interpolation of TEAM EVIDENCE, never coordinates or motion.

    Both endpoints required. Long/edge gaps retain zero observational confidence.
    """
    p, q = probabilities.copy(), quality.copy()
    for lo, hi in runs:
        known = np.flatnonzero((q[lo:hi] > 0) & np.isfinite(p[lo:hi]).all(axis=1))+lo
        for left, right in zip(known[:-1], known[1:]):
            size = right-left-1
            if 0 < size <= max_gap:
                alpha = np.arange(1, size+1)/(size+1)
                p[left+1:right] = p[left]*(1-alpha[:, None])+p[right]*alpha[:, None]
                q[left+1:right] = weight*min(q[left], q[right])
    return p, q
