"""Soft stoppage cues. Neither missing coordinates nor tags alone imply D."""
import numpy as np


def motion_features(ball_m, speed, quality, runs, radius):
    """Local motion only inside contiguous reliable observations, never across gaps."""
    local_speed = np.full(len(ball_m), np.nan)
    displacement = np.full(len(ball_m), np.nan)
    for lo, hi in runs:
        valid = quality[lo:hi] > 0
        edges = np.diff(np.r_[0, valid.astype(int), 0])
        for a, b in zip(np.flatnonzero(edges == 1)+lo, np.flatnonzero(edges == -1)+lo):
            if b-a < 2:
                continue
            indices = np.arange(a, b)
            left, right = np.maximum(a, indices-radius), np.minimum(b-1, indices+radius)
            displacement[a:b] = np.linalg.norm(ball_m[right]-ball_m[left], axis=1)
            # Include only adjacent differences internal to this reliable run.
            steps = speed[a:b].copy(); steps[0] = np.nan
            values = np.r_[0., np.nan_to_num(steps).cumsum()]
            counts = np.r_[0, np.isfinite(steps).cumsum()]
            count = counts[right-a+1]-counts[left-a+1]
            with np.errstate(invalid='ignore', divide='ignore'):
                local_speed[a:b] = (values[right-a+1]-values[left-a+1])/count
    return local_speed, displacement


def set_piece_mask(events, n_frames, runs):
    """Union of actual automatic phases, no GT alignment or asserted restart instant."""
    mask = np.zeros(n_frames)
    for event in events:
        if (event['code'] or '').strip().lower() != 'set pieces':
            continue
        for lo, hi in runs:
            a, b = max(lo, event['start_frame']), min(hi, event['end_frame'])
            if a < b:
                mask[a:b] = 1
    return mask


def dead_evidence(speed, displacement, nearest_m, tag, quality, config):
    available = (quality > 0) & np.isfinite(speed) & np.isfinite(displacement)
    slow = np.exp(-np.nan_to_num(speed, nan=np.inf)/config.slow_mps)
    stationary = np.exp(-np.nan_to_num(displacement, nan=np.inf)/config.displacement_m)
    distant = np.where(np.isfinite(nearest_m), 1-np.exp(-nearest_m/config.distant_player_m), 0)
    score = (config.speed_weight*slow + config.displacement_weight*stationary +
             config.distance_weight*distant + config.tag_weight*tag)
    logit = np.where(available, config.dead_weight*(score-config.dead_reference)*quality, 0)
    return dict(logit=logit, available=available, score=np.where(available, score, np.nan))
