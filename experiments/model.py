"""Pure deterministic candidate construction. No GT or reporting dependencies."""
import numpy as np
from .observation import reliability, continuity_baseline, bridge_evidence
from .control import proximity_evidence, smooth_evidence
from .deadball import motion_features, set_piece_mask, dead_evidence
from .decoder import decode

CANDIDATES = ('E1', 'E2', 'E3', 'E4')


def infer(observations, baseline, runs, events, config, candidate):
    if candidate not in CANDIDATES:
        raise ValueError('unknown candidate '+candidate)
    q = reliability(observations['ball_m'], observations['team_counts'], runs,
                    config.fps, config.jump_limit_mps)
    bridge = round(config.bridge_seconds*config.fps)
    e1 = continuity_baseline(baseline, q['team_quality'] > 0, runs, bridge)
    if candidate == 'E1':
        return e1, dict(reliability=q)
    collective = candidate in ('E2', 'E4')
    include_dead = candidate in ('E3', 'E4')
    if collective:
        p = proximity_evidence(observations['distances'], config.proximity_m, config.nearest_weight)
        p = smooth_evidence(p, q['team_quality'], runs, round(config.smoothing_seconds*config.fps))
    else:
        # Use the unchanged baseline's smoothed nearest-team prediction where available.
        p = np.full((len(baseline), 2), np.nan)
        for code in (0, 1):
            p[baseline == code] = np.eye(2)[code]
    tq = q['team_quality']*np.isfinite(p).all(axis=1)
    p, tq = bridge_evidence(p, tq, runs, bridge, config.bridge_weight)
    motion, displacement = motion_features(observations['ball_m'], q['speed_mps'],
                                            q['ball_quality'], runs,
                                            max(1, round(config.motion_seconds*config.fps)))
    nearest = np.where(np.isfinite(observations['distances']), observations['distances'], np.inf).min(axis=(1, 2))
    nearest[~np.isfinite(nearest)] = np.nan
    tag = set_piece_mask(events, len(baseline), runs)
    dead = dead_evidence(motion, displacement, nearest, tag, q['ball_quality'], config)
    d = dead['logit'] if include_dead else np.zeros(len(baseline))
    scores = np.zeros((len(baseline), 3))
    scores[:, :2] = config.team_weight*(np.nan_to_num(p, nan=.5)-.5)*tq[:, None]-d[:, None]
    scores[:, 2] = d
    supported = (tq > 0) | (dead['available'] if include_dead else False)
    labels = decode(scores, runs, supported, config.fps, config.team_change_penalty,
                    config.dead_change_penalty, allow_dead=include_dead)
    return labels, dict(reliability=q, dead=dead, team_quality=tq)
