"""Tracking-gap features. Positions live in memory only; export via safe_gap_table."""
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from evaluation import possession_metrics as pm
from .tracking import iter_atd

EVENT_MARGIN = 5 * pm.FPS
BOUNDARY_MARGIN = pm.FPS


def tracking_metadata(path):
    """ATD normalised x/y scale and slot team membership, not player identities."""
    root = ET.parse(path).getroot()
    field = root.find('.//FieldSize')
    pitch = (float(field.findtext('Width')), float(field.findtext('Height')))
    if not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError('invalid ATD pitch dimensions')
    teams = [t.get('id') for t in root.iter('Team')]
    if len(teams) != 2 or len(set(teams)) != 2:
        raise ValueError('expected two distinct teams in home/away order')
    mapping = dict(zip(teams, ['H', 'A']))
    slots = [p.get('teamId') for p in root.iter('Player')]
    if not slots or any(t not in mapping for t in slots):
        raise ValueError('unknown or absent slot team membership')
    return pitch, np.array([mapping[t] for t in slots])


def nearest_player(ball, players, teams, pitch):
    """Nearest finite XY observation at one instant; metre scaling is anisotropic.

    Slot order breaks exact distance ties, like the baseline. No slot ID survives.
    """
    ball, players = np.asarray(ball), np.asarray(players)
    valid = np.isfinite(players).all(axis=1)
    if not np.isfinite(ball).all() or not valid.any():
        return np.nan, ''
    distances = np.linalg.norm((players[valid] - ball) * np.asarray(pitch), axis=1)
    best = int(np.argmin(distances))
    return float(distances[best]), str(teams[valid][best])


def load_geometry(path, n_frames, teams, pitch):
    """Memory-only XY and scalar nearest-player context; never serialize this dict."""
    ball_m = np.full((n_frames, 2), np.nan)
    distance = np.full(n_frames, np.nan)
    nearest_team = np.full(n_frames, '', dtype='<U1')
    for frame, raw_players, ball in iter_atd(path, n_frames):
        players = np.fromstring(raw_players.replace(';', ','), sep=',')
        if players.size != len(teams)*3:
            raise ValueError('invalid player slot count at video frame {}'.format(frame))
        if np.isfinite(ball).all():
            ball_m[frame] = ball * np.asarray(pitch)
            distance[frame], nearest_team[frame] = nearest_player(
                ball, players.reshape(-1, 3)[:, :2], teams, pitch)
    return dict(ball_m=ball_m, visible=np.isfinite(ball_m).all(axis=1),
                nearest_m=distance, nearest_team=nearest_team)


def contiguous_intervals(mask, runs):
    """(run index, start, stop) true intervals, clipped independently to each run."""
    out = []
    for run, (lo, hi) in enumerate(runs):
        edges = np.diff(np.r_[0, np.asarray(mask[lo:hi], dtype=np.int8), 0])
        out.extend((run, lo+int(a), lo+int(b)) for a, b in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    return out


def event_context(events, start, stop, run):
    """Automatic set-piece phase overlaps within a fixed ±5s, clipped to run.

    These are contextual intervals, NOT precise or validated restart instants.
    """
    lo, hi = max(run[0], start-EVENT_MARGIN), min(run[1], stop+EVENT_MARGIN)
    near = [e for e in events if (e['code'] or '').strip().lower() == 'set pieces'
            and e['start_frame'] < hi and e['end_frame'] > lo]
    distances = [max(0, start-e['end_frame'], e['start_frame']-stop) for e in near]
    return dict(nearby_event_count=len(near),
                nearby_event_types=';'.join(sorted({e['tags'].get('Type', '') for e in near})),
                nearby_event_teams=';'.join(sorted({e['tags'].get('Team', '') for e in near})),
                nearest_event_distance_frames=min(distances) if distances else np.nan)


def state_distribution(signal, start, stop, prefix):
    return {prefix+'_'+label+'_frames': int((signal[start:stop] == code).sum())
            for code, label in enumerate(pm.LABELS)}


def interval_features(data, pred, intervals, runs, events):
    """Context for intervals; endpoints are nearest visible observations OUTSIDE.

    Endpoint coordinates are intentionally memory-only. At clipped boundaries we
    do not search beyond the scope, so ages/positions can be unknown, not zero.
    """
    rows = []
    visible_by_run = [np.flatnonzero(data['visible'][a:b])+a for a, b in runs]
    transitions_by_run = [pm.transitions(pred, [run]) for run in runs]
    for index, (run, a, b) in enumerate(intervals):
        lo, hi = runs[run]
        observed = visible_by_run[run]
        before = observed[observed < a]
        after = observed[observed >= b]
        last = int(before[-1]) if len(before) else None
        following = int(after[0]) if len(after) else None
        last_xy = data['ball_m'][last].copy() if last is not None else None
        next_xy = data['ball_m'][following].copy() if following is not None else None
        transitions = [t for t in transitions_by_run[run] if a <= t[0] < b]
        row = dict(id=index, run=run, start=a, stop=b, duration_frames=b-a,
                   duration_s=(b-a)/pm.FPS, left_censored=a == lo, right_censored=b == hi,
                   last_ball_m=last_xy, next_ball_m=next_xy,
                   age_start_frames=a-last if last is not None else np.nan,
                   age_end_frames=b-1-last if last is not None else np.nan,
                   next_observation_delay_frames=following-b if following is not None else np.nan,
                   displacement_m=float(np.linalg.norm(next_xy-last_xy))
                   if last_xy is not None and next_xy is not None else np.nan,
                   nearest_before_m=data['nearest_m'][last] if last is not None else np.nan,
                   nearest_after_m=data['nearest_m'][following] if following is not None else np.nan,
                   team_before=str(data['nearest_team'][last]) if last is not None else '',
                   team_after=str(data['nearest_team'][following]) if following is not None else '',
                   baseline_transitions_inside=len(transitions),
                   baseline_transition_changes=';'.join(pm.LABELS[t[1]]+'->'+pm.LABELS[t[2]]
                                                        for t in transitions),
                   visible_frames=int(data['visible'][a:b].sum()),
                   visible_fraction=float(data['visible'][a:b].mean()))
        row.update(state_distribution(pred, a, b, 'pred'))
        row.update(event_context(events, a, b, (lo, hi)))
        rows.append(row)
    return rows


def gap_features(data, pred, runs, events):
    """Same GT-free feature extraction for public and full-match scopes."""
    return interval_features(data, pred, contiguous_intervals(~data['visible'], runs), runs, events)


def public_context(rows, gt, data, runs):
    """Return annotated copies; never apply this to full-match gaps."""
    out = []
    for source in rows:
        row = source.copy()
        a, b, run = row['start'], row['stop'], runs[row['run']]
        row.update(state_distribution(gt, a, b, 'gt'))
        changes = [f for f, old, new in pm.transitions(gt, [run])
                   if old in pm.POSSESSION_LABELS and new in pm.POSSESSION_LABELS]
        near = [f for f in changes if a-BOUNDARY_MARGIN <= f < b+BOUNDARY_MARGIN]
        row.update(missing_during_live=bool(np.any(
                       np.isin(gt[a:b], pm.POSSESSION_LABELS) & ~data['visible'][a:b])),
                   missing_during_dead=bool(np.any((gt[a:b] == pm.D) & ~data['visible'][a:b])),
                   near_possession_change=bool(near), nearby_possession_changes=len(near))
        out.append(row)
    return out


# Positive export allowlist: adding a private feature never adds a CSV column.
GAP_COLUMNS = [
    'id', 'run', 'start', 'stop', 'duration_frames', 'duration_s', 'left_censored',
    'right_censored', 'age_start_frames', 'age_end_frames', 'next_observation_delay_frames',
    'displacement_m', 'nearest_before_m', 'nearest_after_m', 'team_before', 'team_after',
    'baseline_transitions_inside', 'baseline_transition_changes', 'visible_frames', 'visible_fraction',
    'nearby_event_count', 'nearby_event_types', 'nearby_event_teams', 'nearest_event_distance_frames',
] + ['pred_'+label+'_frames' for label in pm.LABELS]
PUBLIC_COLUMNS = ['gt_'+label+'_frames' for label in pm.LABELS] + [
    'missing_during_live', 'missing_during_dead', 'near_possession_change', 'nearby_possession_changes']


def safe_gap_table(rows, public=False):
    """No endpoint positions, vectors, slot IDs or hidden columns can be exported."""
    return pd.DataFrame(rows, columns=GAP_COLUMNS + (PUBLIC_COLUMNS if public else []))
