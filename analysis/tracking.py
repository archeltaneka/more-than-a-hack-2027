"""Streaming ATD visibility. Coordinates are discarded and never exported."""
import numpy as np


def load_visibility(path, n_frames):
    """Validate exactly ordered 1..N ATD IDs; return video-indexed finite XY flags.

    NaN/partial-NaN/inf coordinates mean unavailable, not a dead ball. Missing or
    duplicated ATD rows are structural errors, distinct from absent detections.
    """
    visible = np.zeros(n_frames, dtype=bool)
    for frame, _, ball in iter_atd(path, n_frames):
        visible[frame] = np.isfinite(ball).all()
    return visible


def iter_atd(path, n_frames):
    """Yield (zero-based frame, raw player field, ball XY) without persisting data."""
    count = 0
    with open(path) as handle:
        for count, line in enumerate(handle, 1):
            parts = line.strip().split(':')
            if len(parts) != 3 or int(parts[0]) != count or count > n_frames:
                raise ValueError('invalid ATD frame record at line {}'.format(count))
            ball = parts[2].split(',')
            if len(ball) != 3:
                raise ValueError('invalid ball fields at line {}'.format(count))
            yield count-1, parts[1], np.array([float(ball[0]), float(ball[1])])
    if count != n_frames:
        raise ValueError('ATD has {} records; expected {}'.format(count, n_frames))


def visibility_summary(visible, runs):
    total = missing = longest = gaps = 0
    for a, b in runs:
        absent = ~visible[a:b]
        edges = np.diff(np.r_[0, absent.astype(int), 0])
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        total += b-a
        missing += int(absent.sum())
        gaps += len(lengths)
        longest = max(longest, int(max(lengths, default=0)))
    return dict(frames=total, missing_frames=missing, missing_gaps=gaps,
                longest_missing_gap=longest,
                visible_fraction=(total-missing)/total if total else None)
