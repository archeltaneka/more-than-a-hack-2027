"""Read licensed observations in memory; never writes coordinates or caches."""
import numpy as np
from analysis.tracking import iter_atd


def load_observations(path, n_frames, teams, pitch, top_k):
    xy = np.full((n_frames, 2), np.nan)
    distances = np.full((n_frames, 2, top_k), np.nan)
    counts = np.zeros((n_frames, 2), int)
    for frame, raw_players, ball in iter_atd(path, n_frames):
        players = np.fromstring(raw_players.replace(';', ','), sep=',')
        if players.size != len(teams)*3:
            raise ValueError('player slot count mismatch at frame {}'.format(frame))
        players = players.reshape(-1, 3)[:, :2]
        finite = np.isfinite(players).all(axis=1)
        for code, team in enumerate(('H', 'A')):
            mask = finite & (teams == team)
            counts[frame, code] = int(mask.sum())
            if np.isfinite(ball).all():
                xy[frame] = ball*np.asarray(pitch)
                d = np.sort(np.linalg.norm((players[mask]-ball)*np.asarray(pitch), axis=1))[:top_k]
                distances[frame, code, :len(d)] = d
    return dict(ball_m=xy, distances=distances, team_counts=counts)
