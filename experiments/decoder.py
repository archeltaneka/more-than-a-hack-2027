"""Three-state Viterbi with explicit integrated evidence and transition costs."""
import numpy as np


def decode(scores, runs, supported, fps, team_penalty, dead_penalty, allow_dead=True):
    """Maximise sum(scores/fps) minus transition penalties independently per half.

    Unknown leading stretches remain X. No transition is permitted on unsupported
    frames. Ties prefer H, then A, then D via numpy argmax, deterministically.
    Entirely unobserved halves are X rather than an arbitrary prior prediction.
    """
    if scores.shape != (len(supported), 3) or not np.isfinite(scores).all():
        raise ValueError('expected finite Nx3 scores')
    n_states = 3 if allow_dead else 2
    costs = np.array([[0, team_penalty, dead_penalty],
                      [team_penalty, 0, dead_penalty],
                      [dead_penalty, dead_penalty, 0]])[:n_states, :n_states]
    out = np.full(len(scores), 3, dtype=np.int8)
    for lo, hi in runs:
        observed = np.flatnonzero(supported[lo:hi])
        if not len(observed):
            continue
        start = lo+int(observed[0])
        back = np.zeros((hi-start, n_states), dtype=np.int8)
        value = scores[start, :n_states]/fps
        for f in range(start+1, hi):
            candidate = value[:, None]-costs
            if not supported[f]:
                candidate[~np.eye(n_states, dtype=bool)] = -np.inf
            back[f-start] = np.argmax(candidate, axis=0)
            value = np.max(candidate, axis=0)+scores[f, :n_states]/fps
        state = int(np.argmax(value))
        for f in range(hi-1, start-1, -1):
            out[f] = state
            state = int(back[f-start, state])
    return out
