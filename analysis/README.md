# Public possession-boundary diagnostics

Run from repository root, with Python 3.11 and the licensed tactical bundle under
`data/` (or set `MTAH_DATA_DIR`). Only the named public GT and `baseline_smooth50`
are opened. The baseline, evaluator and input files are read-only.

```bash
python3.11 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r analysis/requirements.txt
.venv-analysis/bin/python -m unittest discover -s tests -v
.venv-analysis/bin/python -m analysis
# Optional alternate licensed bundle root:
.venv-analysis/bin/python -m analysis --data-dir /path/to/bundle
```

The existing project environment also works after installing matplotlib, NumPy
and pandas. `analysis/requirements.txt` pins the environment used for verification.
If the default matplotlib cache directory is read-only, set `MPLCONFIGDIR` to a
writable temporary directory. No kloppy, video decoder or notebook is required.

Outputs always go to **`analysis/output/`**, which is gitignored:

| File | Contents |
|---|---|
| `gt.csv`, `pred.csv` | H/A spells, stable IDs, runs/halves, crop censoring, split/merge candidate flags, aggregate visibility |
| `matches.csv` | Matched and unmatched rows for all five IoUs, joined boundaries/labels, offsets and non-exclusive categories |
| `overlaps.csv` | Every positive same-team overlap; links split/merge evidence to spell IDs |
| `scores.csv` | TP/FP/FN and spellF1 at each IoU |
| `transitions.csv` | Matched/missed/spurious transitions, label changes, offsets and ±25-frame window visibility |
| `visibility_summary.json` | Public-only counts, gap summary and false-stoppage association |
| `timeline.png` | GT, raw/filled predictions, one-second visibility aggregates and suspicious boundaries |
| `report.md` | Measured findings, evidence, definitions and limitations |
| `manifest.json` | Input/code hashes, runtime versions and diagnostic conventions |

Intervals are zero-based **[start, stop)**. Offsets are prediction minus GT in
frames; divide by 25 for seconds. Split/merge flags use all positive same-team
intersections; even a tiny overlap can create a candidate. They may coexist and
are not an exhaustive classification. Missed transitions mean no destination-typed
match within ±25 frames, not necessarily no eventual transition. D is excluded
from spells but retained in transitions. X handling matches the official evaluator.

The official greedy matcher exposes counts/GT flags, not pair identities.
`core._pairs` recovers pairs using its exact deterministic ordering and checks
both official outputs. The CLI additionally checks the full official `evaluate`
result at every IoU and the mean. Tests cover randomised official consistency.
`labelled_runs` explicitly intersects valid label runs with half boundaries;
GT-X gaps also break runs. Crop edges are marked as censored.

Visibility requires finite ball x **and** y. ATD records must be exactly ordered
1..199319; subtracting one maps to video frames. Missing rows are errors; NaN ball
coordinates are missing observations. Raw coordinates are discarded on parsing.
Exports never contain player data, coordinates or a frame-level visibility table.
Interval aggregates may cover a one-frame spell; they are diagnostic summaries,
not a reconstructed tracking export. Keep generated outputs local.

For reproducibility, rerunning overwrites the fixed outputs deterministically.
Compare SHA-256 hashes of output files after two runs in the same environment.
PNG byte equality additionally depends on rendering/fonts; use the recorded
versions and pinned environment. The public sample is descriptive, not a tuning
set or full-match ranking estimate. No parameters are selected from its results.

Reusable entry points: `core.load_labels`, `core.labelled_runs`, `core.analyse`,
`tracking.load_visibility`, `tracking.visibility_summary`, and `report.enrich`.
The CLI deliberately has no arbitrary GT path or model-sweep option.

## Issue #2: tracking gaps and game-state reliability

```bash
.venv-analysis/bin/python -m analysis --gaps
.venv-analysis/bin/python -m unittest discover -s tests -v
```

`--gaps` also loads tactical ATD metadata and Smart Tagging. It regenerates the
original diagnostics and writes additional artifacts in `analysis/output/gaps/`:

| File | Contents |
|---|---|
| `public_gaps.csv` | All missing-ball intervals in the public window, GT/raw baseline state counts, endpoint scalar geometry, nearest teams, observation ages, automatic event context and baseline transitions |
| `false_stoppages.csv`, `missed_stoppages.csv` | Public error episodes with the same context; false = GT H/A, prediction D; missed = GT D, prediction H/A |
| `dead_ball_intervals.csv` | Every genuine public D interval, including visible-ball intervals |
| `public_visibility_state.csv`, `public_visibility_prediction.csv` | Counts and within-state visibility fractions; true labels and predictions kept separate |
| `public_context_distributions.csv` | Nearest-player distance quantiles and set-piece context coverage by true state |
| `fragmentation_context.csv` | Split/merge flags reused from the original analysis, missing frames, internal transitions and nearest-team switches for every public GT possession |
| `scope_distributions.csv` | Gap duration, displacement, nearest-distance and observation-age distributions for public scope, full video, gameplay halves and each outer/half region |
| `full_visibility_prediction.csv`, `full_gameplay_visibility_prediction.csv` | Descriptive baseline/visibility contingency tables, **not full-match accuracy** |
| `gap_distributions.png`, `gap_examples.png` | Distributions and public false/missed stoppage examples; no coordinate plots |
| `report.md`, `manifest.json` | Findings, three testable hypotheses, limitations and reproducibility metadata |

`gaps.load_geometry` stores ball XY **in memory only**, while discarding player
coordinates after computing per-frame nearest distance/team. `gap_features` has
no GT argument and is used unchanged on public and full-match scopes.
`public_context` joins public labels afterward. `safe_gap_table` uses a positive
column allowlist: endpoint positions, vectors and arbitrary extra fields cannot
become CSV columns. Full-match gap rows remain in memory; only distributions are
exported. No tracked slot identity is retained or compared across time.

The ATD XML specifies **105 × 68 m**. This analysis scales normalised x/y by those
axes; it does not use the separate 105.3 m corrected-tracking convention in
`games.py`. Metadata team order is home then away, as in the official baseline.
Ties use slot order; nearest team is an observation, not controlled possession.

All times are video-based. Stops are exclusive. `age_start_frames` and
`age_end_frames` are ages at the gap's first and final missing frames; divide by
25 for seconds. `next_observation_delay_frames` counts from exclusive stop to
next observation. No observation is borrowed across a public/half/region boundary.
Boundary endpoints, positions and derived distances remain unknown when unavailable.
`left_censored`/`right_censored` flag contact with scope edges. For general error
intervals, ages reference the observation before interval start, even if the ball
is observed inside the interval. Next positions are retrospective context.

Near-change flags use **±25 frames around direct H↔A changes**, excluding D
restarts. Nearby event fields describe **set-piece phases overlapping ±125-frame
context**, clipped to scope. Counts are phases, not confirmed restart events.
Blank context means none found; `N/A` team remains unknown. Baseline transitions
count destination frames inside the interval, including its first frame when its
predecessor is available inside the scope. Multiple categories may coexist.

The ±1 s/±5 s context windows are fixed descriptive conventions, not thresholds
chosen using public labels. No model is fitted or altered. Baseline `smooth50`
selects the most frequent H/A observation in ±50 frames and emits D only when
that window has **no** H/A observations; missing observations do not vote for D.
Smart Tagging and geometric proximity are noisy covariates, not truth. The public
sample supports hypotheses only; no full-match ground truth or accuracy is used.
