# First experimental possession pipeline

This is an untuned, deterministic experiment, not a validated improvement over the
organiser baseline. Inference never accepts ground truth. The public sample has
only 17 possessions and helped motivate the architecture; it is not an untouched
validation set. No private truth is used or discovered.

From repository root, using the same dependencies as `analysis/requirements.txt`:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m experiments
# Alternative licensed data location:
.venv/bin/python -m experiments --data-dir /path/to/bundle
# Explicit configuration (no automatic fitting or selection):
.venv/bin/python -m experiments --config experiments/default.json
```

If matplotlib's default cache is unwritable, set `MPLCONFIGDIR` to a writable
location, e.g. `/tmp/mtah-analysis-mpl`. The run takes roughly a few minutes;
it streams tracking once, then decodes four defaults and predefined variations.
Only after generating predictions does the runner read the explicitly named
public GT and call the unmodified official `possession_metrics.evaluate`.

All outputs are under gitignored `analysis/output/experiments/`:

- `submissions/E1.csv` … `E4.csv`: exactly 199,319 ordered zero-based frame labels.
- `public_scores.csv`, `public_iou.csv`, `official_public.json`: official public
  metrics, all five spell IoUs, full metric details; fixed candidate order, no ranking.
- `full_predictions.csv`: whole-match gameplay and separate-half counts, durations,
  duration quantiles and transition rates. These are predictions, not accuracy.
- `sensitivity.csv`: default plus eight one-factor variations for every candidate,
  with full-match descriptive statistics and disagreement with its default.
- `evidence_availability.csv`: aggregate missing/rejected/available frame counts.
- `configurations.json`, `manifest.json`: exact settings, input/source/output
  SHA-256 hashes and environment versions. No timestamps or random algorithms.
- `report.md`, `comparison.png`: comparison, limitations and sensitivity summary.

To independently invoke the official validator:

```bash
.venv/bin/python evaluation/validate_submission.py analysis/output/experiments/submissions/E1.csv
# Repeat for E2, E3 and E4.
```

The runner also validates every default and sensitivity array using the official
validator, and re-reads each written default submission for validation. Repeat the
command in the same environment and compare output hashes to verify reproducibility.
PNG bytes depend on the recorded matplotlib/font environment.

## Components and reusable code

| Component | Implementation | Inputs/output |
|---|---|---|
| Tracking | `inputs.load_observations` | Reuses validated `analysis.tracking.iter_atd`; memory-only ball XY and per-team top-k distances/counts |
| Metadata | `analysis.gaps.tracking_metadata` | 105 × 68 m ATD pitch scale and home/away slot membership |
| Reliability | `observation.reliability` | Explicit finite-coordinate, motion and team-geometry availability |
| Offline continuity | `observation.continuity_baseline`, `bridge_evidence` | Past/future team evidence; never interpolates positions or speed |
| Team control | `control.proximity_evidence`, `smooth_evidence` | Collective proximity probabilities, no persistent slot IDs |
| Stoppage cues | `deadball.motion_features`, `dead_evidence` | Speed, local displacement, nearest distance, set-piece phase |
| Temporal decoder | `decoder.decode` | Deterministic H/A/D dynamic programming with explicit costs |
| Inference | `model.infer` | Observations, baseline, published half runs, tags, config, candidate; **no GT** |
| Evaluation/reporting | `__main__`, `reporting` | Separate from model logic; official scores and aggregate outputs |

`starter.load_smart_tagging.load` supplies automatic phase intervals using its
existing round(seconds × 25) alignment. `games.half_video_blocks` supplies public
half limits. Neither tags nor labels are manually corrected. The baseline file,
starter implementation and official evaluator are unchanged.

## Candidate definitions

| Candidate | Team evidence | Dead evidence | Temporal method |
|---|---|---|---|
| E1 | Published baseline H/A on reliable geometry | None | Short-gap nearest endpoint; long-gap past team carry |
| E2 | Collective proximity | None | H/A Viterbi with persistence |
| E3 | Published baseline H/A on reliable geometry | Dedicated soft cues | H/A/D Viterbi |
| E4 | Collective proximity | Dedicated soft cues | H/A/D Viterbi |

Reliability is shared by all four. E1/E2 intentionally remove the baseline's
missing-observation-to-D assumption **without supplying a replacement stoppage
detector**; therefore they predict no D. Their dead-ball recall limitation is a
negative control, not a belief that the whole match is live. E2 introduces both
collective control and persistence. E3/E4 share the same decoder/costs, so this is
an interpretable component-bundle comparison, not a fully factorial causal study.

E1 was implemented and validated first, before expansion or any public scoring.
Its final reliability mask additionally excludes rejected jumps and absent teams.

## Evidence equations and fixed defaults

All defaults are in `default.json` and validated by immutable `Config`. They were
fixed before public scoring, using round physical/time priors. They are not
calibrated probabilities and not claimed optimal.

**Reliability.** Ball speed is Euclidean displacement between adjacent observed
frames times 25, never across missing frames or half boundaries. Both ends of a
step above 50 m/s are marked unreliable. Finite ball XY is separately retained as
visibility; both teams must have at least one finite player XY for team evidence.
Unknown speed remains NaN, not zero. There is no stable player identity assumption.
Finite out-of-pitch coordinates are not automatically rejected.

For gaps of at most 2 seconds, E1 uses the nearer reliable baseline-team endpoint
(ties use the past); a short leading gap may use a future endpoint. Longer gaps
hold the past team; long leading gaps/all-unknown halves remain X. E2–E4 instead
linearly interpolate **team probabilities** between two reliable endpoints for
short gaps, with confidence 0.25. No endpoint, coordinate or motion interpolation
is used for longer gaps. Those observations remain unavailable.

**Control.** For each team, retain up to the nearest three visible players. Sum
`exp(-distance / 3 metres)` for collective support. Normalise the two team sums.
Separately normalise `exp(-nearest_distance / 3 metres)` for nearest-player
support. Mix these distributions 50/50. An absent team produces unknown evidence,
not evidence of possession by the other team. Average probabilities over ±0.5 s,
weighted by reliability and bounded by each half. Smoothing does not create
confidence in unsupported frames; only the explicit gap bridge can add it.

**Dead ball.** Within each contiguous reliable observed run, compute mean adjacent
speed and ball endpoint displacement over a centred ±0.5 s window (clipped at the
run edges). Define:

```
slow       = exp(-local_speed / 1 m/s)
stationary = exp(-local_displacement / 1 m)
distant    = 1 - exp(-nearest_player_distance / 5 m)  # zero if unavailable
phase      = 1 inside any automatic set-piece phase, otherwise 0
score      = .35*slow + .25*stationary + .15*distant + .25*phase
d          = 2*(score - .55) * ball_quality
```

`d = 0` when motion is unavailable, regardless of tags. Tags alone never establish
D without observed motion. The interval is the whole automatic phase, not a
claimed precise restart instant. Repeated/overlapping tags do not add weight.
Reported ATD ball speed is not used; speeds are recomputed from XY to make their
units and gap handling explicit. Slow live phases and moving dead balls remain
unresolved failure modes. Local net displacement can hide a curved/returning path;
the separate speed cue partly addresses that limitation.

**Decoder.** Emission scores are:

```
H = team_weight*(p_H - .5)*team_quality - d
A = team_weight*(p_A - .5)*team_quality - d
D = d
```

Defaults: `team_weight = 1`; `d = 0` with D disabled in E2. Viterbi maximises
`sum(emission / 25) - sum(transition_cost)`, so penalties compare to integrated
seconds of evidence. H↔A cost is 1.0; D entry/exit cost is 0.75; staying costs zero.
Initial H/A/D scores are symmetric; deterministic argmax ties prefer H, A, D.
No transition is permitted at a frame without team or motion evidence. Each half
resets. Leading unsupported frames and entirely unobserved halves output X;
trailing/long internal gaps hold state without claiming new evidence. These rules
can miss actual unobserved changes. Offline Viterbi/centred features use the future.

Sensitivity is predefined, one factor at a time: transition costs ×0.5/×2,
bridge 1/4 s, proximity scale 2/5 m, and dead weight ×0.5/×1.5. All combinations
with candidates are reported, including inactive-factor negative controls. These
variations are not scored on public GT or searched for a winner.

## Data, interpretation and limitations

Output labels are X outside published halves. Full-match statistics use only
those halves; raw X breaks a predicted spell and is separately counted. No
half-boundary transition is counted. H/A/D transition rates divide by total
published gameplay minutes, not just live time. Predicted counts/durations do not
become ground truth merely because they look plausible.

Distances use metadata's 105 × 68 m, while the published baseline used the
separately documented 105.3 × 68 m convention. Duplicate/missing tracks can bias
collective support; no algorithm identifies controlled possession from geometry
alone. Jump rejection can reject true fast motion or fail to catch plausible
tracking errors. Reliability is a heuristic, not an estimated confidence.

No raw coordinates, player paths, frame-level evidence or video frames are
exported or cached. Generated submission labels and aggregate diagnostics stay
in the existing gitignored output directory. All synthetic tests use invented
observations. Full-match performance is unknown without independent ground truth.
