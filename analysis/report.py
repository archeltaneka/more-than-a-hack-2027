"""Aggregate exports and static timeline; no frame-level tracking exports."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluation import possession_metrics as pm
from .tracking import visibility_summary


def enrich(result, gt, pred, visible, runs, halves):
    """Attach visibility summaries and evidence to spell and transition rows."""
    for name in ('gt', 'pred'):
        table = result[name]
        for column in ('half', 'visible_fraction', 'raw_abstention_frames'):
            table[column] = pd.Series(index=table.index, dtype=float)
        for row in table.itertuples():
            table.at[row.Index, 'half'] = next(h for h, a, b in halves if a <= row.start <= b)
            table.at[row.Index, 'visible_fraction'] = float(visible[row.start:row.stop].mean())
            table.at[row.Index, 'raw_abstention_frames'] = int((pred[row.start:row.stop] == pm.X).sum())
    matches = result['matches']
    for name in ('gt', 'pred'):
        cols = ['id', 'run', 'half', 'start', 'stop', 'label', 'left_censored', 'right_censored',
                'visible_fraction']
        extra = result[name][cols].rename(columns={c: name+'_'+c for c in cols})
        matches = matches.merge(extra, on=name+'_id', how='left', validate='many_to_one')
    result['matches'] = matches
    transitions = result['transitions']
    for column in ('window_start', 'window_stop', 'visible_fraction'):
        transitions[column] = pd.Series(index=transitions.index, dtype=float)
    transitions['suspicious'] = False
    for row in transitions.itertuples():
        f = int(row.gt_frame if pd.notna(row.gt_frame) else row.pred_frame)
        lo, hi = runs[row.run]
        a, b = max(lo, f-25), min(hi, f+26)
        transitions.at[row.Index, 'window_start'] = a
        transitions.at[row.Index, 'window_stop'] = b
        transitions.at[row.Index, 'visible_fraction'] = float(visible[a:b].mean())
        transitions.at[row.Index, 'suspicious'] = row.status != 'matched' or row.offset != 0
    # Coarse one-second bins for the plot: only aggregate visibility is retained.
    bins = []
    for run, (lo, hi) in enumerate(runs):
        for a in range(lo, hi, pm.FPS):
            b = min(hi, a+pm.FPS)
            bins.append(dict(run=run, start=a, stop=b, visible_fraction=float(visible[a:b].mean())))
    result['visibility_bins'] = pd.DataFrame(bins)
    # A false stoppage is a filled predicted D interval inside GT H/A.
    false_d = (result['filled'] == pm.D) & np.isin(gt, [pm.H, pm.A])
    summary = visibility_summary(visible, runs)
    summary['false_stoppage_frames'] = int(false_d.sum())
    summary['false_stoppage_missing_ball_frames'] = int((false_d & ~visible).sum())
    summary['alive_frames'] = int(np.isin(gt, [pm.H, pm.A]).sum())
    summary['alive_missing_ball_frames'] = int((np.isin(gt, [pm.H, pm.A]) & ~visible).sum())
    return summary


def timeline(path, gt, pred, result, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {pm.H: '#2479b4', pm.A: '#e68a25', pm.D: '#bec4cc', pm.X: '#ffffff'}
    # One panel per minute; labels remain legible on the five-minute public sample.
    panels = [(a, min(a + 60*pm.FPS, hi)) for lo, hi in runs
              for a in range(lo, hi, 60*pm.FPS)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 2.3*len(panels)), squeeze=False)
    for ax, (a, b) in zip(axes[:, 0], panels):
        for y, signal in ((3, gt), (2, pred), (1, result['filled'])):
            for start, stop, code in pm.segments(signal, [(a, b)]):
                ax.broken_barh([(start/pm.FPS, (stop-start)/pm.FPS)], (y-.32, .64),
                               facecolors=colors[code], edgecolors='none')
        bins = result['visibility_bins']
        bins = bins[(bins.start < b) & (bins.stop > a)]
        ax.bar(bins.start/pm.FPS, bins.visible_fraction*.64,
               width=(bins.stop-bins.start)/pm.FPS, align='edge', bottom=-.32, color='#248b68')
        for row in result['transitions'].itertuples():
            if not row.suspicious:
                continue
            for f, y in ((row.gt_frame, 3), (row.pred_frame, 1)):
                if pd.notna(f) and a <= f < b:
                    ax.plot(f/pm.FPS, y+.4, marker='v', color='#c12443', markersize=4)
        ax.set(xlim=(a/pm.FPS, b/pm.FPS), ylim=(-.5, 3.7),
               yticks=[0, 1, 2, 3], yticklabels=['Ball visible¹', 'Pred filled', 'Pred raw', 'GT'])
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('Tactical video time (seconds from frame 0)')
    fig.suptitle('Public possession diagnostics · baseline_smooth50\n'
                 'Red markers: missed/spurious or shifted transitions (25-frame matching)', fontsize=13)
    fig.legend(handles=[Patch(color=colors[k], label=pm.LABELS[k]) for k in colors],
               loc='lower center', ncol=4, bbox_to_anchor=(.5, .01))
    fig.text(.01, .015, '¹ Finite ball XY fraction per 1-second bin; height 0–100%.', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(path, dpi=140, metadata={'Software': 'possession-analysis'})
    plt.close(fig)


def write_outputs(out, result, gt, pred, runs, summary, manifest):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Visibility bins are only plotted; no tracking-derived frame-level table is written.
    for key in ('gt', 'pred', 'matches', 'scores', 'overlaps', 'transitions'):
        result[key].to_csv(out / (key+'.csv'), index=False, float_format='%.10g', lineterminator='\n')
    (out / 'visibility_summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    timeline(out / 'timeline.png', gt, pred, result, runs)
    matched = result['matches'].query('threshold == 0.5 and status == "matched"')
    shifted = matched[(matched.start_offset != 0) | (matched.stop_offset != 0)]
    trans = result['transitions']
    counts = trans.status.value_counts()
    split = result['gt'].query('candidate_split == True')
    merge = result['pred'].query('candidate_merge == True')
    scorelines = ['| IoU | TP | FP | FN | F1 |', '|---|---|---|---|---|']
    for row in result['scores'].itertuples():
        scorelines.append('| {:.1f} | {} | {} | {} | {:.6f} |'.format(
            row.threshold, row.tp, row.fp, row.fn, row.f1))
    evidence = []
    for row in split.sort_values('duration', ascending=False).head(3).itertuples():
        links = result['overlaps'].query('gt_id == @row.id').sort_values('pred_id')
        evidence.append('- GT {} ({} [{}, {})): overlaps predicted spells {} (candidate split).'.format(
            row.id, row.label, row.start, row.stop, ', '.join(str(i) for i in links.pred_id)))
    if not evidence:
        evidence.append('- No same-team overlap split candidates.')
    false_d = summary['false_stoppage_frames']
    percent = summary['false_stoppage_missing_ball_frames']/false_d*100 if false_d else 0
    text = '''# Public baseline possession diagnostics

Fixed input: `baseline_smooth50`; public GT only. No labels corrected and no model tuning.
Analysed {frames} frames in {nruns} labelled run(s); {ng} GT possessions, {npred} predicted.
**spellF1 = {score:.6f}**, verified against official `evaluate` at every threshold.

{scoretable}

## Observed patterns

- Fragmentation: {nsplit}/{ng} GT possessions overlap multiple same-team predictions;
  {nmerge} predicted spells overlap multiple same-team GT possessions (merge candidates).
- At IoU 0.5, {nshift}/{nmatch} matched spells have a nonzero boundary offset.
- Destination-typed transitions at ±25 frames: {missed} missed, {spurious} spurious,
  {tm} matched. These counts and spell flags overlap; do not sum them as unique errors.
- False stoppages occupy {false_d} GT-live frames; {false_missing} ({percent:.1f}%) have
  missing ball XY. Across all {alive} GT-live frames, {alive_missing} lack ball XY.
  This is an association, not proof of the cause of each boundary error.
- Ball visible on {visible:.1%} of public frames; {gaps} missing-ball gaps,
  longest {longest} frames ({seconds:.2f} s).

## Examples (zero-based, stop exclusive)

{evidence}

## Definitions and limitations

Same-team positive-overlap graph degree >1 identifies candidate splits (GT degree)
and merges (prediction degree), without a tuned overlap cutoff. Tiny overlaps can
flag ambiguous cases. A spell may have both flags or neither. `overlaps.csv` retains
all candidate links; `matches.csv` contains the official greedy one-to-one results
at all five thresholds, including both unmatched sides. Unmatched rows have blank
IoU/offsets because no pair was assigned. Categories are descriptive, not corrected labels.
Missed transitions are separate events in `transitions.csv`, matched on destination
label within 25 frames using the official function, independently per labelled run.
They include restarts/stoppages; the change columns distinguish these from H↔A.
Offsets are prediction minus GT in frames; positive means late. Any nonzero matched
offset is flagged, with no claim of practical significance. Red timeline markers
show shifted, missed or spurious transitions. Crop/half edges are censored, not observed
transitions; spell tables mark these edges, and their offsets must be interpreted cautiously.

GT X excludes frames. Prediction X is forward-filled then head-backfilled within
each run exactly as in the official metric; an entirely abstaining run stays X.
The timeline shows raw and filled predictions separately. D is always a stoppage.
ATD frame IDs are checked as exactly 1..N and mapped by subtracting one. Finite x AND y
means visible; missing observations do not establish dead ball. Missing ATD records
are rejected, not silently interpreted as occlusion. No coordinates, player data,
or frame-level tracking tables are exported. Visibility CSV fields are interval/window
aggregates; one-second plot bins are in memory only.

This public sample cannot establish full-match error frequencies.
Public crop edges may truncate possessions. Candidate categories are hypotheses for
inspection; their true causes remain unresolved. No private GT was accessed.

## Reproduce

From repository root: `python -m analysis`. See `analysis/README.md` for environment
setup and tests. `manifest.json` records input/code SHA-256 hashes, dependency versions,
half bounds and fixed diagnostic conventions. Outputs contain no timestamps or randomness.
CSV/JSON/PNG bytes should reproduce in the same recorded environment; font/rendering
changes can change PNG bytes across environments. All generated files stay gitignored.
'''.format(frames=summary['frames'], nruns=len(runs), ng=len(result['gt']), npred=len(result['pred']),
           score=result['scores'].f1.mean(), scoretable='\n'.join(scorelines),
           nsplit=len(split), nmerge=len(merge), nshift=len(shifted), nmatch=len(matched),
           missed=counts.get('missed', 0), spurious=counts.get('spurious', 0), tm=counts.get('matched', 0),
           false_d=false_d, false_missing=summary['false_stoppage_missing_ball_frames'], percent=percent,
           alive=summary['alive_frames'], alive_missing=summary['alive_missing_ball_frames'],
           visible=summary['visible_fraction'], gaps=summary['missing_gaps'],
           longest=summary['longest_missing_gap'], seconds=summary['longest_missing_gap']/pm.FPS,
           evidence='\n'.join(evidence))
    (out / 'report.md').write_text(text)
