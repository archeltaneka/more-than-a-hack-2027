"""Public error analysis and full-match descriptive exports; never write coordinates."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluation import possession_metrics as pm
from .gaps import (contiguous_intervals, event_context, gap_features, interval_features,
                   public_context, safe_gap_table, EVENT_MARGIN, BOUNDARY_MARGIN)


def contingency(signal, visible, runs, source):
    mask = np.zeros(len(signal), bool)
    for a, b in runs:
        mask[a:b] = True
    rows = []
    for code, label in enumerate(pm.LABELS):
        state = mask & (signal == code)
        total = int(state.sum())
        for seen in (False, True):
            count = int((state & (visible == seen)).sum())
            rows.append(dict(source=source, state=label, visible=seen, frames=count,
                             state_frames=total, fraction_within_state=count/total if total else np.nan))
    return pd.DataFrame(rows)


def public_intervals(data, pred, gt, runs, events):
    masks = dict(false_stoppages=(pred == pm.D) & np.isin(gt, pm.POSSESSION_LABELS),
                 missed_stoppages=(gt == pm.D) & np.isin(pred, pm.POSSESSION_LABELS),
                 dead_ball_intervals=gt == pm.D)
    out = {}
    for name, mask in masks.items():
        rows = interval_features(data, pred, contiguous_intervals(mask, runs), runs, events)
        out[name] = safe_gap_table(public_context(rows, gt, data, runs), public=True)
    return out


def descriptive_summary(rows, scope, frames, visible_frames):
    """Numeric distributions, not accuracy. No GT parameter or GT fields."""
    out = dict(scope=scope, frames=frames, visible_frames=visible_frames,
               visible_fraction=visible_frames/frames if frames else None, n_gaps=len(rows),
               missing_frames=sum(r['duration_frames'] for r in rows),
               gaps_near_events=sum(r['nearby_event_count'] > 0 for r in rows),
               censored_gaps=sum(r['left_censored'] or r['right_censored'] for r in rows),
               baseline_transitions_in_gaps=sum(r['baseline_transitions_inside'] for r in rows))
    for field in ('duration_s', 'displacement_m', 'nearest_before_m', 'nearest_after_m',
                  'age_end_frames'):
        values = np.array([r[field] for r in rows], float)
        values = values[np.isfinite(values)]
        out[field+'_n'] = len(values)
        for name, q in (('median', 50), ('p90', 90), ('max', 100)):
            out[field+'_'+name] = float(np.percentile(values, q)) if len(values) else None
    available = [r for r in rows if r['team_before'] and r['team_after']]
    out['endpoint_teams_available'] = len(available)
    out['endpoint_team_changed'] = sum(r['team_before'] != r['team_after'] for r in available)
    return out


def fragmentation_context(result, data, events, runs):
    """One public GT spell per row, including unsplit controls; no slot identities."""
    rows = []
    filled = result['filled']
    for spell in result['gt'].itertuples():
        a, b = spell.start, spell.stop
        transitions = pm.transitions(filled, [(a, b)])
        near_gap = sum(not data['visible'][max(a, f-BOUNDARY_MARGIN):min(b, f+BOUNDARY_MARGIN+1)].all()
                       for f, _, _ in transitions)
        distance = data['nearest_m'][a:b]
        finite = distance[np.isfinite(distance)]
        team = data['nearest_team'][a:b]
        valid = (team[:-1] != '') & (team[1:] != '')
        rows.append(dict(gt_id=spell.id, start=a, stop=b, label=spell.label,
                         candidate_split=spell.candidate_split, candidate_merge=spell.candidate_merge,
                         missing_frames=int((~data['visible'][a:b]).sum()),
                         internal_baseline_transitions=len(transitions), transitions_near_missing=near_gap,
                         nearest_distance_median=float(np.median(finite)) if len(finite) else np.nan,
                         nearest_distance_valid_frames=len(finite),
                         nearest_team_switches=int(((team[:-1] != team[1:]) & valid).sum()),
                         **event_context(events, a, b, runs[spell.run])))
    return pd.DataFrame(rows, columns=[
        'gt_id', 'start', 'stop', 'label', 'candidate_split', 'candidate_merge',
        'missing_frames', 'internal_baseline_transitions', 'transitions_near_missing',
        'nearest_distance_median', 'nearest_distance_valid_frames', 'nearest_team_switches',
        'nearby_event_count', 'nearby_event_types', 'nearby_event_teams', 'nearest_event_distance_frames'])


def public_context_distributions(data, gt, runs, events):
    """State-level geometry and tag coverage, including live-state counterexamples."""
    in_scope = np.zeros(len(gt), bool)
    near_event = np.zeros(len(gt), bool)
    for lo, hi in runs:
        in_scope[lo:hi] = True
        for event in events:
            if (event['code'] or '').strip().lower() == 'set pieces':
                a = max(lo, event['start_frame']-EVENT_MARGIN)
                b = min(hi, event['end_frame']+EVENT_MARGIN)
                if a < b:
                    near_event[a:b] = True
    rows = []
    for code, label in enumerate(pm.LABELS):
        mask = in_scope & (gt == code)
        distances = data['nearest_m'][mask]
        finite = distances[np.isfinite(distances)]
        rows.append(dict(state=label, frames=int(mask.sum()), finite_geometry_frames=len(finite),
                         nearest_median_m=float(np.median(finite)) if len(finite) else np.nan,
                         nearest_p90_m=float(np.percentile(finite, 90)) if len(finite) else np.nan,
                         near_set_piece_frames=int((mask & near_event).sum())))
    return pd.DataFrame(rows)


def _plots(out, tables, summary, full_rows, data, gt, pred, runs, events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    c = tables['public_visibility_state']
    x = np.arange(3)
    missing = [int(c.query('state == @label and visible == False').frames.iloc[0]) for label in pm.SCORED]
    visible = [int(c.query('state == @label and visible == True').frames.iloc[0]) for label in pm.SCORED]
    axes[0, 0].bar(x, visible, label='Visible', color='#248b68')
    axes[0, 0].bar(x, missing, bottom=visible, label='Missing', color='#b9bec5')
    axes[0, 0].set(xticks=x, xticklabels=pm.SCORED, ylabel='Public frames', title='Visibility is not game state')
    axes[0, 0].legend()
    gaps = tables['public_gaps']
    bottom = np.zeros(len(gaps))
    colors = {'H': '#2479b4', 'A': '#e68a25', 'D': '#b9bec5'}
    for label in pm.SCORED:
        height = gaps['gt_'+label+'_frames'].to_numpy(float)/pm.FPS
        axes[0, 1].bar(np.arange(len(gaps)), height, bottom=bottom, label=label, color=colors[label])
        bottom += height
    axes[0, 1].set(title='Public gaps: true-state composition', xlabel='Gap ID', ylabel='Seconds')
    axes[0, 1].legend()
    for scope, rows in (('Public', gaps.to_dict('records')), ('Full video', full_rows)):
        values = np.sort([r['duration_s'] for r in rows])
        if len(values):
            axes[1, 0].step(values, np.arange(1, len(values)+1)/len(values), where='post', label=scope)
    axes[1, 0].set(xscale='log', xlabel='Gap duration (s, log scale)', ylabel='Cumulative fraction',
                  title='Duration distributions (full video includes breaks)')
    if axes[1, 0].get_legend_handles_labels()[0]:
        axes[1, 0].legend()
    for label in pm.SCORED:
        values = data['nearest_m'][gt == pm.LABELS.index(label)]
        values = np.sort(values[np.isfinite(values)])
        if len(values):
            axes[1, 1].step(values, np.arange(1, len(values)+1)/len(values), label=label, color=colors[label])
    axes[1, 1].set(xlabel='Nearest tracked player distance (m)', ylabel='Cumulative fraction',
                  title='Public geometry by true state; finite observations only')
    if axes[1, 1].get_legend_handles_labels()[0]:
        axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(out/'gap_distributions.png', dpi=140, metadata={'Software': 'possession-analysis'})
    plt.close(fig)

    examples = []
    for name in ('false_stoppages', 'missed_stoppages'):
        for row in tables[name].sort_values(['duration_frames', 'start'], ascending=[False, True]).head(2).itertuples():
            examples.append((name, row))
    if not examples:
        examples = [('no_errors', None)]
    fig, axes = plt.subplots(len(examples), 1, figsize=(13, 2.7*len(examples)), squeeze=False)
    colors.update(X='#ffffff')
    for ax, (name, row) in zip(axes[:, 0], examples):
        if row is None:
            ax.text(.5, .5, 'No public false/missed stoppage intervals', ha='center')
            continue
        lo, hi = runs[row.run]
        a, b = max(lo, row.start-EVENT_MARGIN), min(hi, row.stop+EVENT_MARGIN)
        for y, signal in ((2, gt), (1, pred)):
            for start, stop, code in pm.segments(signal, [(a, b)]):
                ax.broken_barh([(start/pm.FPS, (stop-start)/pm.FPS)], (y-.25, .5),
                               facecolors=colors[pm.LABELS[code]])
        for start in range(a, b, pm.FPS):
            stop = min(b, start+pm.FPS)
            ax.bar(start/pm.FPS, data['visible'][start:stop].mean()*.5,
                   width=(stop-start)/pm.FPS, align='edge', bottom=-.25, color='#248b68')
        for event in events:
            if (event['code'] or '').lower() == 'set pieces' and event['start_frame'] < b and event['end_frame'] > a:
                start, stop = max(a, event['start_frame']), min(b, event['end_frame'])
                ax.broken_barh([(start/pm.FPS, (stop-start)/pm.FPS)], (2.5, .15), facecolors='#8e5eb0')
        ax.axvspan(row.start/pm.FPS, row.stop/pm.FPS, facecolor='none', edgecolor='#c12443', linewidth=1.5)
        ax.set(xlim=(a/pm.FPS, b/pm.FPS), ylim=(-.35, 2.8), yticks=[0, 1, 2],
               yticklabels=['Visible fraction¹', 'Baseline', 'GT'], xlabel='Video seconds',
               title='{} [{}, {}) · nearby events: {}'.format(name.replace('_', ' '), row.start, row.stop,
                                                             row.nearby_event_types or 'none'))
    fig.legend(handles=[Patch(color=colors[k], label=k) for k in pm.SCORED]+
               [Patch(color='#8e5eb0', label='Automatic set-piece phase')], loc='lower center', ncol=4, bbox_to_anchor=(.5, .025))
    fig.text(.01, .01, '¹ One-second aggregate. Red outline: error interval. No tracking positions shown.', fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, 1))
    fig.savefig(out/'gap_examples.png', dpi=140, metadata={'Software': 'possession-analysis'})
    plt.close(fig)


def run_gap_analysis(out, data, gt, pred, result, public_runs, halves, events, manifest):
    """Public diagnostics plus identical GT-free extraction across all video regions."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    public_rows = public_context(gap_features(data, pred, public_runs, events), gt, data, public_runs)
    tables = public_intervals(data, pred, gt, public_runs, events)
    tables['public_gaps'] = safe_gap_table(public_rows, public=True)
    tables['public_visibility_state'] = contingency(gt, data['visible'], public_runs, 'ground_truth')
    tables['public_visibility_prediction'] = contingency(pred, data['visible'], public_runs, 'baseline')
    tables['fragmentation_context'] = fragmentation_context(result, data, events, public_runs)
    tables['public_context_distributions'] = public_context_distributions(data, gt, public_runs, events)
    # Published boundaries only; no GT enters full-match extraction or summaries.
    edges = sorted({0, len(pred)} | {a for _, a, _ in halves} | {b+1 for _, _, b in halves})
    full_runs = list(zip(edges[:-1], edges[1:]))
    full_rows = gap_features(data, pred, full_runs, events)
    summaries = []
    for scope, selected, runs in [('public', public_rows, public_runs), ('full_video', full_rows, full_runs)]:
        summaries.append(descriptive_summary(selected, scope, sum(b-a for a, b in runs),
                                            sum(int(data['visible'][a:b].sum()) for a, b in runs)))
    gameplay_runs = [(a, b+1) for _, a, b in halves]
    gameplay_ids = [i for i, r in enumerate(full_runs) if r in gameplay_runs]
    selected = [r for r in full_rows if r['run'] in gameplay_ids]
    summaries.append(descriptive_summary(selected, 'full_gameplay_halves', sum(b-a for a, b in gameplay_runs),
                                        sum(int(data['visible'][a:b].sum()) for a, b in gameplay_runs)))
    for i, (a, b) in enumerate(full_runs):
        summaries.append(descriptive_summary([r for r in full_rows if r['run'] == i],
                         'region_{}_{}_{}'.format(i, a, b), b-a, int(data['visible'][a:b].sum())))
    tables['scope_distributions'] = pd.DataFrame(summaries)
    tables['full_visibility_prediction'] = contingency(pred, data['visible'], full_runs, 'baseline_full_video')
    tables['full_gameplay_visibility_prediction'] = contingency(pred, data['visible'], gameplay_runs,
                                                               'baseline_gameplay_halves')
    for name, table in tables.items():
        table.to_csv(out/(name+'.csv'), index=False, float_format='%.10g', lineterminator='\n')
    (out/'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    _plots(out, tables, summaries, full_rows, data, gt, pred, public_runs, events)
    _report(out, tables, summaries)
    return tables


def _report(out, tables, summaries):
    gaps = tables['public_gaps']
    c = tables['public_visibility_state']
    def count(label, visible):
        return int(c[(c.state == label) & (c.visible == visible)].frames.iloc[0])
    live_missing = count('H', False)+count('A', False)
    dead_missing, dead_visible = count('D', False), count('D', True)
    false, missed, dead = (tables[k] for k in ('false_stoppages', 'missed_stoppages', 'dead_ball_intervals'))
    fragments = tables['fragmentation_context']
    splits = fragments[fragments.candidate_split == True]
    full = summaries[1]
    gameplay = summaries[2]
    lines = ['# Tracking gaps and game-state reliability', '',
             'Public-only error analysis of fixed `baseline_smooth50`; full-match results are descriptive.', '',
             '## Public observations', '',
             '- {} missing-ball intervals: {} contain live frames, {} contain dead frames, {} are near direct H↔A changes.'.format(
                 len(gaps), int(gaps.missing_during_live.sum()), int(gaps.missing_during_dead.sum()),
                 int(gaps.near_possession_change.sum())),
             '- Missing ball: {} live frames and {} genuine dead frames. Visible ball: {} genuine dead frames.'.format(
                 live_missing, dead_missing, dead_visible),
             '- False stoppages: {} intervals / {} frames. Missed genuine stoppages: {} intervals / {} frames.'.format(
                 len(false), int(false.duration_frames.sum()), len(missed), int(missed.duration_frames.sum())),
             '- {} of {} true D intervals contain visible ball; {} are entirely visible.'.format(
                 int((dead.visible_frames > 0).sum()), len(dead), int((dead.visible_fraction == 1).sum())),
             '- {} of {} candidate-split GT spells contain missing ball; {}/{} internal baseline transitions in split spells are within ±1 s of missing observations.'.format(
                 int((splits.missing_frames > 0).sum()), len(splits), int(splits.transitions_near_missing.sum()),
                 int(splits.internal_baseline_transitions.sum())), '',
             '| True state | Ball missing | Ball visible |', '|---|---:|---:|']
    for label in pm.SCORED:
        lines.append('| {} | {} | {} |'.format(label, count(label, False), count(label, True)))
    context = tables['public_context_distributions']
    lines += ['', '| True state | Finite geometry frames | Median nearest distance (m) | Frames near set-piece phase |',
              '|---|---:|---:|---:|']
    for row in context.itertuples():
        if row.frames:
            lines.append('| {} | {} | {:.2f} | {} |'.format(
                row.state, row.finite_geometry_frames, row.nearest_median_m, row.near_set_piece_frames))
    lines += ['', 'Near-event frame counts use the union of ±5-second expanded phases, so overlapping tags are not double-counted.',
              '', 'Counts are descriptive, frame-correlated observations; no significance or causal claim.', '',
              '## Examples', '']
    for name, table in (('False stoppage', false), ('Missed genuine stoppage', missed)):
        if table.empty:
            lines.append('- {}: none observed.'.format(name))
        for row in table.sort_values(['duration_frames', 'start'], ascending=[False, True]).head(2).itertuples():
            lines.append('- {} [{}, {}): {} frames, {:.1%} ball visibility; nearby automatic tags: {}. '
                         'Nearest distances before/after: {:.2f}/{:.2f} m; teams {}/{}.'.format(
                             name, row.start, row.stop, row.duration_frames, row.visible_fraction,
                             row.nearby_event_types or 'none', row.nearest_before_m, row.nearest_after_m,
                             row.team_before or 'unknown', row.team_after or 'unknown'))
    lines += ['', '## Three hypotheses to test on independent labelled data', '',
              '1. **Separate observation loss from game state.** {} missing-ball frames are live, while {} dead-ball frames still have visible ball. '
              'The baseline emits D only when no usable player–ball observation exists anywhere in its ±50-frame window; '
              'it does not take a majority vote including D. An explicit state model with uncertainty could avoid treating long occlusions as stoppages. '
              'This is a hypothesis, not a fitted rule.'.format(live_missing, dead_visible),
              '2. **Use temporal team-control evidence beyond instantaneous proximity.** {} of {} split-candidate possessions have no missing-ball frames, '
              'so occlusion handling alone cannot explain those splits. The split spells contain {} adjacent-frame nearest-team switches '
              '(only pairs with observations on both frames count). Test collective team evidence and persistence without assuming slot identity; '
              'nearest-player geometry does not establish controlled possession.'.format(
                  int((splits.missing_frames == 0).sum()), len(splits), int(splits.nearest_team_switches.sum())),
              '3. **Treat set-piece tags as a noisy restart prior alongside geometry.** {}/{} missed-stoppage intervals and {}/{} true D intervals '
              'are within ±5 s of automatic set-piece phases. Investigate whether these phases plus player–ball context add evidence for stoppages '
              'even when the ball remains visible. Coverage is not precision: tags also occur in live phases, have uncertain timing, and may omit teams.'.format(
                  int((missed.nearby_event_count > 0).sum()), len(missed),
                  int((dead.nearby_event_count > 0).sum()), len(dead)), '',
              '## Full match: distributions only', '',
              '- Full video: {} gaps, {:.1%} visible frames; median gap {:.2f} s, p90 {:.2f} s.'.format(
                  full['n_gaps'], full['visible_fraction'], full['duration_s_median'] or 0, full['duration_s_p90'] or 0),
              '- Published gameplay halves: {} gaps, {:.1%} visible frames; median gap {:.2f} s, p90 {:.2f} s.'.format(
                  gameplay['n_gaps'], gameplay['visible_fraction'], gameplay['duration_s_median'] or 0, gameplay['duration_s_p90'] or 0),
              '- Endpoint geometry available for {} full-video gaps; median displacement {:.2f} m. {}/{} gaps with known endpoint teams change nearest team.'.format(
                  full['displacement_m_n'], full['displacement_m_median'] or 0,
                  full['endpoint_team_changed'], full['endpoint_teams_available']),
              '- `scope_distributions.csv` also separates pre-match, each half, halftime and post-match. '
              'Baseline/visibility tables contain predictions, not true state or full-match accuracy.', '',
              '## Definitions, assumptions and limitations', '',
              '- Intervals are zero-based [start, stop). ATD IDs are one-based. Public context is clipped to labelled runs; full context is clipped at published half/outer-region edges. '
              'Censored endpoints remain unknown; no search across a scope boundary.',
              '- Pitch coordinates use ATD XML FieldSize: 105 × 68 m. DATA.md also documents games.py’s 105.3 m corrected-tracking length; '
              'we use the automatic export’s dimensions consistently. Nearest-team ties use slot order; slots are never treated as persistent players.',
              '- Last/next visible positions are computed in memory only, then reduced to a scalar displacement. '
              'No positions, displacement vectors, slot IDs, video frames or frame-level tracking tables are exported. '
              'Full-match intervals themselves stay in memory; only distributions leave memory.',
              '- Gap age at start/last missing frame is measured from the last finite XY observation inside the scope. '
              'Seconds = frames / 25. For non-gap error intervals these ages reference the observation before interval start, '
              'not observations inside the interval. Next endpoints and displacement use future data and are retrospective features, not online inputs.',
              '- Nearest distance/team before and after use the nearest visible ball frames; if all players are missing at an endpoint, geometry stays unknown. '
              'No interpolation or out-of-pitch clipping. Finite XY defines visibility; a missing record is a validation error.',
              '- Missing-live, missing-dead and near-change flags overlap. Near change means direct H↔A transition within ±25 frames of a gap; '
              'it excludes D restarts, whose context is reported separately. Baseline transitions count destination frames inside the interval, including its first frame when observable.',
              '- Event context is overlap with an interval expanded by ±125 frames, clipped to scope. '
              'Smart Tagging times use the existing loader’s round(seconds × 25). Set-piece phases are not exact restart instants. '
              'N/A team tags are preserved as unknown; empty windows have zero count and blank/NaN context.',
              '- Public tables use raw baseline labels. False stoppage = GT H/A and predicted D; missed stoppage = GT D and predicted H/A. '
              'X is reported separately and never called live. Fragmentation reuses official abstention-filled spells. '
              'The supplied baseline contains no X in this public interval.',
              '- Five minutes are insufficient to establish general error mechanisms; full-match truth is unavailable. '
              'No labels changed, no thresholds fitted, no private GT opened. The three hypotheses require independent validation.', '',
              '## Reproduce', '',
              '`python -m analysis --gaps` regenerates baseline diagnostics and these outputs under `analysis/output/gaps/`. '
              '`manifest.json` records inputs, source hashes, versions, pitch dimensions and fixed windows. '
              'See `analysis/README.md` for tests and output schemas.', '']
    (out/'report.md').write_text('\n'.join(lines))
