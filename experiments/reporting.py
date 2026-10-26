"""GT-free prediction descriptions and public-only experiment reporting."""
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation import possession_metrics as pm


def describe(pred, runs, fps=25):
    spells = pm.possession_spells(pred, runs)
    durations = np.array([(b-a)/fps for a, b, _ in spells])
    indices = np.concatenate([np.arange(a, b) for a, b in runs])
    changes = pm.transitions(pred, runs)
    real_changes = [t for t in changes if t[1] != pm.X and t[2] != pm.X]
    frames = len(indices)
    result = dict(frames=frames, predicted_possessions=len(spells),
                  mean_duration_s=float(durations.mean()) if len(durations) else np.nan,
                  p10_duration_s=float(np.percentile(durations, 10)) if len(durations) else np.nan,
                  median_duration_s=float(np.median(durations)) if len(durations) else np.nan,
                  p90_duration_s=float(np.percentile(durations, 90)) if len(durations) else np.nan,
                  max_duration_s=float(durations.max()) if len(durations) else np.nan,
                  transitions=len(real_changes), transitions_including_X=len(changes),
                  transitions_per_minute=len(real_changes)/(frames/fps/60) if frames else 0,
                  direct_team_changes=sum(t[1] in (pm.H, pm.A) and t[2] in (pm.H, pm.A) for t in real_changes))
    for code, label in enumerate(pm.LABELS):
        result[label+'_frames'] = int((pred[indices] == code).sum())
    return result


def evidence_summary(details, runs):
    idx = np.concatenate([np.arange(a, b) for a, b in runs])
    q = details['reliability']
    row = dict(ball_missing_frames=int((~q['visible'][idx]).sum()),
               rejected_visible_frames=int((q['visible'][idx] & (q['ball_quality'][idx] == 0)).sum()),
               team_geometry_unavailable_frames=int((q['team_quality'][idx] == 0).sum()))
    if 'dead' in details:
        row.update(dead_evidence_available_frames=int(details['dead']['available'][idx].sum()),
                   positive_dead_evidence_frames=int((details['dead']['logit'][idx] > 0).sum()))
    return row


def write_report(out, public, full, sensitivity):
    out = Path(out)
    lines = ['# First possession-inference experiments', '',
             'Fixed, untuned E1–E4 ablations. Public scoring uses only 17 possessions; no general winner is selected.', '',
             '## Public official scores', '',
             '| Candidate | spellF1 (mean) | macroF1 | spellF1 @ .5 | D F1 | Pred. spells | Pred. D frames |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in public.itertuples():
        lines.append('| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {} | {} |'.format(
            row.candidate, row.spell_f1, row.macro_f1, row.spell_f1_05, row.dead_f1,
            row.public_pred_spells, row.public_D_frames))
    failed_dead = public[public.candidate.isin(['E3', 'E4']) & (public.public_D_frames == 0)]
    if not failed_dead.empty:
        lines += ['', '**Observed failure:** '+', '.join(failed_dead.candidate)+
                  ' produce no D frames on the public sample despite containing a dead-ball component. '
                  'Synthetic stoppage support does not establish real-data effectiveness. '
                  'Parameters were not adjusted in response to these scores.']
    lines += ['', 'All five IoU scores/counts are in `public_iou.csv`; complete official results in `official_public.json`. '
              'These observations diagnose behaviour, not private/full-match performance. No bootstrap confidence claim is made.', '',
              '## Full-match predictions (published gameplay halves only)', '',
              '| Candidate | Possessions | Mean s | Median s | p90 s | Transitions/min | D frames | X frames |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in full[full.scope == 'both_halves'].itertuples():
        lines.append('| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} | {} |'.format(
            row.candidate, row.predicted_possessions, row.mean_duration_s, row.median_duration_s,
            row.p90_duration_s, row.transitions_per_minute, row.D_frames, row.X_frames))
    lines += ['', 'Duration quantiles, direct H↔A changes and separate half statistics are in `full_predictions.csv`. '
              'Counts use raw predictions: X breaks a spell. Transitions/minute count only H/A/D-to-H/A/D changes and use all published gameplay minutes. '
              'No transitions cross halftime. Outside published halves, candidates output X; these frames are excluded here.', '',
              '## Predefined sensitivity (no GT scoring)', '',
              '| Candidate | Possession count range | D-frame range | Maximum label disagreement with default |',
              '|---|---:|---:|---:|']
    for candidate in ('E1', 'E2', 'E3', 'E4'):
        group = sensitivity[sensitivity.candidate == candidate]
        lines.append('| {} | {}–{} | {}–{} | {:.1%} |'.format(
            candidate, group.predicted_possessions.min(), group.predicted_possessions.max(),
            group.D_frames.min(), group.D_frames.max(), group.disagreement_fraction.max()))
    lines += ['', 'All variations were declared before public scoring: transition costs ×0.5/×2, bridge 1/4 s, '
              'proximity scale 2/5 m, dead weight ×0.5/×1.5, one factor at a time. Inactive factors are retained as negative controls. '
              'No variation is selected. See `sensitivity.csv` and the exact settings in `configurations.json`.', '',
              '## Interpretation and limitations', '',
              '- E1 retains supported baseline H/A and fills unsupported periods using short offline endpoint context or long-gap past-state carry. '
              'E2 replaces control evidence with collective proximity plus persistence. E3 adds a stoppage detector to baseline team evidence. E4 combines both.',
              '- Reliability is shared. E1/E2 intentionally have no positive dead-ball detector and predict no D; this is an ablation limitation, not a game-state assertion. '
              'E2 also adds a decoder, while E3/E4 share that decoder. Differences from E1 are component bundles, not estimates of individual causal effects.',
              '- Missing coordinates are unknown, not stationary or dead. Speeds/displacements are never computed across gaps. '
              'Implausible jumps (>configured limit) suppress geometry at both endpoints. Missing either team suppresses team-control evidence.',
              '- Top-k distance-weighted team support avoids persistent player identities but remains vulnerable to duplicate tracks, incomplete teams and loose-ball ambiguity. '
              'Nearest-team changes are not treated as immediate possession changes.',
              '- Slow ball/local displacement, large player distance and set-piece phases provide soft stoppage evidence. '
              'These proxies also occur during live play; moving restarts and dead balls may be missed. Tags alone cannot trigger D without motion observations.',
              '- Long occlusions contribute zero evidence. The decoder holds its previous state and cannot switch inside unsupported stretches; '
              'it can therefore carry the wrong team or D through a genuine unobserved transition. Leading unknown stretches and wholly unsupported halves remain X.',
              '- Offline endpoint bridging, centred windows and Viterbi use future observations. Boundaries in weak evidence can be temporally ambiguous. '
              'This is not a streaming model. The same configuration is used for both halves.',
              '- ATD pitch scaling is 105 × 68 m from metadata; the organiser baseline used 105.3 × 68 m. '
              'The baseline labels are reused unchanged. Geometry excludes no finite out-of-pitch points by default.',
              '- Parameters are round physical/time priors chosen before evaluation, not estimated from labels. '
              'Diagnostics motivated the architecture, so the public sample is not an untouched validation set. No private GT was accessed.',
              '- Only submission labels and aggregate diagnostics are exported. No tracking coordinates, player trajectories, frame-level evidence or video frames are written.', '',
              '## Reproduce', '',
              '`python -m experiments` generates defaults, runs the predefined sensitivity, validates all arrays and saved submissions, '
              'then opens the public GT to call official `possession_metrics.evaluate`. `manifest.json` records input, source and output hashes and versions. '
              'Outputs stay under gitignored `analysis/output/experiments/`. See `experiments/README.md` for formulas and tests.', '']
    (out/'report.md').write_text('\n'.join(lines))
    _plot(out, public, full, sensitivity)


def _plot(out, public, full, sensitivity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(public))
    axes[0].bar(x, public.spell_f1, color='#2479b4')
    axes[0].set(xticks=x, xticklabels=public.candidate, ylim=(0, 1), title='Public spellF1 · 17 GT spells', ylabel='Official mean F1')
    f = full[full.scope == 'both_halves']
    axes[1].bar(np.arange(len(f)), f.predicted_possessions, color='#248b68')
    axes[1].set(xticks=np.arange(len(f)), xticklabels=f.candidate, title='Full-match predicted possessions', ylabel='Count (not accuracy)')
    for i, candidate in enumerate(('E1', 'E2', 'E3', 'E4')):
        values = sensitivity[sensitivity.candidate == candidate].predicted_possessions.to_numpy()
        axes[2].scatter(np.full(len(values), i), values, color='#e68a25', alpha=.6)
    axes[2].set(xticks=range(4), xticklabels=['E1', 'E2', 'E3', 'E4'], title='Predefined sensitivity · all variations', ylabel='Predicted possessions')
    for ax in axes[:2]:
        ax.tick_params(axis='x', labelrotation=25)
    fig.suptitle('Untuned experimental comparison — no full-match accuracy claim', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out/'comparison.png', dpi=140, metadata={'Software': 'possession-experiments'})
    plt.close(fig)
