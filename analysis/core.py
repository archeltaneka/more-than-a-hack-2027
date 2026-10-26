"""Spell and transition diagnostics using the official metric's conventions."""
import numpy as np
import pandas as pd
from evaluation import possession_metrics as pm


def load_labels(path, n_expected):
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, n_expected)
    if reason:
        raise ValueError('{}: {}'.format(path, reason))
    return pm.label_codes(table)


def labelled_runs(gt, halves):
    """Intersect public label runs with inclusive published half blocks.

    Reject labels outside halves rather than silently discarding them.
    """
    covered = np.zeros(len(gt), dtype=bool)
    runs = []
    previous = -1
    for _, start, last in halves:
        if start <= previous or not 0 <= start <= last < len(gt):
            raise ValueError('invalid or overlapping half blocks')
        previous = last
        covered[start:last + 1] = True
        runs.extend((start + a, start + b) for a, b in pm.valid_runs(gt[start:last + 1]))
    if np.any((gt != pm.X) & ~covered):
        raise ValueError('ground truth labels outside published halves')
    return runs


def _spell_table(spells, runs):
    rows = []
    for i, (a, b, k) in enumerate(spells):
        run = next(j for j, (lo, hi) in enumerate(runs) if lo <= a < b <= hi)
        lo, hi = runs[run]
        rows.append(dict(id=i, run=run, start=a, stop=b, duration=b-a,
                         label=pm.LABELS[k], left_censored=a == lo,
                         right_censored=b == hi, candidate_split=False, candidate_merge=False))
    return pd.DataFrame(rows, columns=['id', 'run', 'start', 'stop', 'duration', 'label',
                                      'left_censored', 'right_censored',
                                      'candidate_split', 'candidate_merge'])


def _pairs(overlaps, n_gt, n_pred, threshold):
    """Expose pair identities absent from official API; mirror its greedy ordering.

    Both matched GT identities and TP/FP/FN are checked against official functions.
    """
    pairs, used_g, used_p = [], set(), set()
    for iou, g, p in overlaps:
        if iou < threshold:
            break
        if g not in used_g and p not in used_p:
            pairs.append((iou, g, p))
            used_g.add(g)
            used_p.add(p)
    flags = np.array([g in used_g for g in range(n_gt)], dtype=bool)
    if not np.array_equal(flags, pm.matched_gt_flags(overlaps, n_gt, threshold)):
        raise RuntimeError('diagnostic matching differs from official GT identities')
    counts = (len(pairs), n_pred-len(pairs), n_gt-len(pairs))
    if counts != pm.match_spells(overlaps, n_gt, n_pred, threshold):
        raise RuntimeError('diagnostic matching differs from official counts')
    return pairs, used_g, used_p


def analyse(gt, pred, runs):
    """Return spell tables, all-threshold matches, topology and transition tables.

    Inputs are full video-indexed codes. Runs must not cross halves or GT-X gaps.
    Candidate flags use *any* positive same-team overlap, so tiny overlaps may flag
    ambiguous cases. They are non-exclusive and do not assert corrected labels.
    """
    if len(gt) != len(pred):
        raise ValueError('label length mismatch')
    filled = pm.fill_abstentions(pred, runs)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    gtab, ptab = _spell_table(gs, runs), _spell_table(ps, runs)
    overlaps = pm.spell_overlaps(gs, ps)
    gdegree, pdegree = np.zeros(len(gs), int), np.zeros(len(ps), int)
    for _, g, p in overlaps:
        gdegree[g] += 1
        pdegree[p] += 1
    for _, g, p in overlaps:
        if gdegree[g] > 1:
            gtab.at[g, 'candidate_split'] = ptab.at[p, 'candidate_split'] = True
        if pdegree[p] > 1:
            gtab.at[g, 'candidate_merge'] = ptab.at[p, 'candidate_merge'] = True
    rows, scores = [], []
    for t in pm.SPELL_IOUS:
        pairs, used_g, used_p = _pairs(overlaps, len(gs), len(ps), t)
        entries = [('matched', iou, g, p) for iou, g, p in pairs]
        entries += [('unmatched_gt', None, g, None) for g in range(len(gs)) if g not in used_g]
        entries += [('unmatched_pred', None, None, p) for p in range(len(ps)) if p not in used_p]
        for status, iou, g, p in entries:
            start_offset = ps[p][0]-gs[g][0] if status == 'matched' else None
            stop_offset = ps[p][1]-gs[g][1] if status == 'matched' else None
            tags = set()
            for table, index in ((gtab, g), (ptab, p)):
                if index is not None:
                    for flag in ('candidate_split', 'candidate_merge'):
                        if table.at[index, flag]:
                            tags.add(flag)
            if status == 'matched' and (start_offset or stop_offset):
                tags.add('boundary_shift')
            rows.append(dict(threshold=t, status=status, gt_id=g, pred_id=p, iou=iou,
                             start_offset=start_offset, stop_offset=stop_offset,
                             categories=';'.join(sorted(tags))))
        tp = len(pairs)
        scores.append(dict(threshold=t, tp=tp, fp=len(ps)-tp, fn=len(gs)-tp,
                           f1=2*tp/(len(gs)+len(ps)) if gs or ps else 0.0))
    # Match each run independently: even adjacent synthetic halves cannot pair.
    trans = []
    for run_id, run in enumerate(runs):
        gt_t, pred_t = pm.transitions(gt, [run]), pm.transitions(filled, [run])
        pairs, missed, spurious = pm.match_transitions(gt_t, pred_t, 25)
        entries = [('matched', g, p) for g, p in pairs]
        entries += [('missed', g, None) for g in missed]
        entries += [('spurious', None, p) for p in spurious]
        for status, g, p in entries:
            trans.append(dict(run=run_id, status=status, gt_frame=g[0] if g else None,
                              pred_frame=p[0] if p else None,
                              gt_change='->'.join(pm.LABELS[k] for k in g[1:]) if g else '',
                              pred_change='->'.join(pm.LABELS[k] for k in p[1:]) if p else '',
                              offset=p[0]-g[0] if g and p else None,
                              candidate_missed_transition=status == 'missed'))
    return dict(gt=gtab, pred=ptab, filled=filled, matches=pd.DataFrame(rows, columns=[
        'threshold', 'status', 'gt_id', 'pred_id', 'iou', 'start_offset', 'stop_offset', 'categories']),
        scores=pd.DataFrame(scores), overlaps=pd.DataFrame(overlaps, columns=['iou', 'gt_id', 'pred_id']),
        transitions=pd.DataFrame(trans, columns=['run', 'status', 'gt_frame', 'pred_frame',
                                               'gt_change', 'pred_change', 'offset',
                                               'candidate_missed_transition']))
