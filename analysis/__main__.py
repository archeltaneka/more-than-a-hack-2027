"""Run: python -m analysis (from repository root)."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from evaluation import possession_metrics as pm
from games import GAME, asset_paths, half_video_blocks, source_frames
from .core import analyse, labelled_runs, load_labels
from .report import enrich, write_outputs
from .tracking import load_visibility

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', help='Licensed bundle root (otherwise MTAH_DATA_DIR or data/)')
    parser.add_argument('--gaps', action='store_true', help='Also run public and full-match tracking-gap diagnostics')
    args = parser.parse_args()
    gt_path = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    pred_path = ROOT / 'evaluation/results/baseline_smooth50.csv'
    atd_path = Path(asset_paths(data_dir=args.data_dir, source='tactical')['atd_raw'])
    n = source_frames('tactical')
    gt, pred = load_labels(gt_path, n), load_labels(pred_path, n)
    halves = half_video_blocks(GAME)
    runs = labelled_runs(gt, halves)
    if not runs:
        raise ValueError('public ground truth has no labelled frames')
    result = analyse(gt, pred, runs)
    official = pm.evaluate(gt, pred, bootstrap_n=0)
    for row in result['scores'].itertuples():
        expected = official['spell_f1'][row.threshold]
        if (row.tp, row.fp, row.fn) != (expected['n_matched'], expected['n_spurious'], expected['n_missed']):
            raise RuntimeError('official evaluate count mismatch')
        if not np.isclose(row.f1, expected['f1'], rtol=0, atol=1e-15):
            raise RuntimeError('official evaluate F1 mismatch')
    if not np.isclose(result['scores'].f1.mean(), official['spell_f1_mean'], rtol=0, atol=1e-15):
        raise RuntimeError('official mean spellF1 mismatch')
    assets = asset_paths(data_dir=args.data_dir, source='tactical')
    if args.gaps:
        from .gaps import load_geometry, tracking_metadata
        from starter.load_smart_tagging import load as load_events
        pitch, teams = tracking_metadata(assets['atd_meta'])
        geometry = load_geometry(atd_path, n, teams, pitch)
        visible = geometry['visible']
        events = load_events(data_dir=args.data_dir)
    else:
        visible = load_visibility(atd_path, n)
    summary = enrich(result, gt, pred, visible, runs, halves)
    sources = sorted((ROOT/'analysis').glob('*.py')) + [ROOT/'evaluation/possession_metrics.py', ROOT/'games.py']
    manifest = dict(inputs={'public_gt': sha256(gt_path), 'baseline_smooth50': sha256(pred_path),
                            'tactical_atd': sha256(atd_path)},
                    code={str(p.relative_to(ROOT)): sha256(p) for p in sources},
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  pandas=pd.__version__, matplotlib=matplotlib.__version__),
                    runs=runs, half_blocks=halves, fps=pm.FPS, ious=pm.SPELL_IOUS,
                    transition_tolerance_frames=25, split_merge_rule='any positive same-team overlap',
                    official_spell_f1=official['spell_f1_mean'])
    out = ROOT / 'analysis/output'
    write_outputs(out, result, gt, pred, runs, summary, manifest)
    if args.gaps:
        from .gap_report import run_gap_analysis
        gap_manifest = dict(manifest)
        gap_manifest['inputs'] = dict(manifest['inputs'],
                                      tactical_atd_metadata=sha256(assets['atd_meta']),
                                      smart_tagging=sha256(assets['smart_tagging']))
        gap_manifest['code'] = dict(manifest['code'])
        for name in ('starter/load_smart_tagging.py', 'starter/baseline.py'):
            gap_manifest['code'][name] = sha256(ROOT/name)
        gap_manifest.update(pitch_m=pitch, pitch_source='ATD XML FieldSize Width/Height',
                            event_context_frames=125, possession_change_context_frames=25,
                            full_match_ground_truth_used=False,
                            coordinate_export=False)
        run_gap_analysis(out/'gaps', geometry, gt, pred, result, runs, halves, events, gap_manifest)
    print(json.dumps(dict(output=str(out), spellF1=official['spell_f1_mean'],
                          gt_spells=len(result['gt']), predicted_spells=len(result['pred'])), indent=2))


if __name__ == '__main__':
    main()
