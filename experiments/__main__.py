"""Generate, validate and describe four fixed experiments; evaluate public GT last."""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from analysis.core import load_labels
from analysis.gaps import tracking_metadata
from evaluation import possession_metrics as pm
from games import GAME, asset_paths, half_video_blocks, source_frames
from starter.load_smart_tagging import load as load_events
from .config import Config, variations
from .inputs import load_observations
from .model import infer, CANDIDATES
from .reporting import describe, evidence_summary, write_report

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_table(codes, n):
    if len(codes) != n or not np.isin(codes, [0, 1, 2, 3]).all():
        raise ValueError('invalid prediction codes/length')
    table = pd.DataFrame({'frame': np.arange(n), 'label': np.array(pm.LABELS)[codes]})
    reason = pm.validate_submission(table, n)
    if reason:
        raise ValueError(reason)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', help='Licensed bundle root; otherwise MTAH_DATA_DIR or data/')
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/default.json')
    args = parser.parse_args()
    config = Config(**json.loads(args.config.read_text()))
    if config.fps != pm.FPS:
        raise ValueError('match time base is fixed at 25fps')
    settings = {'default': config, **variations(config)}
    out = ROOT/'analysis/output/experiments'
    submissions = out/'submissions'
    submissions.mkdir(parents=True, exist_ok=True)
    (out/'configurations.json').write_text(json.dumps({k: asdict(v) for k, v in settings.items()},
                                                    sort_keys=True, indent=2)+'\n')
    assets = asset_paths(data_dir=args.data_dir)
    n = source_frames()
    runs = [(a, b+1) for _, a, b in half_video_blocks(GAME)]
    pitch, teams = tracking_metadata(assets['atd_meta'])
    baseline_path = ROOT/'evaluation/results/baseline_smooth50.csv'
    baseline = load_labels(baseline_path, n)
    observations = load_observations(assets['atd_raw'], n, teams, pitch, config.top_k)
    events = load_events(data_dir=args.data_dir)
    defaults = {'baseline_smooth50': baseline}
    evidence, stats, sensitive = [], [], []
    for candidate in CANDIDATES:
        pred, details = infer(observations, baseline, runs, events, config, candidate)
        defaults[candidate] = pred
        checked_table(pred, n).to_csv(submissions/(candidate+'.csv'), index=False, lineterminator='\n')
        # Re-read bytes actually written through the official validator.
        load_labels(submissions/(candidate+'.csv'), n)
        evidence.append(dict(candidate=candidate, **evidence_summary(details, runs)))
        print(candidate+' generated and validated; no GT used.', flush=True)
    idx = np.concatenate([np.arange(a, b) for a, b in runs])
    for candidate, pred in defaults.items():
        for scope, spans in [('both_halves', runs), ('half_1', runs[:1]), ('half_2', runs[1:])]:
            stats.append(dict(candidate=candidate, scope=scope, **describe(pred, spans)))
    for setting, cfg in settings.items():
        for candidate in CANDIDATES:
            pred = defaults[candidate] if setting == 'default' else infer(
                observations, baseline, runs, events, cfg, candidate)[0]
            checked_table(pred, n)
            sensitive.append(dict(candidate=candidate, variation=setting,
                                  disagreement_fraction=float((pred[idx] != defaults[candidate][idx]).mean()),
                                  **describe(pred, runs)))
        print('Sensitivity complete: '+setting+' (GT-free).', flush=True)
    # Evaluation is deliberately last. No model function accepts GT or scores.
    gt_path = ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    gt = load_labels(gt_path, n)
    public, ious, official = [], [], {}
    for candidate, pred in defaults.items():
        result = pm.evaluate(gt, pred, bootstrap_n=0)
        official[candidate] = result
        public.append(dict(candidate=candidate, spell_f1=result['spell_f1_mean'],
                           macro_f1=result['macro_f1'], spell_f1_05=result['spell_f1_at_headline'],
                           dead_f1=result['per_class']['D']['f1'],
                           public_gt_spells=result['n_gt_spells'], public_pred_spells=result['n_pred_spells'],
                           public_D_frames=int(((gt != pm.X) & (pred == pm.D)).sum())))
        for threshold, score in result['spell_f1'].items():
            ious.append(dict(candidate=candidate, iou=threshold, **score))
    public, full, sensitivity = pd.DataFrame(public), pd.DataFrame(stats), pd.DataFrame(sensitive)
    for name, frame in [('public_scores', public), ('public_iou', pd.DataFrame(ious)),
                        ('full_predictions', full), ('sensitivity', sensitivity),
                        ('evidence_availability', pd.DataFrame(evidence))]:
        frame.to_csv(out/(name+'.csv'), index=False, float_format='%.10g', lineterminator='\n')
    (out/'official_public.json').write_text(json.dumps(official, sort_keys=True, indent=2)+'\n')
    write_report(out, public, full, sensitivity)
    sources = sorted((ROOT/'experiments').glob('*.py')) + [ROOT/p for p in (
        'analysis/tracking.py', 'analysis/gaps.py', 'analysis/core.py', 'games.py',
        'starter/load_smart_tagging.py', 'evaluation/possession_metrics.py')]
    manifest = dict(inputs={key: digest(path) for key, path in dict(
        tactical_atd=assets['atd_raw'], metadata=assets['atd_meta'], smart_tagging=assets['smart_tagging'],
        baseline=baseline_path, public_gt=gt_path, config=args.config).items()},
        code={str(p.relative_to(ROOT)): digest(p) for p in sources},
        output_hashes={str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
                       if p.is_file() and p.name != 'manifest.json'},
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      pandas=pd.__version__, matplotlib=matplotlib.__version__),
        n_frames=n, runs=runs, pitch_m=pitch, public_gt_used_for_fit=False,
        private_gt_used=False, sensitivity_public_scoring=False)
    (out/'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    print(public.to_string(index=False), flush=True)
    print('Report: '+str(out/'report.md'), flush=True)


if __name__ == '__main__':
    main()
