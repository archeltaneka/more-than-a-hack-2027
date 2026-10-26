import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.core import analyse, load_labels, labelled_runs
from analysis.tracking import load_visibility, visibility_summary
from evaluation import possession_metrics as pm


def codes(s):
    return pm.label_codes(list(s))


class DiagnosticsTests(unittest.TestCase):
    def test_half_boundaries_and_one_frame(self):
        g = codes('HHHH')
        runs = labelled_runs(g, [(1, 0, 1), (2, 2, 3)])
        self.assertEqual(runs, [(0, 2), (2, 4)])
        result = analyse(g, g, runs)
        self.assertEqual(len(result['gt']), 2)
        result = analyse(codes('HDAXH'), codes('HDAXH'), [(0, 3), (4, 5)])
        self.assertEqual(result['gt'].duration.tolist(), [1, 1, 1])
        self.assertEqual(len(result['transitions']), 2)

    def test_abstentions(self):
        g, p = codes('HHHHXAA'), codes('XHXXXAX')
        r = analyse(g, p, pm.valid_runs(g))
        self.assertEqual(r['filled'].tolist(), codes('HHHHXAA').tolist())
        self.assertTrue((r['scores'].f1 == 1).all())
        r = analyse(codes('HH'), codes('XX'), [(0, 2)])
        self.assertEqual(len(r['pred']), 0)
        self.assertTrue((r['scores'].fn == 1).all())

    def test_split_merge_and_nonexclusive_flags(self):
        split = analyse(codes('HHHHH'), codes('HHDHH'), [(0, 5)])
        self.assertTrue(split['gt'].iloc[0].candidate_split)
        self.assertEqual(split['pred'].candidate_split.sum(), 2)
        merge = analyse(codes('HHAHH'), codes('HHHHH'), [(0, 5)])
        self.assertTrue(merge['pred'].iloc[0].candidate_merge)
        self.assertEqual((merge['transitions'].status == 'missed').sum(), 2)
        both = analyse(codes('HHHDHHH'), codes('HDHHHHH'), [(0, 7)])
        self.assertTrue(both['pred'].iloc[1].candidate_split)
        self.assertTrue(both['pred'].iloc[1].candidate_merge)

    def test_offsets_and_official_consistency(self):
        g, p = codes('DDHHHHDD'), codes('DDDHHHHD')
        r = analyse(g, p, [(0, 8)])
        row = r['matches'].query('threshold == 0.5 and status == "matched"').iloc[0]
        self.assertEqual((row.start_offset, row.stop_offset), (1, 1))
        self.assertAlmostEqual(row.iou, 3 / 5)
        rng = np.random.RandomState(0)
        for _ in range(30):
            g = rng.randint(0, 4, 80).astype(np.int8)
            p = rng.randint(0, 4, 80).astype(np.int8)
            r = analyse(g, p, pm.valid_runs(g))
            self.assertAlmostEqual(r['scores'].f1.mean(), pm.evaluate(g, p)['spell_f1_mean'])

    def test_empty_spell_tables_export(self):
        from analysis.report import enrich, write_outputs
        for truth, prediction in [('HH', 'XX'), ('DD', 'DD')]:
            g, p = codes(truth), codes(prediction)
            r = analyse(g, p, [(0, 2)])
            summary = enrich(r, g, p, np.array([True, False]), [(0, 2)], [(1, 0, 1)])
            with tempfile.TemporaryDirectory() as d:
                write_outputs(d, r, g, p, [(0, 2)], summary, {})
                self.assertTrue((Path(d) / 'timeline.png').is_file())
                self.assertIn('spellF1', (Path(d) / 'report.md').read_text())

    def test_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'labels.csv'
            for body in ['frame,label\n1,H\n0,A\n', 'frame,label\n0,H\n0,A\n',
                         'frame,label\n0,H\n1,Z\n', 'frame,label\n0,H\n1,\n',
                         'frame,label\n0.0,H\n1.0,A\n']:
                path.write_text(body)
                with self.assertRaises(ValueError):
                    load_labels(path, 2)
            path.write_text('frame,label\n0,H\n1,A\n')
            self.assertEqual(load_labels(path, 2).tolist(), [pm.H, pm.A])

    def test_missing_ball_gaps_and_one_based_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'atd.txt'
            path.write_text('1:unused:0.2,0.4,NaN\n2:unused:NaN,NaN,NaN\n'
                            '3:unused:0.2,NaN,0\n4:unused:0.3,0.5,0\n')
            v = load_visibility(path, 4)
            self.assertEqual(v.tolist(), [True, False, False, True])
            s = visibility_summary(v, [(0, 4)])
            self.assertEqual((s['missing_frames'], s['longest_missing_gap']), (2, 2))
            s = visibility_summary(v, [(0, 2), (2, 4)])
            self.assertEqual(s['longest_missing_gap'], 1)
            path.write_text('1:unused:0,0,0\n3:unused:0,0,0\n')
            with self.assertRaises(ValueError):
                load_visibility(path, 2)
            path.write_text('1:unused:0,0,0\n')
            with self.assertRaises(ValueError):
                load_visibility(path, 2)


if __name__ == '__main__':
    unittest.main()
