import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.gaps import (gap_features, contiguous_intervals, nearest_player,
                           event_context, public_context, safe_gap_table, load_geometry)
from evaluation import possession_metrics as pm


class GapTests(unittest.TestCase):
    def fixture(self):
        xy = np.array([[0., 0.], [np.nan, np.nan], [np.nan, np.nan], [3., 4.]])
        return dict(ball_m=xy, visible=np.isfinite(xy).all(axis=1),
                    nearest_m=np.array([1., np.nan, np.nan, 2.]),
                    nearest_team=np.array(['H', '', '', 'A']))

    def test_gap_endpoints_and_age(self):
        rows = gap_features(self.fixture(), np.array([0, 0, 2, 1]), [(0, 4)], [])
        row = rows[0]
        self.assertEqual((row['start'], row['stop'], row['duration_frames']), (1, 3, 2))
        np.testing.assert_array_equal(row['last_ball_m'], [0., 0.])
        np.testing.assert_array_equal(row['next_ball_m'], [3., 4.])
        self.assertEqual(row['displacement_m'], 5)
        self.assertEqual((row['age_start_frames'], row['age_end_frames']), (1, 2))
        self.assertEqual(row['baseline_transitions_inside'], 1)
        self.assertEqual(row['baseline_transition_changes'], 'H->D')

    def test_boundary_gaps_do_not_borrow_outside_window(self):
        data = self.fixture()
        rows = gap_features(data, np.zeros(4, int), [(1, 4)], [])
        self.assertIsNone(rows[0]['last_ball_m'])
        self.assertTrue(rows[0]['left_censored'])
        self.assertTrue(np.isnan(rows[0]['age_start_frames']))
        rows = gap_features(data, np.zeros(4, int), [(0, 3)], [])
        self.assertIsNone(rows[0]['next_ball_m'])
        self.assertTrue(rows[0]['right_censored'])
        self.assertEqual(contiguous_intervals(np.ones(4, bool), [(0, 2), (2, 4)]),
                         [(0, 0, 2), (1, 2, 4)])

    def test_missing_observations(self):
        data = self.fixture()
        data['ball_m'][:] = np.nan
        data['visible'][:] = False
        row = gap_features(data, np.zeros(4, int), [(0, 4)], [])[0]
        self.assertIsNone(row['last_ball_m'])
        self.assertIsNone(row['next_ball_m'])
        self.assertTrue(np.isnan(row['displacement_m']))
        self.assertEqual(gap_features(self.fixture(), np.zeros(4, int), [(0, 1)], []), [])

    def test_coordinate_conversion_and_missing_players(self):
        dist, team = nearest_player([.5, .5], np.array([[.6, .5], [.5, .7]]),
                                    np.array(['H', 'A']), (105, 68))
        self.assertAlmostEqual(dist, 10.5)
        self.assertEqual(team, 'H')
        dist, team = nearest_player([.5, .5], np.array([[np.nan, 0.], [0., np.nan]]),
                                    np.array(['H', 'A']), (105, 68))
        self.assertTrue(np.isnan(dist))
        self.assertEqual(team, '')
        dist, team = nearest_player([np.nan, .5], np.array([[0., 0.]]),
                                    np.array(['H']), (105, 68))
        self.assertTrue(np.isnan(dist))

    def test_empty_and_half_clipped_event_windows(self):
        empty = event_context([], 10, 20, (0, 30))
        self.assertEqual(empty['nearby_event_count'], 0)
        event = dict(code='SET PIECES', start_frame=35, end_frame=40,
                     tags={'Type': 'Throw-in', 'Team': 'N/A'})
        self.assertEqual(event_context([event], 10, 20, (0, 30))['nearby_event_count'], 0)
        found = event_context([event], 10, 20, (0, 60))
        self.assertEqual(found['nearby_event_types'], 'Throw-in')
        self.assertEqual(found['nearby_event_teams'], 'N/A')

    def test_public_annotations_and_private_export_protection(self):
        data = self.fixture()
        rows = gap_features(data, np.array([0, 2, 2, 1]), [(0, 4)], [])
        self.assertFalse(any(k.startswith('gt_') for k in rows[0]))
        annotated = public_context(rows, pm.label_codes(list('HHDA')), data, [(0, 4)])
        self.assertEqual(annotated[0]['gt_H_frames'], 1)
        self.assertEqual(annotated[0]['gt_D_frames'], 1)
        self.assertTrue(annotated[0]['missing_during_live'])
        self.assertTrue(annotated[0]['missing_during_dead'])
        rows[0]['accidental_coordinates'] = [123, 456]
        table = safe_gap_table(rows)
        self.assertNotIn('accidental_coordinates', table)
        self.assertNotIn('last_ball_m', table)
        self.assertNotIn('next_ball_m', table)
        self.assertIn('displacement_m', table)

    def test_geometry_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'atd.txt'
            path.write_text('1:0.5,0.5,0;NaN,NaN,NaN:0.6,0.5,0\n'
                            '2:NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN\n')
            data = load_geometry(path, 2, np.array(['H', 'A']), (105, 68))
            np.testing.assert_allclose(data['ball_m'][0], [63, 34])
            self.assertAlmostEqual(data['nearest_m'][0], 10.5)
            self.assertTrue(np.isnan(data['nearest_m'][1]))
            path.write_text('1:0,0,0:0,0,0\n2:0,0,0:0,0,0\n')
            with self.assertRaises(ValueError):
                load_geometry(path, 2, np.array(['H', 'A']), (105, 68))

    def test_contingency_and_error_intervals(self):
        from analysis.gap_report import contingency, public_intervals
        data = self.fixture()
        gt, pred = pm.label_codes(list('HHDD')), pm.label_codes(list('HDHA'))
        table = contingency(gt, data['visible'], [(0, 4)], 'ground_truth')
        self.assertEqual(table.frames.sum(), 4)
        self.assertEqual(table.query('state == "D" and visible == True').frames.iloc[0], 1)
        tables = public_intervals(data, pred, gt, [(0, 4)], [])
        self.assertEqual(tables['false_stoppages'].duration_frames.sum(), 1)
        self.assertEqual(tables['missed_stoppages'].duration_frames.sum(), 2)
        self.assertEqual(tables['dead_ball_intervals'].visible_frames.sum(), 1)

    def test_full_distributions_no_gt_and_empty_gaps(self):
        from analysis.gap_report import descriptive_summary
        summary = descriptive_summary([], 'full_video', 4, 4)
        self.assertEqual(summary['n_gaps'], 0)
        self.assertIsNone(summary['duration_s_median'])
        self.assertFalse(any('gt' in key or 'accuracy' in key for key in summary))

    def test_all_dead_no_gap_export(self):
        from analysis.core import analyse
        from analysis.gap_report import run_gap_analysis
        data = self.fixture()
        data['visible'][:] = True
        data['ball_m'][:] = 0
        gt = pred = pm.label_codes(list('DDDD'))
        result = analyse(gt, pred, [(0, 4)])
        with tempfile.TemporaryDirectory() as directory:
            tables = run_gap_analysis(directory, data, gt, pred, result, [(0, 4)],
                                      [(1, 0, 3)], [], {})
            self.assertTrue(tables['public_gaps'].empty)
            self.assertTrue(tables['fragmentation_context'].empty)
            self.assertTrue((Path(directory) / 'report.md').is_file())
