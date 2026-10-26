import unittest
import numpy as np
from experiments.observation import continuity_baseline


class ContinuityTests(unittest.TestCase):
    def test_missing_ball_continuity(self):
        base = np.array([0, 0, 2, 2, 0])
        result = continuity_baseline(base, np.array([1, 1, 0, 0, 1], bool), [(0, 5)], 2)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0])

    def test_long_occlusion_and_offline_short_gap(self):
        base = np.array([0, 2, 2, 2, 2, 1])
        reliable = np.array([1, 0, 0, 0, 0, 1], bool)
        np.testing.assert_array_equal(continuity_baseline(base, reliable, [(0, 6)], 2),
                                      [0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(continuity_baseline(base, reliable, [(0, 6)], 4),
                                      [0, 0, 0, 1, 1, 1])

    def test_half_boundaries_and_all_unknown(self):
        base = np.array([0, 2, 2, 2, 1, 1])
        reliable = np.array([1, 0, 0, 0, 1, 1], bool)
        result = continuity_baseline(base, reliable, [(0, 2), (3, 6)], 1)
        np.testing.assert_array_equal(result, [0, 0, 3, 1, 1, 1])
        result = continuity_baseline(np.full(4, 2), np.zeros(4, bool), [(0, 4)], 2)
        np.testing.assert_array_equal(result, [3, 3, 3, 3])


class EvidenceTests(unittest.TestCase):
    def test_collective_support_and_slot_permutation(self):
        from experiments.control import proximity_evidence
        distances = np.array([[[1., 2., 2.5], [1.1, 10., 20.]]])
        p = proximity_evidence(distances, 3, .5)
        self.assertGreater(p[0, 0], .5)
        np.testing.assert_allclose(p, proximity_evidence(distances[:, :, ::-1], 3, .5))
        missing = distances.copy(); missing[:, 1] = np.nan
        self.assertTrue(np.isnan(proximity_evidence(missing, 3, .5)).all())

    def test_jump_reliability_and_missing_speed(self):
        from experiments.observation import reliability
        xy = np.array([[0., 0.], [1., 0.], [np.nan, np.nan], [100., 0.], [100., 0.]])
        q = reliability(xy, np.ones((5, 2)), [(0, 5)], 25, 50)
        self.assertTrue(np.isnan(q['speed_mps'][3]))
        self.assertEqual(q['ball_quality'][2], 0)
        jump = reliability(np.array([[0., 0.], [100., 0.]]), np.ones((2, 2)), [(0, 2)], 25, 50)
        self.assertTrue((jump['ball_quality'] == 0).all())

    def test_viterbi_persistence_and_controlled_change(self):
        from experiments.decoder import decode
        scores = np.zeros((100, 3)); scores[:, 0] = 1; scores[:, 2] = -10
        scores[20, :2] = [0, 1]
        scores[50:, :2] = [0, 1]
        labels = decode(scores, [(0, 100)], np.ones(100, bool), 25, 1, .75)
        self.assertTrue((labels[:50] == 0).all())
        self.assertTrue((labels[50:] == 1).all())
        # No transition is allowed inside an unsupported gap.
        scores[30:70, :2] = 0
        supported = np.ones(100, bool); supported[30:70] = False
        labels = decode(scores, [(0, 100)], supported, 25, 1, .75)
        self.assertTrue((labels[30:70] == labels[29]).all())

    def test_actual_stoppage_and_no_tag_only_stoppage(self):
        from experiments.deadball import dead_evidence
        from experiments.config import Config
        c = Config()
        stationary = dead_evidence(np.zeros(20), np.zeros(20), np.full(20, 8.),
                                   np.ones(20), np.ones(20), c)
        self.assertTrue((stationary['logit'] > 0).all())
        unknown = dead_evidence(np.full(20, np.nan), np.full(20, np.nan),
                                np.full(20, np.nan), np.ones(20), np.zeros(20), c)
        self.assertTrue((unknown['logit'] == 0).all())
        self.assertFalse(unknown['available'].any())

    def test_decoder_half_reset_and_missing_half(self):
        from experiments.decoder import decode
        scores = np.array([[1., 0, -1]]*50 + [[0, 1., -1]]*50)
        supported = np.ones(100, bool)
        out = decode(scores, [(0, 50), (50, 100)], supported, 25, 1000, 1000)
        self.assertTrue((out[:50] == 0).all())
        self.assertTrue((out[50:] == 1).all())
        supported[:50] = False
        out = decode(scores, [(0, 50), (50, 100)], supported, 25, 1, .75)
        self.assertTrue((out[:50] == 3).all())

    def test_control_numerical_stability(self):
        from experiments.control import proximity_evidence
        distances = np.array([[[10., 11., 12.], [11., 12., 13.]]])
        for weight in (.5, 1.):
            p = proximity_evidence(distances, .001, weight)
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(axis=1), 1)
            self.assertGreater(p[0, 0], p[0, 1])

    def test_config_validation(self):
        from experiments.config import Config
        for kwargs in ({'proximity_m': 0}, {'team_weight': float('nan')}, {'top_k': 1.5}):
            with self.assertRaises(ValueError):
                Config(**kwargs)


class PipelineTests(unittest.TestCase):
    def test_combined_visible_stoppage_control_and_motion_gaps(self):
        from experiments.config import Config
        from experiments.model import infer
        n = 300
        xy = np.zeros((n, 2)); xy[:100, 0] = np.arange(100)*.1
        xy[100:200, 0] = 9.9
        xy[200:, 0] = 9.9+np.arange(100)*.1
        distances = np.full((n, 2, 3), 12.)
        distances[:100, 0, :] = [1, 2, 3]
        distances[200:, 1, :] = [1, 2, 3]
        raw = dict(ball_m=xy, distances=distances, team_counts=np.ones((n, 2)))
        baseline = np.r_[np.zeros(200, int), np.ones(100, int)]
        event = dict(code='SET PIECES', start_frame=100, end_frame=200)
        pred, details = infer(raw, baseline, [(0, n)], [event], Config(), 'E4')
        self.assertEqual(pred[50], 0)
        self.assertEqual(pred[150], 2)
        self.assertEqual(pred[250], 1)
        raw['ball_m'][100:200] = np.nan
        pred, details = infer(raw, baseline, [(0, n)], [event], Config(), 'E4')
        self.assertFalse(details['dead']['available'][100:200].any())
        self.assertTrue((pred[100:200] == pred[99]).all())

    def test_no_parameter_can_make_tag_only_missing_geometry_dead(self):
        from experiments.config import Config
        from experiments.model import infer
        n = 80
        raw = dict(ball_m=np.full((n, 2), np.nan), distances=np.full((n, 2, 3), np.nan),
                   team_counts=np.zeros((n, 2)))
        for candidate in ('E1', 'E2', 'E3', 'E4'):
            pred, _ = infer(raw, np.full(n, 2), [(0, n)],
                            [dict(code='SET PIECES', start_frame=0, end_frame=n)], Config(), candidate)
            self.assertTrue((pred == 3).all())


class DecoderOracleTests(unittest.TestCase):
    def test_viterbi_matches_exhaustive_paths(self):
        import itertools
        from experiments.decoder import decode
        rng = np.random.RandomState(7)
        costs = np.array([[0, 1, .75], [1, 0, .75], [.75, .75, 0]])
        for _ in range(12):
            scores = rng.normal(size=(5, 3))
            supported = np.r_[True, rng.randint(0, 2, 4).astype(bool)]
            path = decode(scores, [(0, 5)], supported, 1, 1, .75)
            def objective(states):
                if any(states[i] != states[i-1] and not supported[i] for i in range(1, 5)):
                    return -np.inf
                return sum(scores[i, states[i]] for i in range(5))-sum(
                    costs[states[i-1], states[i]] for i in range(1, 5))
            expected = max(objective(p) for p in itertools.product(range(3), repeat=5))
            self.assertAlmostEqual(objective(path), expected)

    def test_prediction_stats_respect_halves(self):
        from experiments.reporting import describe
        stats = describe(np.array([0, 0, 1, 1]), [(0, 2), (2, 4)])
        self.assertEqual(stats['predicted_possessions'], 2)
        self.assertEqual(stats['transitions'], 0)
