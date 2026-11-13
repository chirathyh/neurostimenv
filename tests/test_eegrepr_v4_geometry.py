import unittest
import numpy as np
from experiments.eeg_representation.h1.v4.geometry import (
    covariance_geometry, deduplicate_prefixes, grouped_geometry, innovation_geometry,
    geometry_controls, _parse_state, event_generator_inventory,
)


class GeometryTests(unittest.TestCase):
    def test_known_geometry_distinguishes_rank_from_manifold(self):
        result = geometry_controls(); self.assertTrue(result['passed'])
        self.assertEqual(result['controls']['circle']['numerical_rank'], 2)
        self.assertEqual(result['controls']['known_intrinsic_dimensions']['circle'], 1)
        self.assertFalse(result['controls']['nonlinear_id']['gate_passed'])

    def test_rank_ceiling_and_declared_scaling(self):
        x = np.random.default_rng(2).normal(size=(5, 40))
        result = covariance_geometry(x)
        self.assertEqual(result['rank_ceiling'], 4); self.assertLessEqual(result['numerical_rank'], 4)
        with self.assertRaises(ValueError): covariance_geometry(x, scaling='train_coordinate_standardized')
        self.assertEqual(covariance_geometry(x[:1])['status'], 'unavailable')

    def test_dedup_prefix_and_detect_nonidentical_fork(self):
        x = np.array([[1., 2.], [1., 2.], [2., 3.]])
        np.testing.assert_array_equal(deduplicate_prefixes(x, ['a', 'a', 'b']), [0, 2])
        with self.assertRaises(ValueError): deduplicate_prefixes(x, ['a', 'a', 'a'])

    def test_within_groups_excludes_between_group_location(self):
        x = np.array([[0., 0.], [0., 1.], [100., 0.], [100., 1.]])
        result = grouped_geometry(x, [1, 1, 2, 2], ['a'] * 4)
        self.assertGreater(result['pooled']['covariance_trace'], 1000)
        self.assertLess(result['pooled_group_centered']['covariance_trace'], 1)

    def test_innovation_crosscov_unclipped_and_disjoint(self):
        rng = np.random.default_rng(4)
        y = rng.normal(size=(12, 8, 7))
        result, a = innovation_geometry(y)
        self.assertFalse(result['psd_clipping'])
        self.assertTrue(np.any(a['symmetric_cross_eigenvalues'] < 0))
        for row in result['half_indices']:
            self.assertFalse(set(row['first_half']) & set(row['second_half']))
        expected = np.cov(y[:, :4].mean(1).T, y[:, 4:].mean(1).T)[:7, 7:]
        np.testing.assert_allclose(a['cross_covariance'], expected, atol=1e-14)

    def test_innovation_total_signal_noise_synthetic_control(self):
        rng = np.random.default_rng(19); means = rng.normal(size=(3000, 3))
        y = means[:, None] + rng.normal(scale=.5, size=(3000, 16, 3))
        result, arrays = innovation_geometry(y)
        self.assertAlmostEqual(result['trace_mean_response_estimate'], 3., delta=.15)
        self.assertAlmostEqual(result['trace_conditional_variance'], .75, delta=.04)
        self.assertAlmostEqual(result['trace_total_one_future_estimate'], 3.75, delta=.2)

    def test_common_event_inventory_counts_active_generators(self):
        private = event_generator_inventory(2560, 2, 0.)
        mixed = event_generator_inventory(2560, 2, .02)
        common = event_generator_inventory(2560, 2, 1.)
        self.assertEqual(private['conditional_independent_poisson_generators_total'], 2560)
        self.assertEqual(mixed['conditional_independent_poisson_generators_total'], 2562)
        self.assertEqual(common['conditional_independent_poisson_generators_total'], 2)
        self.assertEqual(common['independent_private_event_generators'], 0)
        self.assertEqual(common['stored_contact_event_streams'], 2560)

    def test_contact_aggregation_is_bookkeeping_not_dimension(self):
        # One cell: soma v/vext + 3 gates, two same-location synapse A/B,
        # followed by 21 passive segment pairs.
        syn = np.array([[0, 0, .5, .5, 3., 0.], [0, 0, .5, .5, 3., 0.]])
        state = np.r_[[-65., 0., .1, .2, .3, 1., 2., 4., 8.], np.zeros(42)]
        mem, agg = _parse_state(state, syn, 1, 22)
        self.assertEqual(len(mem), 47); self.assertEqual(len(agg), 1)
        np.testing.assert_array_equal(next(iter(agg.values())), [5., 10.])
        with self.assertRaises(ValueError): _parse_state(np.r_[state, 0.], syn, 1, 22)



class LargeFiniteCovarianceTests(unittest.TestCase):
    def test_participation_ratio_is_stable_when_covariance_square_overflows(self):
        from experiments.eeg_representation.h1.v4.geometry import covariance_geometry
        x=np.asarray([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        small=covariance_geometry(x)
        large=covariance_geometry(x*1e100)
        self.assertAlmostEqual(large['participation_ratio'],small['participation_ratio'])
        self.assertAlmostEqual(large['participation_ratio'],2.)
        self.assertTrue(np.isfinite(large['covariance_trace']))

if __name__ == '__main__': unittest.main()
