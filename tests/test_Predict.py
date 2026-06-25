
import numpy as np
import scipy
import unittest
import tempfile
import os

import bgshr


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


class TestGetBMap(unittest.TestCase):

    def test_uniform_map(self):
        unif_B = 0.99
        xs = [0, 100]
        Bs = [unif_B, unif_B]
        Bmap = bgshr.Predict.get_Bmap(xs, Bs)
        self.assertTrue(np.all(Bmap(np.arange(xs[-1] + 1)) == unif_B))


class TestGetSIndices(unittest.TestCase):
    # This function retrieves indices of the nearest s-coefficents in the grid

    def test_get_s_indices(self):
        s_grid = np.array([-1, -0.5, -0.1, -0.05, -0.01, 0])
        s_vals = np.array([-0.75])
        e_idx_0, e_idx_1 = (np.array([0]), np.array([1]))
        idx_0, idx_1 = bgshr.Predict._get_s_indices(s_vals, s_grid)
        self.assertEqual(idx_0[0], e_idx_0[0])
        self.assertEqual(idx_1[0], e_idx_1[0])

        s_vals = np.array([-1e-3])
        e_idx_0, e_idx_1 = (np.array([4]), np.array([5]))
        idx_0, idx_1 = bgshr.Predict._get_s_indices(s_vals, s_grid)
        self.assertEqual(idx_0[0], e_idx_0[0])
        self.assertEqual(idx_1[0], e_idx_1[0])


class TestGetSFacs(unittest.TestCase):

    def test_get_s_facs(self):
        s_grid = np.array([-1, -0.5, -0.1, -0.05, -0.01, 0])
        s_vals = np.array([-0.75])
        idx_0, idx_1 = bgshr.Predict._get_s_indices(s_vals, s_grid)
        facs_0, facs_1 = bgshr.Predict._get_s_facs(s_vals, s_grid, idx_0, idx_1)
        # s_val -0.75 is halfway between -1, -0.5
        self.assertEqual(facs_0[0], 0.5)
        self.assertEqual(facs_1[0], 0.5)

        s_vals = np.array([-0.6])
        idx_0, idx_1 = bgshr.Predict._get_s_indices(s_vals, s_grid)
        facs_0, facs_1 = bgshr.Predict._get_s_facs(s_vals, s_grid, idx_0, idx_1)
        self.assertTrue(np.isclose(facs_0[0], 0.2))
        self.assertTrue(np.isclose(facs_1[0], 0.8))

        s_vals = np.array([-0.01])
        idx_0, idx_1 = bgshr.Predict._get_s_indices(s_vals, s_grid)
        self.assertEqual(idx_0[0], 3)
        self.assertEqual(idx_1[0], 4)
        facs_0, facs_1 = bgshr.Predict._get_s_facs(s_vals, s_grid, idx_0, idx_1)
        self.assertTrue(np.isclose(facs_0[0], 0))
        self.assertTrue(np.isclose(facs_1[0], 1))


class TestDistances(unittest.TestCase):
    # These functions calculate genetic map distances - not recombination
    # fractions `r`.

    def test_get_distances(self):
        r = 1e-8
        L = 10000
        rmap = bgshr.Util.build_uniform_rmap(r, L)
        xs = np.array([500, 1500])
        windows = np.array([[100, 200], [1300, 1500], [1900, 2000]])
        dists = bgshr.Predict._get_distances(xs, windows, rmap)
        self.assertTrue(np.all(dists >= 0))

    def test_get_signed_distances(self):
        r = 1e-8
        L = 10000
        rmap = bgshr.Util.build_uniform_rmap(r, L)
        xs = np.array([500, 1500])
        windows = np.array([[100, 200], [1300, 1500], [1900, 2000]])
        midpoints = np.mean(windows, axis=1)
        xs_map = r * xs
        mids_map = r * midpoints
        e_dists = np.array([
            [mids_map[0] - xs_map[0], mids_map[0] - xs_map[1]],
            [mids_map[1] - xs_map[0], mids_map[1] - xs_map[1]],
            [mids_map[2] - xs_map[0], mids_map[2] - xs_map[1]],
            ])
        dists = bgshr.Predict._get_signed_distances(xs, windows, rmap)
        self.assertTrue(np.all(dists == e_dists))


class TestAdjustMutationArrays(unittest.TestCase):

    def test_with_uniform_map(self):
        unif_B = 0.95
        xs = [0, 100]
        Bmap = bgshr.Predict.get_Bmap(xs, [unif_B, unif_B])
        U_arrs = [np.array([1e-4, 1e-3])]
        windows = np.array([[0, 50], [50, 100]])
        adjusted_U = bgshr.Predict._adjust_mutation_arrays(
            U_arrs, windows, Bmap)
        e_U = U_arrs[0] * unif_B
        self.assertTrue(np.all(adjusted_U[0] == e_U))


