
import numpy as np
import scipy
import pandas
import unittest
import tempfile
import os

import bgshr


class TestExpectedPi0(unittest.TestCase):

    def test_3_bin_dfe(self):
        cols = ["uL", "r", "s", "Hl"]
        data = [[1e-8, 0, 0, 1e-3 / 2],
                [1e-8, 0, -0.5, 5e-4 / 2],
                [1e-8, 0, -1, 0]]
        df = pandas.DataFrame(data, columns=cols)
        dfes = [{"type": "gamma", "shape": 0.2, "scale": 0.02}]
        u = np.array([1e-8, 1e-8, 1e-8, 2e-8])
        elements = [np.array([[1, 3]])]
        e_pi0 = np.array([1e-3, 5e-4, 5e-4, 2e-3])
        pi0 = bgshr.Inference.expected_pi0(u, df, elements=elements, dfes=dfes)
        self.assertTrue(np.all(e_pi0 == pi0))


class TestPiDFE(unittest.TestCase):

    def test_3_bin_dfe(self):
        cols = ["r", "s", "Hl"]
        # `Hl` in the lookup table is *unsorted* H, hence the factor of two
        data = [[0, 0, 1e-3 / 2],
                [0, -0.5, 5e-4 / 2],
                [0, -1, 0]]
        df = pandas.DataFrame(data, columns=cols)
        # Approx. all weight is on "s=-0.5" bin
        dfe = {"type": "gamma", "shape": 0.2, "scale": 0.02}
        e_pi = 5e-4
        pi = bgshr.Inference._get_pi_dfe(df, dfe)
        self.assertTrue(np.isclose(pi, e_pi))

    def test_4_bin_dfe(self):
        # Manually calculate some DFE weights
        def get_dfe_weight(shape, scale, s0, s1):
            x0 = -s1
            x1 = -s0
            F0, F1 = scipy.stats.gamma.cdf([x0, x1], shape, scale=scale)
            return F1 - F0

        cols = ["r", "s", "Hl"]
        data = [[0, 0, 1e-3 / 2],
                [0, -0.01, 5e-4 / 2],
                [0, -0.1, 1e-4 / 2],
                [0, -1, 0]]
        df = pandas.DataFrame(data, columns=cols)
        a = 0.2
        b = 0.05
        dfe = {"type": "gamma", "shape": a, "scale": b}
        w_1 = get_dfe_weight(a, b, -0.055, 0)
        w_2 = get_dfe_weight(a, b, -0.55, -0.055)
        e_pi = w_1 * 5e-4 + w_2 * 1e-4
        pi = bgshr.Inference._get_pi_dfe(df, dfe)
        self.assertTrue(np.isclose(pi, e_pi))


class TestLikelihood(unittest.TestCase):

    def test_nd_ns(self):
        counts = (99, 45)
        e_nd = counts[0] * counts[1]
        n = np.sum(counts)
        nt = n * (n - 1) / 2
        e_ns = nt - e_nd
        nd, ns = bgshr.Inference.num_diff_same(counts)
        self.assertEqual(e_ns, ns)
        self.assertEqual(e_nd, nd)


