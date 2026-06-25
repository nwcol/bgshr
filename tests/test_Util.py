
import numpy as np
import scipy
import pandas
import unittest
import tempfile
import os

import bgshr


# -----------------------------------------------------------------------------
# Lookup table manipulations
# -----------------------------------------------------------------------------


class TestScaleLookupTable(unittest.TestCase):

    def test_equilibrium_table(self):
        tbl_content = """r,s,uL,Order,Generation,Hr,pi0,B,uR,Hl,piN_pi0,piN_piS,Ns,Ts
0.01,-0.01,1e-8,0,0,0.000396,0.0004,0.99,1e-8,0.002,0,0,10000,0"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as flike:
            flike.write(tbl_content)
            path = flike.name

        df = pandas.read_csv(path)

        # Doubling N_target should halve parameters
        _df = bgshr.Util.scale_lookup_table(df, 2e4)
        self.assertTrue(np.all(_df["r"] == df["r"] / 2))
        self.assertTrue(np.all(_df["s"] == df["s"] / 2))
        self.assertTrue(np.all(_df["uL"] == df["uL"] / 2))
        self.assertTrue(np.all(_df["Ns"] == 2e4))

        # Halving N_target should double parameters
        _df = bgshr.Util.scale_lookup_table(df, 5e3)
        self.assertTrue(np.all(_df["r"] == df["r"] * 2))
        self.assertTrue(np.all(_df["s"] == df["s"] * 2))
        self.assertTrue(np.all(_df["uL"] == df["uL"] * 2))
        self.assertTrue(np.all(_df["Ns"] == 5e3))

    def test_2_epoch_table(self):
        tbl_content = """r,s,uL,Order,Generation,Hr,pi0,B,uR,Hl,piN_pi0,piN_piS,Ns,Ts
0.01,-0.01,1e-8,400,0,0.000396,0.0004,0.99,1e-8,0.002,0,0,20000;10000,0;10000"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as flike:
            flike.write(tbl_content)
            path = flike.name

        df = pandas.read_csv(path)

        # Doubling N_target should halve parameters
        _df = bgshr.Util.scale_lookup_table(df, 2e4)
        self.assertTrue(np.all(_df["r"] == df["r"] / 2))
        self.assertTrue(np.all(_df["s"] == df["s"] / 2))
        self.assertTrue(np.all(_df["uL"] == df["uL"] / 2))
        self.assertTrue(np.all(_df["Ns"] == "40000;20000"))
        self.assertTrue(np.all(_df["Ts"] == "0;20000"))

        # Halving N_target should double parameters
        _df = bgshr.Util.scale_lookup_table(df, 5e3)
        self.assertTrue(np.all(_df["r"] == df["r"] * 2))
        self.assertTrue(np.all(_df["s"] == df["s"] * 2))
        self.assertTrue(np.all(_df["uL"] == df["uL"] * 2))
        self.assertTrue(np.all(_df["Ns"] == "10000;5000"))
        self.assertTrue(np.all(_df["Ts"] == "0;5000"))

    def test_r_removal(self):
        # The scaling function should remove r scaled to be > 0.5
        tbl_content = """r,s,uL,Order,Generation,Hr,pi0,B,uR,Hl,piN_pi0,piN_piS,Ns,Ts
0.25,-0.01,1e-8,0,0,0.000396,0.0004,0.99,1e-8,0.002,0,0,10000,0
0.3,-0.01,1e-8,0,0,0.000396,0.0004,0.99,1e-8,0.002,0,0,10000,0"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as flike:
            flike.write(tbl_content)
            path = flike.name

        df = pandas.read_csv(path)

        _df = bgshr.Util.scale_lookup_table(df, 5e3)
        self.assertTrue(len(_df) == 1)
        self.assertTrue(next(iter(_df["r"])) == 0.5)


class TestFillInLookupTable(unittest.TestCase):

    def _test_step_spacing(self):
        fname = os.path.join(os.path.dirname(__file__),
            "data/lookup_tbl_one_line_equilibrium.csv")
        df = pandas.read_csv(fname)
        df1 = bgshr.Util.fill_in_lookup_table(df, n_steps=4, max_M=10)
        self.assertTrue(len(df1) == 5)
        self.assertTrue(len(np.unique(df1["M"])) == 5)
        self.assertTrue(np.max(df1["M"]) == 10)
        self.assertTrue(np.all(df[df["M"] == 10]["B"] == 1))


class TestConvertLookupTableToMorgans(unittest.TestCase):

    def _test_truncation(self):
        # Should cut off the table row with r = 0.45 and insert one with
        # r ~ 0.43
        fname = os.path.join(os.path.dirname(__file__),
            "data/lookup_tbl_one_line_equilibrium.csv")
        df = pandas.read_csv(fname)
        df_extend = df.copy()
        df_extend["r"] = 0.45
        df = pandas.concat([df, df_extend])
        df1 = bgshr.Util.convert_lookup_table_to_morgans(df, max_M=1)
        self.assertTrue(np.max(df1["M"]) == 1)
        self.assertTrue(np.max(df1["r"]) == bgshr.Util.haldane_map_function(1))
        self.assertTrue(np.all(df1[df1["M"] == 1]["B"] == 1))

    def _test_upper_bound(self):
        fname = os.path.join(os.path.dirname(__file__),
            "data/lookup_tbl_one_line_equilibrium.csv")
        df = pandas.read_csv(fname)
        df_extend = df.copy()
        df_extend["r"] = 0.5
        df = pandas.concat([df, df_extend])
        df1 = bgshr.Util.convert_lookup_table_to_morgans(df)
        max_M = np.max(df1["M"])
        self.assertTrue(np.isfinite(max_M))
        self.assertTrue(np.max(df1["r"]) == bgshr.Util.haldane_map_function(max_M))
        self.assertTrue(np.all(df1[df1["M"] == max_M]["B"] == 1))


# -----------------------------------------------------------------------------
# DFEs
# -----------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    pass


class TestWeightsGammaDFE(unittest.TestCase):

    def test_total_weight(self):
        # The sum of DFE weights should equal 1.
        a = 0.3
        b = 0.01
        ss = np.append(-np.logspace(0, -5, 48), 0)
        self.assertTrue(
            np.isclose(np.sum(bgshr.Util.weights_gamma_dfe(ss, a, b)), 1))
        ss = np.append(-np.logspace(0, -6, 48), 0)
        self.assertTrue(
            np.isclose(np.sum(bgshr.Util.weights_gamma_dfe(ss, a, b)), 1))
        ss = np.append(-np.logspace(0, -6, 100), 0)
        self.assertTrue(
            np.isclose(np.sum(bgshr.Util.weights_gamma_dfe(ss, a, b)), 1))

    def test_zero_bin(self):
        # The bin corresponding to s = 0 should receive zero probability mass.
        a = 0.3
        b = 0.01
        ss = np.append(-np.logspace(0, -6, 48), 0)
        self.assertEqual(bgshr.Util.weights_gamma_dfe(ss, a, b)[-1], 0)

    def test_discretization(self):
        # Compare expected mass for a few bins to the calculated value
        def expected_weight(shape, scale, s0, s1):
            x0 = -s1
            x1 = -s0
            F0, F1 = scipy.stats.gamma.cdf([x0, x1], shape, scale=scale)
            return F1 - F0

        a = 0.25
        b = 0.02
        ss = np.append(-np.logspace(0, -6, 48), 0)
        ws = bgshr.Util.weights_gamma_dfe(ss, a, b)

        # The zeroth bin extends to -inf
        i = 0
        midpoint_1 = (ss[i] + ss[i + 1]) / 2
        ew = expected_weight(a, b, -np.inf, midpoint_1)
        self.assertEqual(ws[0], ew)

        # The last nonzero bin extends to zero
        i = -2
        midpoint_0 = (ss[i - 1] + ss[i]) / 2
        ew = expected_weight(a, b, midpoint_0, 0)
        self.assertEqual(ws[-2], ew)

        i = 1
        midpoint_0 = (ss[i - 1] + ss[i]) / 2
        midpoint_1 = (ss[i] + ss[i + 1]) / 2
        ew = expected_weight(a, b, midpoint_0, midpoint_1)
        self.assertEqual(ws[i], ew)

        i = 10
        midpoint_0 = (ss[i - 1] + ss[i]) / 2
        midpoint_1 = (ss[i] + ss[i + 1]) / 2
        ew = expected_weight(a, b, midpoint_0, midpoint_1)
        self.assertEqual(ws[i], ew)

        i = 45
        midpoint_0 = (ss[i - 1] + ss[i]) / 2
        midpoint_1 = (ss[i] + ss[i + 1]) / 2
        ew = expected_weight(a, b, midpoint_0, midpoint_1)
        self.assertEqual(ws[i], ew)


class TestWeightsGammaNeutralDFE(unittest.TestCase):

    def test_neutral_mass(self):
        def expected_weight(shape, scale, s0, s1):
            x0 = -s1
            x1 = -s0
            F0, F1 = scipy.stats.gamma.cdf([x0, x1], shape, scale=scale)
            return F1 - F0

        a = 0.25
        b = 0.02
        p = 0.3
        ss = np.append(-np.logspace(0, -6, 48), 0)
        ws = bgshr.Util.weights_gamma_neutral_dfe(ss, a, b, p)
        self.assertTrue(np.isclose(np.sum(ws), 1))
        self.assertTrue(ws[-1] == p)
        self.assertTrue(np.isclose(np.sum(ws[:-1]), 1 - p))

        # Test a particular bin
        i = 10
        midpoint_0 = (ss[i - 1] + ss[i]) / 2
        midpoint_1 = (ss[i] + ss[i + 1]) / 2
        ew = (1 - p) * expected_weight(a, b, midpoint_0, midpoint_1)
        self.assertEqual(ws[i], ew)


# -----------------------------------------------------------------------------
# Elements and masks
# -----------------------------------------------------------------------------


class TestElementManips(unittest.TestCase):

    def test_collapse_elements(self):
        redundant_elems = np.array([
            [5, 10],
            [8, 12],
            [13, 20],
            [15, 25]])
        e_collapsed = np.array([[5, 12], [13, 25]])
        collapsed = bgshr.Util.collapse_elements(redundant_elems)
        self.assertEqual(len(e_collapsed), len(collapsed))
        self.assertTrue(np.all(e_collapsed == collapsed))

    def test_break_up_elements(self):
        pass

    def test_resolve_elements(self):
        elements = [
            np.array([[25, 50]]),
            np.array([[15, 30], [45, 60]]),
            np.array([[0, 30], [55, 75]]),
            np.array([[55, 60]])
        ]
        e_resolved = [
            np.array([[25, 50]]),
            np.array([[15, 25], [50, 60]]),
            np.array([[0, 15], [60, 75]]),
            np.stack([[], []], 1)
        ]
        resolved = bgshr.Util.resolve_elements(elements)
        self.assertTrue(np.all(resolved[0] == e_resolved[0]))
        self.assertTrue(np.all(resolved[1] == e_resolved[1]))
        self.assertTrue(np.all(resolved[2] == e_resolved[2]))
        self.assertTrue(np.all(resolved[3] == e_resolved[3]))

    def test_elements_to_mask(self):
        elements = np.array([[1, 2], [3, 4], [4, 6]])
        e_mask = np.array([1, 0, 1, 0, 0, 0], bool)
        mask = bgshr.Util.elements_to_mask(elements)
        self.assertTrue(np.all(mask == e_mask))

        # Alternative algorithm
        elements = np.array([
            [45, 123],
            [455, 899],
            [920, 1231],
            [1267, 1455]])
        e_mask = np.full(elements[-1, 1], 1, bool)
        for start, end in elements:
            for i in range(start, end):
                e_mask[i] = False
        mask = bgshr.Util.elements_to_mask(elements)
        self.assertTrue(np.all(mask == e_mask))

    def test_bounded_elements_to_mask(self):
        elements = np.array([[1, 2], [3, 4], [4, 6]])
        L = 5
        e_mask = np.array([1, 0, 1, 0, 0], bool)
        mask = bgshr.Util.elements_to_mask(elements, L=L)
        self.assertTrue(np.all(mask == e_mask))

    def test_mask_to_elements(self):
        mask = np.array([False, False, True, False, False, True, False])
        e_elems = np.array([[0, 2], [3, 5], [6, 7]])
        elems = bgshr.Util.mask_to_elements(mask)
        self.assertTrue(np.all(elems == e_elems))

        mask = np.array([True, False, True, False, True, True, True])
        e_elems = np.array([[1, 2], [3, 4]])
        elems = bgshr.Util.mask_to_elements(mask)
        self.assertTrue(np.all(elems == e_elems))

    def test_intersect_elements(self):
        elements = [
            np.array([[0, 100], [150, 200]]),
            np.array([[10, 20], [90, 160], [190, 255]])]
        e_intersect = np.array([[10, 20], [90, 100], [150, 160], [190, 200]])
        intersect = bgshr.Util.intersect_elements(elements)
        self.assertTrue(np.all(intersect == e_intersect))

    def test_intersect_elements_bounded(self):
        elements = [
            np.array([[0, 100], [150, 200]]),
            np.array([[10, 20], [90, 160], [190, 255]])]
        L = 150
        e_intersect = np.array([[10, 20], [90, 100]])
        intersect = bgshr.Util.intersect_elements(elements, L=L)
        self.assertTrue(np.all(intersect == e_intersect))

    def test_merge_elements(self):
        elements = [np.array([[50, 60]]), np.array([[55, 65], [90, 100]])]
        e_merge = np.array([[50, 65], [90, 100]])
        merge = bgshr.Util.merge_elements(elements)
        self.assertTrue(np.all(merge == e_merge))

    def test_subtract_elements(self):
        elems0 = np.array([[0, 100]])
        elems1 = np.array([[40, 50]])
        e_result = np.array([[0, 40], [50, 100]])
        result = bgshr.Util.subtract_elements(elems0, elems1)
        self.assertTrue(np.all(result == e_result))

# -----------------------------------------------------------------------------
# Mutation rate averaging/aggregation
# -----------------------------------------------------------------------------


class TestComputeWindowedAverages(unittest.TestCase):

    def test_compute_windowed_averages(self):
        windows = np.array([[0, 2], [2, 4], [4, 5]])
        site_map = np.ma.array([1, 1, 2, 0, 0], mask=[0, 0, 0, 1, 1])
        avgs, num_sites = bgshr.Util.compute_window_averages(
            windows, site_map)
        e_avgs = np.array([1, 2, 0])
        e_num_sites = np.array([2, 1, 0])
        self.assertTrue(np.all(e_avgs == avgs))
        self.assertTrue(np.all(num_sites == e_num_sites))


class TestWindowMutationRates(unittest.TestCase):

    def test_window_mutation_rates(self):
        windows = np.array([[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]])
        elems = np.array([[5, 9], [15, 25], [29, 30], [35, 36]])
        u_arr = np.ma.array(np.full(50, 1e-8), mask=np.zeros(50))
        u_arr[29] = 1e-7
        u_arr.mask[35] = True
        U_arr, sites = bgshr.Util.compute_window_mutation_rates(
            windows, elems, u_arr, fill_val="mean")
        mean_u = np.mean(u_arr)
        e_U = np.array([4e-8, 5e-8, 1.5e-7, mean_u, 0])
        e_sites = np.array([4, 5, 6, 1, 0])
        self.assertTrue(np.all(U_arr == e_U))
        self.assertTrue(np.all(sites == e_sites))


class TestDecomposeElements(unittest.TestCase):

    def test_decompose_elements(self):
        windows = np.array([[0, 10], [10, 20], [20, 30], [30, 40]])
        elements = np.array([[5, 15], [16, 18], [20, 37]])
        e_elems = np.array([
            [5, 10],
            [10, 15],
            [16, 18],
            [20, 30],
            [30, 37]])
        e_index = np.array([0, 1, 1, 2, 3])
        index, elems = bgshr.Util.decompose_elements(windows, elements)
        self.assertTrue(np.all(index == e_index))
        self.assertTrue(np.all(elems == e_elems))


class TestComputeElementMutationRates(unittest.TestCase):

    def test_compute_element_mutation_rates(self):
        elements = np.array([[0, 2], [2, 4], [4, 5]])
        u_arr = np.ma.array([1e-8, 1e-8, 2e-8, 2e-8, 0], mask=[0, 0, 0, 0, 1])
        result = bgshr.Util.compute_element_mutation_rates(
            elements, u_arr, fill_val="mean")
        e_result = np.array([1e-8, 2e-8, 1.5e-8])
        self.assertTrue(np.all(np.isclose(result, e_result)))

        result = bgshr.Util.compute_element_mutation_rates(
            elements, u_arr, fill_val=1e-8)
        e_result = np.array([1e-8, 2e-8, 1e-8])
        self.assertTrue(np.all(result == e_result))

    def test_nan_rate_error(self):
        elements = np.array([[0, 2], [2, 4], [4, 5]])
        u_arr = np.ma.array([1e-8, 1e-8, 2e-8, 2e-8, np.nan])
        with self.assertRaises(AssertionError):
            result = bgshr.Util.compute_element_mutation_rates(
                elements, u_arr, fill_val=1e-8)

# -----------------------------------------------------------------------------
# Recombination maps
# -----------------------------------------------------------------------------


class TestRecombinationMaps(unittest.TestCase):

    def test_build_uniform_rmap(self):
        L = 100
        r = 1e-8
        rmap = bgshr.Util.build_uniform_rmap(r, L)
        self.assertEqual(rmap(L), r * L)
        self.assertEqual(rmap(L / 2), r * L / 2)
        self.assertEqual(rmap(1), r)
        self.assertEqual(rmap(0), 0)

# -----------------------------------------------------------------------------
# Misc other utilities
# -----------------------------------------------------------------------------


