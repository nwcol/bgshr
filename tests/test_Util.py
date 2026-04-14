
import bgshr
import numpy as np
import pandas
import unittest
import os


class TestScaleLookupTable(unittest.TestCase):

    def test_equilibrium_table(self):
        fname = os.path.join(os.path.dirname(__file__),
            "data/lookup_tbl_one_line_equilibrium.csv")
        df = pandas.read_csv(fname)

        # Doubling N_target halfs parameters
        df1 = bgshr.Util.scale_lookup_table(df, 2e4)
        self.assertTrue(np.all(df1["r"] == df["r"] / 2))
        self.assertTrue(np.all(df1["s"] == df["s"] / 2))
        self.assertTrue(np.all(df1["uL"] == df["uL"] / 2))
        self.assertTrue(np.all(df1["Ns"] == 2e4))

        # Halving N_target doubles parameters
        df1 = bgshr.Util.scale_lookup_table(df, 5e3)
        self.assertTrue(np.all(df1["r"] == df["r"] * 2))
        self.assertTrue(np.all(df1["s"] == df["s"] * 2))
        self.assertTrue(np.all(df1["uL"] == df["uL"] * 2))
        self.assertTrue(np.all(df1["Ns"] == 5e3))

    def test_2_epoch_table(self):
        fname = os.path.join(os.path.dirname(__file__),
            "data/lookup_tbl_one_line_2_epochs.csv")
        df = pandas.read_csv(fname)

        # Doubling N_target halfs parameters
        df1 = bgshr.Util.scale_lookup_table(df, 2e4)
        self.assertTrue(np.all(df1["r"] == df["r"] / 2))
        self.assertTrue(np.all(df1["s"] == df["s"] / 2))
        self.assertTrue(np.all(df1["uL"] == df["uL"] / 2))
        self.assertTrue(np.all(df1["Ns"] == "40000;20000"))
        self.assertTrue(np.all(df1["Ts"] == "0;20000"))

        # Halving N_target doubles parameters
        df1 = bgshr.Util.scale_lookup_table(df, 5e3)
        self.assertTrue(np.all(df1["r"] == df["r"] * 2))
        self.assertTrue(np.all(df1["s"] == df["s"] * 2))
        self.assertTrue(np.all(df1["uL"] == df["uL"] * 2))
        self.assertTrue(np.all(df1["Ns"] == "10000;5000"))
        self.assertTrue(np.all(df1["Ts"] == "0;5000"))

    def test_removing_invalid_rs(self):
        fname = os.path.join(os.path.dirname(__file__),
            "data/lookup_tbl_equilibrium.csv.gz")
        df = pandas.read_csv(fname)
        df1 = bgshr.Util.scale_lookup_table(df, 1e3)
        self.assertTrue(np.all(df1["r"] < 0.5))


class TestExtendLookupTableR(unittest.TestCase):

    def test_step_spacing(self):
        fname = os.path.join(os.path.dirname(__file__),
            "data/lookup_tbl_one_line_equilibrium.csv")
        df = pandas.read_csv(fname)
        df1 = bgshr.Util.extend_lookup_table_r(df, n_steps=4)
        self.assertTrue(len(df1) == 5)
        self.assertTrue(len(np.unique(df1["r"])) == 5)
        self.assertTrue(np.max(df1["r"]) == 0.5)
        self.assertTrue(np.all(df[df["r"] == 0.5]["B"] == 1))


class TestConvertLookupTableToMorgans(unittest.TestCase):

    def test_truncation(self):
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

    def test_upper_bound(self):
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


def test_uniform_rmap():
    L = 100
    for r in [0, 1e-8, 1.5e-6]:
        rmap = bgshr.Util.build_uniform_rmap(r, L)
        assert rmap(L) == r * L
        assert rmap(L / 2) == r * L / 2
        assert rmap(0) == 0
