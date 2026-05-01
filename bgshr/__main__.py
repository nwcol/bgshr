"""
Command-line interface
"""

import argparse
import numpy as np
import pandas

from bgshr import Util, ClassicBGS, Predict, Inference


class Command():
    """
    The hierarchy of command classes is:

    Command
        CommonCommand
            ComputePiCommand
            CommonPredictCommand
                PredictBCommand
                FitNeCommand
    """

    def __init__(self, subparsers, subcommand):
        self.parser = subparsers.add_parser(subcommand)
        self.parser.set_defaults(func=self)


class CommonCommand(Command):
    """
    Arguments shared by commands that compute pi and/or B.
    """

    def __init__(self, subparsers, subcommand):
        super().__init__(subparsers, subcommand)
        # TODO implement file-based DFE spec.
        #self.parser.add_argument(
        #    "-d",
        #    "--dfes",
        #    type=str,
        #    default=None,
        #    help="path to file defining DFE parameters (.yaml)"
        #)
        self.parser.add_argument(
            "-b",
            "--bed",
            type=str, 
            required=True, 
            nargs="*",
            help="path to file(s) defining constrained sites (.bed)"
        )
        self.parser.add_argument(
            "-t",
            "--lookup_tbl",
            type=str, 
            required=True,
            help="path to lookup table file (.csv)"
        )
        self.parser.add_argument(
            "--rmap",
            type=str,
            default=None,
            help="path to recombination map file (.bedgraph, .csv or .txt)"
        )
        self.parser.add_argument(
            "--rmap_pos_col",
            type=str,
            default="Position(bp)",
            help="recombination map position column (default 'Position(bp)')"
        )
        self.parser.add_argument(
            "--rmap_rate_col",
            type=str,
            default="Rate(cM/Mb)",
            help="recombination map rate column (default 'Rate(cM/Mb)')"
        )
        self.parser.add_argument(
            "-r",
            "--rec_rate",
            type=float,
            default=None,
            help="uniform recombination rate"
        )
        self.parser.add_argument(
            "-L",
            "--L",
            type=int,
            default=None,
            help="chromosome length (defaults to XXXX)"
        )
        self.parser.add_argument(
            "--umap",
            type=str,
            default=None,
            help="mutation map file (.bedgraph, .csv, .txt or .npy)"
        )
        self.parser.add_argument(
            "--umap_rate_col",
            type=str,
            default="rate",
            help="mutation map rate column (default 'rate')"
        )
        self.parser.add_argument(
            "-u",
            "--mut_rate",
            type=float,
            default=None,
            help="uniform mutation rate"
        )
        self.parser.add_argument(
            "--mask",
            type=str,
            default=None,
            help="genomic mask for filtering expected diversity"
        )
        self.parser.add_argument(
            "--shapes",
            type=float,
            required=True,
            nargs="*",
            help="shape parameter(s) of gamma DFE(s)"
        )
        self.parser.add_argument(
            "--scales",
            type=float,
            required=True,
            nargs="*",
            help="scale parameter(s) of gamma DFE(s)"
        )
        self.parser.add_argument(
            "--p_neus",
            type=float,
            default=None,
            nargs="*",
            help=""
        )
        self.parser.add_argument(
            "--resolution",
            type=int,
            default=None,
            help="window size of output table (defaults to spacing)"
        )
        self.parser.add_argument(
            "--cbgs_start",
            type=float,
            default=None,
            help="highest s-coefficient to use in CBGS lookup table extension"
        )
        self.parser.add_argument(
            "--n_s_cbgs",
            type=int,
            default=20,
            help="number of s-coefficients to use in CBGS lookup table extension (default 20)"
        )
        self.parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="path for output file (.csv)"
        )
        self.parser.add_argument(
            "--rich",
            action="store_true",
            help="save extra data, including deleterious pi and mutation rate"
        )
        self.parser.add_argument(
            "-o",
            "--out",
            type=str,
            required=True,
            help="path for output file (.csv)"
        )


class ComputePiCommand(CommonCommand):

    def __init__(self, subparsers):
        super().__init__(subparsers, "compute_pi")
        self.parser.add_argument(
            "-B",
            "--Bmap",
            type=str,
            required=True,
            help="path to input B-map file (.csv)"
        )
        self.parser.add_argument(
            "--Ne",
            type=float,
            default=None,
            help="effective population size parameter"
        )

    def __call__(self, args):
        compute_pi(args)


class CommonPredictCommand(CommonCommand):
    """
    Arguments shared by commands that predict B.
    """

    def __init__(self, subparsers, subcommand):
        super().__init__(subparsers, subcommand)
        self.parser.add_argument(
            "-n",
            "--n_cores",
            type=int,
            default=1,
            help=""
        )
        self.parser.add_argument(
            "--chunk_size",
            type=int,
            default=100,
            help=""
        )
        self.parser.add_argument(
            "--B_unlinked",
            type=float,
            default=None,
            help=""
        )
        self.parser.add_argument(
            "-c",
            "--n_corrs",
            type=int,
            default=0,
            help="number of interference corrections to perform (default 0)"
        )
        self.parser.add_argument(
            "--save_corrs",
            action="store_true",
            help="retain each interference correction in output file"
        )
        self.parser.add_argument(
            "--spacing",
            type=int,
            default=1000,
            help="spacing between focal sites for B prediction (default 1000)"
        )
        self.parser.add_argument(
            "--window_size",
            type=int,
            default=1000,
            help="window size for aggregating constrained sites (default 1000)"
        )


class PredictBCommand(CommonPredictCommand):

    def __init__(self, subparsers):
        super().__init__(subparsers, "predict_B")
        self.parser.add_argument(
            "--Ne",
            type=float,
            default=None,
            help="effective population size parameter"
        )

    def __call__(self, args):
        predict_B(args)


class FitNeCommand(CommonPredictCommand):

    def __init__(self, subparsers):
        super().__init__(subparsers, "fit_Ne")
        self.parser.add_argument(
            "--ndns",
            type=str,
            required=True,
            help="path to .npy file holding site-resolution diversity data"
        )
        self.parser.add_argument(
            "--Ne0",
            type=float,
            required=True,
            help="initial guess of the effective population size parameter"
        )
        self.parser.add_argument(
            "--maxiter",
            type=int,
            required=True,
            help="maximum number of optimization iterations"
        )
        self.parser.add_argument(
            "--log_out",
            type=str,
            required=True,
            help="path for output log file (.txt)"
        )

    def __call__(self, args):
        fit_Ne(args)


def compute_pi(args):
    """
    Loads data and a precomputed B-map, and computes expected pi.
    """

    if args.verbose:
        print(Util._get_time(), "loading data")

    # DFE handling
    dfes = get_dfes(args.shapes, args.scales, args.p_neus)

    # The lookup table is used to find deleterious/neutral expected pi0.
    df, splines = get_lookup_table(
        args.lookup_tbl,
        Ne=args.Ne,
        n_s_cbgs=args.n_s_cbgs,
        cbgs_start=args.cbgs_start,
        verbose=args.verbose)

    # Load mutation rates
    if args.umap is None and args.L is None:
        raise ValueError("you must provide either `umap` or `L`")
    umap = get_umap(args.umap, args.umap_rate_col, args.mut_rate, args.L)

    if args.L is not None:
        L = args.L
    else:
        L = len(umap)

    # Load elements
    elements = [Util.read_bedfile(f) for f in args.bed]
    elements = Util.resolve_elements(elements, L=L, verbose=args.verbose)

    # Optionally load a genetic mask to filter expected pi
    # TODO masking verbosity
    if args.mask:
        mask_regions, chrom_num = Util.read_bedfile(args.mask, get_chrom=True)
        mask = Util.elements_to_mask(mask_regions, L=L)
    # Otherwise, mask sites where the mutation map lacks data
    else:
        chrom_num = None
        mask = umap.mask

    # Load input B-map table
    B_df = pandas.read_csv(args.Bmap)
    in_windows = np.stack([B_df[B_df.columns[1]], B_df[B_df.columns[2]]], 1)
    in_Bs = np.array(B_df["B"])
    # Find focal sites and build the B interpolation function
    in_xs = (in_windows[:, 1] + in_windows[:, 0]) / 2
    Bmap = Predict.get_Bmap(in_xs, in_Bs)

    if args.verbose:
        print(Util._get_time(), "loaded data")

    # Set up focal sites and windows for binning output data
    if args.resolution is None:
        # Use input windows
        res = in_windows[0, 1] - in_windows[0, 0]
        out_windows = in_windows
        foc_Bs = in_Bs
        if "avg_rec" in B_df:
            avg_rec = np.array(B_df["avg_rec"])
    else:
        res = args.resolution
        out_windows = np.stack([np.arange(0, L - res, res),
            np.arange(res, L, res)], axis=1, dtype=np.int64)
        xs = (out_windows[:, 1] + out_windows[:, 0]) / 2
        foc_Bs = Bmap(xs)
        rmap = get_rmap(
            args.rmap,
            args.rmap_pos_col,
            args.rmap_rate_col,
            args.rec_rate,
            L=L)
        avg_rec = Util.compute_average_recombination_rate(out_windows, rmap)

    # Compute expected diversity
    site_B = Bmap(np.arange(L))
    site_pi0 = Inference.expected_pi0(umap, df, elements=elements, dfes=dfes)
    site_pi = Inference.expected_pi(site_pi0, site_B, mask=mask)
    exp_pi, num_sites = Util.compute_window_averages(out_windows, site_pi)

    if args.verbose:
        print(Util._get_time(), "computed expected pi")

    # Find the chromosome number
    if chrom_num is None:
        bed_tbl = pandas.read_csv(args.bed[0], sep="\\s+")
        chrom_num = next(iter(bed_tbl[bed_tbl.columns[0]]))

    # Calculate other quantities of interest
    comb_elements = Util.combine_elements(elements)
    element_mask = Util.elements_to_mask(comb_elements, L=L)
    # Re-mask `site_pi` to retain only selectively constrained sites
    del_sites_mask = np.logical_or(mask, element_mask)
    site_pi.mask = del_sites_mask
    exp_del_pi, del_sites = Util.compute_window_averages(out_windows, site_pi)

    umap.mask = mask
    avg_mut, _ = Util.compute_window_averages(out_windows, umap)
    umap.mask = del_sites_mask
    del_mut, _ = Util.compute_window_averages(out_windows, umap)

    if args.verbose:
        print(Util._get_time(), "computed other maps")

    data = {
        "chrom": [chrom_num] * len(out_windows),
        "chromStart": out_windows[:, 0],
        "chromEnd": out_windows[:, 1],
        "num_sites": num_sites,
        "del_sites": del_sites,
        "avg_mut": avg_mut,
        "del_mut": del_mut,
        "avg_rec": avg_rec,
        "exp_pi": exp_pi,
        "exp_del_pi": exp_del_pi,
        "B": foc_Bs}

    output = pandas.DataFrame(data)
    output.to_csv(args.out, index=False)

    if args.verbose:
        print(Util._get_time(), "saved output")
    return


def predict_B(args):
    """
    Loads data and computes expected B and pi.

    :param args:
    """

    if args.verbose:
        print(Util._get_time(), "loading data")

    # DFE handling
    dfes = get_dfes(args.shapes, args.scales, args.p_neus)

    df, splines = get_lookup_table(
        args.lookup_tbl,
        Ne=args.Ne,
        n_s_cbgs=args.n_s_cbgs,
        cbgs_start=args.cbgs_start,
        verbose=args.verbose)

    # Load mutation rates
    if args.umap is None and args.L is None:
        raise ValueError("you must provide either `umap` or `L`")
    umap = get_umap(args.umap, args.umap_rate_col, args.mut_rate, args.L)

    if args.L is not None:
        L = args.L
    else:
        L = len(umap)

    # Load recombination map
    rmap = get_rmap(
        args.rmap,
        args.rmap_pos_col,
        args.rmap_rate_col,
        args.rec_rate,
        L=L)

    # Load elements and compute their mutation rates
    elements, windows, U_arrs = get_elements(
        args.bed,
        umap,
        L=L,
        window_size=args.window_size,
        verbose=args.verbose)

    # Optionally load a genetic mask to filter expected pi
    # TODO masking verbosity
    if args.mask:
        mask_regions, chrom_num = Util.read_bedfile(args.mask, get_chrom=True)
        mask = Util.elements_to_mask(mask_regions, L=L)
    # Otherwise, mask sites where the mutation map lacks data
    else:
        chrom_num = None
        mask = umap.mask

    if args.verbose:
        print(Util._get_time(), "loaded data")

    # Set up focal site array
    xs = np.arange(args.spacing // 2, L - args.spacing // 2, args.spacing)

    # Predict B-values at focal sites
    interf_Bs = Predict.interference_Bvals(
        xs,
        splines,
        windows=windows,
        U_arrs=U_arrs,
        rmap=rmap,
        dfes=dfes,
        n_corrs=args.n_corrs,
        chunk_size=args.chunk_size,
        n_cores=args.n_cores,
        B_unlinked=args.B_unlinked,
        verbose=args.verbose)

    # Set up focal sites and windows for binning pi
    if args.resolution is None:
        res = args.spacing
    else:
        res = args.resolution

    out_windows = np.stack([np.arange(0, L - res, res),
        np.arange(res, L, res)], axis=1, dtype=np.int64)

    # Compute expected diversity
    B_xs = interf_Bs[-1]
    Bmap = Predict.get_Bmap(xs, B_xs)
    site_B = Bmap(np.arange(L))
    site_pi0 = Inference.expected_pi0(umap, df, elements=elements, dfes=dfes)
    site_pi = Inference.expected_pi(site_pi0, site_B, mask=mask)

    if res == args.spacing:
        foc_B = B_xs
    else:
        midpoints = np.mean(out_windows, axis=1)
        foc_B = Bmap(midpoints)

    exp_pi, num_sites = Util.compute_window_averages(out_windows, site_pi)

    if args.verbose:
        print(Util._get_time(), "computed expected pi")

    # Find the chromosome number
    if chrom_num is None:
        bed_tbl = pandas.read_csv(args.bed[0], sep="\\s+")
        chrom_num = next(iter(bed_tbl[bed_tbl.columns[0]]))

    data = {
        "chrom": [chrom_num] * len(out_windows),
        "chromStart": out_windows[:, 0],
        "chromEnd": out_windows[:, 1],
        "num_sites": num_sites,
        "exp_pi": exp_pi,
        "B": foc_B}

    # Calculate other quantities of interest
    if args.rich:
        comb_elements = Util.combine_elements(elements)
        element_mask = Util.elements_to_mask(comb_elements, L=L)
        # Re-mask `site_pi` to retain only selectively constrained sites
        del_sites_mask = np.logical_or(mask, element_mask)
        site_pi.mask = del_sites_mask
        data["exp_del_pi"], data["del_sites"] = Util.compute_window_averages(
            out_windows, site_pi)

        umap.mask = mask
        data["avg_mut"], _ = Util.compute_window_averages(out_windows, umap)
        umap.mask = del_sites_mask
        data["del_mut"], _ = Util.compute_window_averages(out_windows, umap)
        data["avg_rec"] = Util.compute_average_recombination_rate(
            out_windows, rmap)

        if args.verbose:
            print(Util._get_time(), "computed other maps")

    # Save interference correction rounds
    if args.save_corrs:
        for i, B_xs_i in enumerate(interf_Bs[:-1]):
            if res != args.spacing:
                Bmap = Predict.get_Bmap(xs, B_xs_i)
                foc_B_i = Bmap(midpoints)
            else:
                foc_B_i = B_xs_i
            data[f"B_{i}"] = foc_B_i

    output = pandas.DataFrame(data)
    output.to_csv(args.out, index=False)

    if args.verbose:
        print(Util._get_time(), "saved output")
    return


def fit_Ne(args):
    """
    Fits `Ne` to observed data.
    """

    # TODO Fill in. Currently not tested.

    if args.verbose:
        print(Util._get_time(), "loading data")

    # DFE handling
    dfes = get_dfes(args.shapes, args.scales, args.p_neus)

    df, splines = get_df(
        args.lookup_tbl,
        Ne=args.Ne,
        n_s_cbgs=args.n_s_cbgs,
        cbgs_start=args.cbgs_start)

    # Load mutation rates
    if args.umap is None and args.L is None:
        raise ValueError("you must provide either `umap` or `L`")
    umap = get_umap(args.umap, args.umap_rate_col, args.mut_rate, args.L)

    if args.L is not None:
        L = args.L
    else:
        L = len(umap)

    # Load recombination map
    rmap = get_rmap(
        args.rmap, args.rmap_pos_col, args.rmap_rate_col, args.rec_rate, L)

    # Load elements and compute their mutation rates
    elements = [Util.load_elements(f) for f in args.bed]
    elements = Util.resolve_elements(elements, verbose=args.verbose)
    mean_ss = None # check order and raise warning ...
    ws = args.window_size
    windows = np.stack([np.arange(0, L - ws, ws), np.arange(ws, L, ws)], axis=1)
    U_arrs = [Util.compute_window_mutation_rates(windows, elems, umap)[0]
              for elems in elements]
    windows, U_arrs = Util.filter_empty_windows(windows, U_arrs)

    # Optionally load a genetic mask to filter expected pi
    if args.mask:
        mask_regions, chrom_num = Util.read_bedfile(args.mask)
        mask = Util.elements_to_mask(mask_regions, L=L)
    else:
        chrom_num = None
        mask = None

    if args.verbose:
        print(Util._get_time(), "loaded data")

    opt_args = (
        xs,
        df,
        ndns,
        mask,
        u_map,
        elements,
        windows,
        U_arrs,
        rmap,
        dfes,
        args.n_corrs,
        args.chunk_size,
        args.n_cores,
        args.B_unlinked)

    params = np.array([Ne_init])
    opt = optimize.fmin(
        objective_func,
        params,
        args=opt_args,
        maxiter=maxiter,
        maxfun=maxiter,
        xtol=10,
        ftol=1,
        full_output=True)

    # TODO write log file
    # TODO recover highest-LL maps from cache and save them.

    # TODO
    if chrom_num is None:
        bed_tbl = pandas.read_csv(args.bed[0], sep="\\s+")
        chrom_num = next(iter(bed_tbl[bed_tbl.columns[0]]))

    data = {
        "chrom": [chrom_num] * len(out_windows),
        "chromStart": out_windows[:, 0],
        "chromEnd": out_windows[:, 1],
        "num_sites": num_sites,
        "exp_pi": exp_pi,
        "B": foc_B}

    # Save interference correction rounds
    if args.save_corrs:
        for i, B_xs_i in enumerate(interf_Bs[:-1]):
            if res != args.spacing:
                Bmap = Predict.get_Bmap(xs, B_xs_i)
                foc_B_i = Bmap(midpoints)
            else:
                foc_B_i = B_xs_i
            data[f"B_{i}"] = foc_B_i

    output = pandas.DataFrame(data)
    output.to_csv(args.out, index=False)

    if args.verbose:
        print(Util._get_time(), "saved output")
    return


# Helper functions for loading data with variable file type/structure


def get_lookup_table(
    fname,
    Ne=None,
    n_s_cbgs=20,
    cbgs_start=None,
    verbose=False
):
    """
    Loads a lookup table, optionally scales it, extends it with CBGS, and adds
    a map distance column. Then builds linear splines.

    :returns: Lookup table, dictionary of splines
    """

    # Load lookup table
    df = Util.load_lookup_table(fname)
    df = Util.cap_max_lookup_table_B(df)
    # Scale lookup table to target Ne
    if Ne:
        df = Util.scale_lookup_table(df, Ne)
    # Extend lookup table `r` so they can be safely converted to Morgans later
    df = Util.fill_in_lookup_table(df)

    # Extend lookup table with classic BGS
    min_s = np.min(df["s"])
    if cbgs_start is None:
        ss_extend = -np.logspace(0, np.log10(-min_s), n_s_cbgs + 1)[:-1]
    else:
        assert cbgs_start < 0
        # We shouldn't leave gaps between moments++/CBGS s-grids
        assert min_s <= cbgs_start
        df = df[df["s"] > cbgs_start]
        ss_extend = -np.logspace(0, np.log10(-cbgs_start), n_s_cbgs)
    df = ClassicBGS.extend_lookup_table(df, ss_extend)

    if verbose:
        print(Util._get_time(), "extended lookup table s-grid from "
              f"{ss_extend[-1]:.3} in {len(ss_extend)} steps")

    # Add a map distance column to the lookup table
    df = Util.convert_lookup_table_to_morgans(df)
    # Make splines
    splines = Util.generate_linear_splines(df)[2]
    return df, splines


def get_dfes(shapes, scales, p_neus):
    """
    Build DFE dictionaries from lists of DFE parameters.
    """
    assert len(shapes) == len(scales)

    if p_neus is not None:
        assert len(shapes) == len(p_neus)
    else:
        p_neus = [0] * len(shapes)

    dfes = []
    for shape, scale, p_neu in zip(shapes, scales, p_neus):
        dfe = {"shape": shape, "scale": scale}
        if p_neu > 0:
            dfe["type"] = "gamma_neutral"
            dfe["p_neu"] = p_neu
        else:
            dfe["type"] = "gamma"
        dfes.append(dfe)
    return dfes


def get_umap(fname, rate_col="rate", u=None, L=None):
    """
    Loads a mutation map, or builds a uniform map if `u` and `L` are given.

    Maps may be loaded from site-resolution .npy files, or from tables (.csv,
    .tsv, .bedgraph) that assign uniform rates to genomic windows.

    Importantly, this function returns a masked array with no nan in it.
    """
    if fname is not None:
        if fname.endswith(".npy"):
            umap = np.load(fname)
        else:
            try:
                u_tbl = pandas.read_csv(fname)
            except:
                try:
                    u_tbl = pandas.read_csv(fname, sep="\\s+")
                except:
                    raise ValueError("could not read mutation map file format")
            starts = np.array(u_tbl[u_tbl.columns[1]])
            ends = np.array(u_tbl[u_tbl.columns[2]])
            windows = np.stack([starts, ends], axis=1)
            us = np.array(u_tbl[rate_col])
            umap = Util.build_site_map(windows, us, L=L)
    else:
        assert u is not None and L is not None
        umap = np.full(L, u)
    umap = np.ma.array(umap, mask=np.isnan(umap))
    return umap


def get_rmap(
    fname,
    pos_col="Position(bp)",
    rate_col="Rate(cM/Mb)",
    r=None,
    L=None
):
    """
    Loads a recombination map, or builds a uniform one if `r` and `L` are given.
    """
    if fname is not None:
        rmap = Util.load_recombination_map(
            fname, L, pos_col=pos_col, rate_col=rate_col)
    else:
        assert r is not None and L is not None
        rmap = Util.build_uniform_rmap(r, L)
    return rmap


def get_elements(fnames, umap, window_size=1000, verbose=False, L=None):
    """
    Load elements and element mutation rates.
    """
    if L is None:
        L = len(umap)
    elements = [Util.read_bedfile(f) for f in fnames]
    elements = Util.resolve_elements(elements, verbose=verbose)
    ws = window_size
    windows = np.stack([np.arange(0, L - ws, ws), np.arange(ws, L, ws)], axis=1)
    U_arrs = [Util.compute_window_mutation_rates(windows, elems, umap)[0]
              for elems in elements]
    windows, U_arrs = Util.filter_empty_windows(windows, U_arrs)
    return elements, windows, U_arrs


def main():
    parser = argparse.ArgumentParser(prog="bgshr")
    subparsers = parser.add_subparsers(dest="subcommand")
    # Subcommands
    ComputePiCommand(subparsers)
    PredictBCommand(subparsers)
    FitNeCommand(subparsers)
    args = parser.parse_args()
    if args.subcommand is None:
        print("Please use a command")
        exit(1)
    args.func(args)
    return


if __name__ == "__main__":
    main()
