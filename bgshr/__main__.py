"""
Command-line interface
"""

import argparse
from bgshr import Util, ClassicBGS, Predict, Inference


class Command():

    def __init__(self, subparsers, subcommand):
        self.parser = subparsers.add_parser(subcommand)
        self.parser.set_defaults(func=self)


class CommonCommand(Command):
    # These commands are common to prediction/fitting functions

    def __init__(self, subparsers, subcommand):
        super().__init__(subparsers, subcommand)
        self.parser.add_argument(
            "-d",
            "--dfes",
            type=str,
            default=None,
            help="path to file defining DFE parameters (.yaml)"
        )
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
            "-r",
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
            "--rec_rate",
            type=float, 
            default=None,
            help="uniform recombination rate"
        )
        self.parser.add_argument(
            "-u",
            "--umap",
            type=str, 
            default=None,
            help="mutation map file (.bedgraph, .csv, .txt or .npy)"
        )
        self.parser.add_argument(
            "--umap_pos_col",
            type=str, 
            default=None,
            help="mutation map position column"
        )
        self.parser.add_argument(
            "--umap_rate_col",
            type=str, 
            default=None,
            help="mutation map rate column"
        )
        self.parser.add_argument(
            "--mut_rate",
            type=float, 
            default=None,
            help="uniform mutation rate"
        )
        self.parser.add_argument(
            "-c",
            "--n_corrs",
            type=int, 
            default=0,
            help="number of interference corrections to perform (default 0)"
        )
        self.parser.add_argument(
            "--keep_corrs",
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
        self.parser.add_argument(
            "--resolution",
            type=int,
            default=None,
            help="window size of output table (defaults to spacing)"
        )
        self.parser.add_argument(
            "-L",
            "--L",
            type=int,
            default=None,
            help="chromosome length (defaults to XXXX)"
        )
        self.parser.add_argument(
            "-n",
            "--n_cores",
            type=str,
            default=1000,
            help=""
        )
        self.parser.add_argument(
            "--chunk_size",
            type=str,
            default=1000,
            help=""
        )
        self.parser.add_argument(
            "--B_unlinked",
            type=float,
            default=None,
            help=""
        )
        self.parser.add_argument(
            "--min_alpha_cbgs",
            type=float,
            default=None,
            help="maximimum |2Ne*s| value in the lookup table below CBGS domain"
        )
        self.parser.add_argument(
            "--n_s_extend",
            type=float,
            default=None,
            help="number of CBGS selection coefficients to model"
        )
        self.parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="path for output file (.csv)"
        )
        self.parser.add_argument(
            "-o",
            "--out",
            type=str,
            required=True,
            help="path for output file (.csv)"
        )


class PredictCommand(CommonCommand):

    def __init__(self, subparsers):
        super().__init__(subparsers, "predict")
        self.parser.add_argument(
            "--Ne",
            type=float,
            default=None,
            help="effective size parameter"
        )
        self.parser.add_argument(
            "--mask",
            type=str,
            default=None,
            help=""
        )


    def __call__(self, args):
        predict(args)


class FitNeCommand(CommonCommand):

    def __init__(self, subparsers):
        super().__init__(subparsers, "fit_Ne")
        self.parser.add_argument(
            "--ndns",
            type=str,
            required=True,
            help=""
        )
        self.parser.add_argument(
            "--mask",
            type=str,
            required=True,
            help=""
        )
        self.parser.add_argument(
            "--Ne0",
            type=float,
            required=True,
            help=""
        )
        self.parser.add_argument(
            "--maxiter",
            type=int,
            required=True,
            help="Maximum number of optimization iterations"
        )
        self.parser.add_argument(
            "--log_out",
            type=str,
            required=True,
            help="Path for output log file (.txt)"
        )


    def __call__(self, args):
        fit_Ne(args)


def predict(args):
    """
    Load data and computed expected B and pi.

    :param args:
    """

    print(bgshr.Util._get_time(), "Loading data")

    # DFE handling
    dfes = []

    # Load and set up lookup table; build splines
    df = Util.load_lookup_table(lookup_table)
    ###
    splines = Util.build_cubic_splines(df)[2]
    max_dists = None

    # Load mutation rates
    L = None
    umap = get_umap()

    # Load recombination map
    rmap = get_rmap()

    # Load elements and compute their mutation rates
    elements = [Util.load_elements(f) for f in args.bed]
    elements = Util.resolve_elements(elements, verbose=args.verbose)
    mean_ss = None # check order and raise warning ...
    ws = args.window_size
    windows = np.stack([np.arange(0, L - ws, ws), np.arange(ws, L, ws)], axis=1)
    U_arrs = [bgshr.Util.compute_window_mutation_rate(
        elems, windows, umap)[1] for elems in elements]
    windows, U_arrs = bgshr.Util.filter_empty_windows(windows, U_arrs)

    # For predicting pi ...
    if args.mask:
        mask_regions, chrom_num = bgshr.Util.read_bedfile(args.mask)
        mask = bgshr.Util.elements_to_mask(mask_regions, L=L)
    else:
        mask = None

    # Mask site mutation rates
    print(bgshr.Util._get_time(), "Loaded data")

    # Set up focal site array
    xs = np.arange(args.spacing // 2, L, args.spacing)

    # Set up focal sites and windows for binning pi
    if args.resolution is None:
        res = args.spacing
    else:
        res = args.resolution

    out_windows = np.stack([np.arange(0, L - res, res),
        np.arange(res, L, res)], axis=1, dtype=np.int64)

    interf_Bs = Predict.interference_Bvals(
        xs,
        splines,
        windows=windows,
        U_arrs=U_arrs,
        rmap=rmap,
        max_dists=max_dists,
        dfes=dfes,
        chunk_size=args.chunk_size,
        n_cores=args.n_cores,
        B_unlinked=args.B_unlinked,
        verbose=args.verbose
    )

    # Compute expected diversity
    B_xs = interf_Bs[-1]
    Bmap = Predict.get_Bmap(xs, B_xs)
    site_B = Bmap(np.arange(L))
    site_pi0 = Predict.expected_pi0(umap, df, elements=elements, dfes=dfes)
    site_pi = Predict.expected_pi(pi0, site_B, mask=mask)

    if res == args.spacing:
        foc_B = B_xs
    else:
        midpoints = np.mean(out_windows, axis=1)
        foc_B = Bmap(midpoints)

    window_pi, num_sites = Util.compute_window_averages(out_windows, site_pi)

    # Save output in a .csv file
    data = {
        "chrom": [chrom] * len(windows),
        "chromStart": out_windows[:, 0],
        "chromEnd": out_windows[:, 1],
        "num_sites": num_sites
    }

    # Save interference correction rounds
    if args.save_corrs:
        for i, B_xs_i in enumerate(interf_Bs[:-1]):
            if res != args.spacing:
                Bmap = bgshr.Predict.get_Bmap(xs, B_xs_i)
                foc_B_i = Bmap(midpoints)
            else:
                foc_B_i = B_xs_i
            data[f"B_{i}"] = foc_B_i

    data["B"] = foc_B
    data["exp_pi"] = window_pi
    output = pandas.DataFrame(data)
    output.to_csv(args.out, index=False)
    return


def fit_Ne():


    # get DFEs

    df = pandas.read_csv(lookup_table)
    nd, ns = np.load(ndns_table).T
    L = len(nd)
    regions, chrom_num = bgshr.Util.read_bedfile(mask_file)
    mask = bgshr.Util.regions_to_mask(regions, L=L)
    # Mask ndns array
    ndns = (np.ma.array(nd, mask=mask), np.ma.array(ns, mask=mask))

    # Load site mutation data
    if mut_map.endswith(".csv.gz") or mut_map.endswith(".csv"):
        avg_u = next(iter(pandas.read_csv(mut_map)["rate"]))
        site_u = np.full(L, avg_u)
    else:
        site_u = np.load(mut_map)


    # Load elements and compute deleterious mutation rates
    windows = np.stack([np.arange(0, L - window_size, window_size),
        np.arange(window_size, L, window_size)], axis=1)
    elements = [bgshr.Util.read_bedfile(f)[0] for f in annot_bedfiles]
    U_arrs = [bgshr.Util.compute_windowed_mutation_rate(
        elems, windows, site_u)[1] for elems in elements]
    windows, U_arrs = bgshr.Util.filter_empty_windows(windows, U_arrs)

    # Mask site mutation rates
    site_u = np.ma.array(site_u, mask=mask)
    site_u.mask[np.isnan(site_u)] = True

    rmap = bgshr.Util.load_recombination_map(rec_map, L)

    # Set up focal sites and windows for binning pi
    output_windows = np.stack([np.arange(0, L - spacing, spacing),
        np.arange(spacing, L, spacing)], axis=1, dtype=np.int64)
    xs = np.mean(output_windows, axis=1).astype(np.int64)

    opt_args = (

    )
    params = np.array([Ne_init])
    opt = optimize.fmin(
        objective_func,
        params,
        args=opt_args,
        maxiter=maxiter,
        maxfun=maxiter,
        xtol=10,
        ftol=10,
        full_output=True
    )

    # Write log file
    xopt, fopt, iters, calls, flag = opt
    log_filename = f"{out_prefix}_log.txt"
    global times, Nes, lls
    with open(log_filename, "w") as f:
        f.write(f"xopt\t{xopt[0]}\n")
        f.write(f"fopt\t{fopt}\n")
        f.write(f"n_iters\t{iters}\n")
        f.write(f"n_calls\t{calls}\n")
        f.write(f"flag\t{flag}\n")
        f.write(f"convergence:\n")
        f.write("time\tNe\tll\n")
        for (t, Ne, ll) in zip(times, Nes, lls):
            f.write(f"{t}\t{Ne}\t{ll}\n")

    global last_interf_Bs, last_exp_pi
    out_fname = f"{out_prefix}_map.csv"
    save_maps(
        out_fname, chrom_num, output_windows, last_interf_Bs, ndns, last_exp_pi)

    return


# Helper functions for loading data with variable file type/structure


def load_dfes():
   if unlinked_B is None:
        unlinked_B = 1.0

    # Build DFEs
    assert len(annot_bedfiles) == len(shapes) == len(scales)
    if p_neus is not None:
        assert len(annot_bedfiles) == len(p_neus)
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
            dfe["p_neu"] = 0
        dfes.append(dfe)
    return


def load_rmap():

    return


def load_umap():
    """
    It's important that this function returns a masked array with no nan in it.
    """
    # Load site mutation data
    if mut_map.endswith(".csv.gz") or mut_map.endswith(".csv"):
        avg_u = next(iter(pandas.read_csv(mut_map)["rate"]))
        site_u = np.full(L, avg_u)
    else:
        site_u = np.load(mut_map)


    site_u = np.ma.array(site_u, mask=mask)
    site_u.mask[np.isnan(site_u)] = True

    return


def main():
    parser = argparse.ArgumentParser(prog="bgshr")
    subparsers = parser.add_subparsers(dest="subcommand")

    # Subcommands
    PredictCommand(subparsers)
    FitNeCommand(subparsers)

    args = parser.parse_args()
    if args.subcommand is None:
        print("Please use a command")
        exit(1)
    args.func(args)
    return


if __name__ == "__main__":
    main()
