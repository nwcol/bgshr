
from datetime import datetime
import gzip
import numpy as np
import pandas
from scipy import interpolate
from scipy import integrate
from scipy import stats
import warnings


# Lookup table handling.


def load_lookup_table(df_name, sep=","):
    df = pandas.read_csv(df_name, sep=sep)
    return df


def subset_lookup_table(df, generation=0, Ns=None, Ts=None, uL=None):
    """
    Subset a lookup table to make sure that it represents a single demographic
    history (`Ts`, `Ns`), `uL`, and sampling `generation`.
    """
    if type(Ns) == list:
        raise ValueError("Need to implement")
    if type(Ts) == list:
        raise ValueError("Need to implement")
    
    df_sub = df[df["Generation"] == generation]
    if Ns is not None:
        df_sub = df_sub[df_sub["Ns"] == Ns]
    if Ts is not None:
        df_sub = df_sub[df_sub["Ts"] == Ts]
    if uL is not None:
        df_sub = df_sub[df_sub["uL"] == uL]

    if len(np.unique(df_sub["uL"])) != 1:
        raise ValueError("Require a single uL value - please specify")
    if len(np.unique(df_sub["Ns"])) != 1:
        raise ValueError("Require a single Ns history")
    if len(np.unique(df_sub["Ts"])) != 1:
        raise ValueError("Require a single Ts history")
    return df_sub


def generate_cubic_splines(df_sub, use_M=False):
    """
    df_sub is the dataframe subsetted to a single demography, uR and t

    Cubic spline functions are created over r, for each combination of
    uL and s, returning the fractional reduction based on piR and pi0.

    This function returns the (sorted) arrays of uL and s and the
    dictionary of cubic spline functions with keys (uL, s).
    """

    # Check that only a single entry exists for each item
    assert len(np.unique(np.array(df_sub["Ts"]))) == 1
    assert len(np.unique(np.array(df_sub["Ns"]))) == 1
    assert len(np.unique(df_sub["uR"])) == 1
    assert len(np.unique(df_sub["uL"])) == 1
    assert len(np.unique(df_sub["Generation"])) == 1

    # Get arrays of selection and mutation values
    s_vals = np.array(sorted(list(set(df_sub["s"]))))
    u_vals = np.array(sorted(list(set(df_sub["uL"]))))

    # Store cubic splines of fractional reduction for each pair of u and s, over r
    splines = {}
    for u in u_vals:
        for s in s_vals:
            key = (u, s)
            # Subset to given s and u values
            df_s_u = df_sub[(df_sub["s"] == s) & (df_sub["uL"] == u)]
            if use_M:
                rs = np.array(df_s_u["M"])
            else:
                rs = np.array(df_s_u["r"])
            Bs = np.array(df_s_u["B"])
            inds = np.argsort(rs)
            rs = rs[inds]
            Bs = Bs[inds]
            splines[key] = interpolate.CubicSpline(rs, Bs, bc_type="natural")
    return u_vals, s_vals, splines


def generate_linear_splines(df_sub, use_M=False):
    """
    Generate linear splines to interpolate B-values across r.
    """
    # Check that only a single entry exists for each item
    assert len(np.unique(np.array(df_sub["Ts"]))) == 1
    assert len(np.unique(np.array(df_sub["Ns"]))) == 1
    assert len(np.unique(df_sub["uR"])) == 1
    assert len(np.unique(df_sub["uL"])) == 1
    assert len(np.unique(df_sub["Generation"])) == 1

    # Get arrays of selection and mutation values
    s_vals = np.array(sorted(list(set(df_sub["s"]))))
    u_vals = np.array(sorted(list(set(df_sub["uL"]))))

    # Store cubic splines of fractional reduction for each pair of u and s, over r
    splines = {}
    for u in u_vals:
        for s in s_vals:
            key = (u, s)
            # Subset to given s and u values
            df_s_u = df_sub[(df_sub["s"] == s) & (df_sub["uL"] == u)]
            if use_M:
                rs = np.array(df_s_u["M"])
            else:
                rs = np.array(df_s_u["r"])
            Bs = np.array(df_s_u["B"])
            inds = np.argsort(rs)
            rs = rs[inds]
            Bs = Bs[inds]
            splines[key] = interpolate.make_interp_spline(rs, Bs, k=1)
    return u_vals, s_vals, splines


def scale_lookup_table(df, N_target):
    """
    Takes an equilibrium lookup table created with moments++ and scales physical
    parameters r, s, u by N_ref / N_target, so that the table can be used to
    make predictions with N_e = N_target.

    `N_ref` is automatically determined with the `Ns` column of the input table.
    If the table is not an equilibrium table, N_ref is taken to be the ancestral
    population size.

    If scaling places any `r` values > 0.5, these rows are dropped from the
    table.

    :param df: Lookup table, as a pandas DataFrame
    :param N_target: Target N_e

    :returns: Scaled lookup table
    """
    Ns_strs = set(df["Ns"])
    Ts_strs = set(df["Ts"])
    assert len(Ns_strs) == 1
    assert len(Ts_strs) == 1
    Ns = next(iter(Ns_strs))
    Ts = next(iter(Ts_strs))

    # Scale Ns, Ts by N_target / N_ref
    if str(Ts).isnumeric():
        N_ref = float(Ns)
        new_Ns = int(N_target)
        new_Ts = Ts
    else:
        Ns = Ns.split(";")
        Ts = Ts.split(";")
        N_ref = float(Ns[-1])
        new_Ns = ";".join([str(int(N_target / N_ref * float(N))) for N in Ns])
        new_Ts = ";".join([str(int(N_target / N_ref * float(T))) for T in Ts])

    # Scale parameters by N_ref / N_target
    fac = N_ref / N_target
    df_scaled = df.copy()
    df_scaled["r"] *= fac
    df_scaled["s"] *= fac
    df_scaled["uL"] *= fac
    df_scaled["uR"] *= fac
    df_scaled["Generation"] *= (N_target / N_ref)
    df_scaled["Ns"] = new_Ns
    df_scaled["Ts"] = new_Ts

    # Drop any rows where scaling produced r > 0.5
    if np.any(df_scaled["r"] > 0.5):
        df_scaled = df_scaled[df_scaled["r"] <= 0.5]
    return df_scaled


def fill_in_lookup_table(df, n_steps=16, max_M=10):
    """
    Makes filler lookup table entries for large recombination distances. Should
    be applied only to pure moments++ tables, where using B~1 at large `r` is
    not too bad an approximation.

    :param df: Lookup table, as a pandas DataFrame
    :param max_r: Maximum recombination distance (default 0.5)
    :param n_steps: Number of r steps to add
    """
    ss = np.sort(np.unique(df["s"]))
    rs = np.sort(np.unique(df["r"]))

    if np.max(rs) == 0.5:
        Ms = inverse_haldane_map_function(rs[:-1])
    else:
        Ms = inverse_haldane_map_function(rs)

    Ms_extend = np.logspace(np.log10(Ms[-1]), np.log10(max_M), n_steps + 1)[1:]
    rs_extend = haldane_map_function(Ms_extend)

    cols = [
        "r",
        "s",
        "uL",
        "Order",
        "Generation",
        "Hr",
        "pi0",
        "B",
        "uR",
        "Hl",
        "piN_pi0",
        "piN_piS",
        "Ns",
        "Ts",
    ]
    data = {
        "uL": np.unique(df["uL"])[0],
        "Order": 0,
        "Generation": 0,
        "pi0": np.unique(df["pi0"])[0],
        "uR": np.unique(df["uR"])[0],
        "Ns": next(iter(set(df["Ns"]))),
        "Ts": next(iter(set(df["Ts"]))),
    }

    # We assume that B~1 across rs_extend
    data["B"] = 1
    data["Hr"] = data["pi0"]

    new_data = []
    for s in ss:
        data["s"] = s
        data["Hl"] = np.array(df[df["s"] == s]["Hl"])[0]
        data["piN_pi0"] = data["Hl"] / data["pi0"]
        data["piN_piS"] = data["Hl"] / data["Hr"]
        for r in rs_extend:
            data["r"] = r
            new_row = [data[k] for k in cols]
            new_data.append(new_row)
    df_new = pandas.DataFrame(new_data, columns=cols)
    df_extended = pandas.concat([df, df_new], ignore_index=True)
    return df_extended


def convert_lookup_table_to_morgans(df):
    """
    Adds a column with recombination distances in Morgans to a lookup table.

    Drops rows with r = 0.5.

    :param df: Lookup table
    """
    df_copy = pandas.DataFrame({k: df[k] for k in list(df.columns)})
    df_copy = df_copy[df_copy["r"] < 0.5]
    df_copy["M"] = inverse_haldane_map_function(np.array(df_copy["r"]))
    return df_copy


def cap_max_lookup_table_B(df):
    """
    Set maximum lookup table B entry to `1`. Without this, some entries may be
    > 1 by small amounts due to numerical error.
    """
    df_copy = pandas.DataFrame({k: df[k] for k in list(df.columns)})
    Bs = np.array(df_copy["B"])
    Bs[Bs > 1] = 1
    df_copy["B"] = Bs
    return df_copy


# Recombination maps.


def build_uniform_rmap(r, L):
    """
    Builds a cumulative recombination map for given per-base recombination rate r.
    """
    pos = [0, L]
    cum = [0, r * L]
    return interpolate.interp1d(pos, cum)


def build_pw_constant_rmap(pos, rates):
    pass


def adjust_uniform_rmap(r, L, bmap, steps=1000):
    # this may need to be revisited
    pos = [0]
    cum = [0]
    for x in np.linspace(0, L, steps + 1)[1:]:
        pos.append(x)
        cum.append(bmap.integrate(0, x) * r)
    return interpolate.interp1d(pos, cum)


def build_recombination_map(pos, rates):
    """
    From arrays of positions and rates, build a linearly interpolated recombination
    map. The length of pos must be one greater than rates, and must be monotonically
    increasing.
    """
    cum = [0]
    for left, right, rate in zip(pos[:-1], pos[1:], rates):
        if rate < 0:
            raise ValueError("recombination rate is negative")
        bp = right - left
        if bp <= 0:
            raise ValueError("positions are not monotonically increasing")
        cum += [cum[-1] + bp * rate]
    return interpolate.interp1d(pos, cum)


def adjust_recombination_map(rmap, bmap):
    # from interp1d get the x and ys, adjust ys, and rebuild
    # probably not that nice of a way to do it...
    pos = np.sort(np.unique(np.concatenate((bmap.x, rmap.x))))
    map_pos = rmap.x
    map_vals = np.diff(rmap.y) / np.diff(rmap.x)
    bs = np.array([bmap.integrate(a, b) for a, b in zip(pos[:-1], pos[1:])]) / np.diff(
        pos
    )
    rates = np.zeros(len(pos) - 1)
    i = -1
    for j, x in enumerate(pos):
        if x in map_pos:
            if x == map_pos[-1]:
                break
            i += 1
        rates[j] = map_vals[i]
    rates *= bs
    rmap = build_recombination_map(pos, rates)
    return rmap


def load_recombination_map(
    fname,
    L=None,
    scaling=1,
    pos_col="Position(bp)",
    rate_col="Rate(cM/Mb)"
):
    """
    Get positions and rates to build recombination map.

    If L is not None, we extend the map to L if it is greater than the
    last point in the input file, or we truncate the map at L if it is
    less than the last point in the input file.

    If L is not given, the interpolated map does not extend beyond
    the final data point.
    """
    map_df = pandas.read_csv(fname, sep="\\s+")
    pos = np.concatenate(([0], map_df[pos_col]))
    rates = np.concatenate(([0], map_df[rate_col])) / 100 / 1e6
    if L is not None:
        if L > pos[-1]:
            pos = np.insert(pos, len(pos), L)
        elif L < pos[-1]:
            cutoff = np.where(L <= pos)[0][0]
            pos = pos[:cutoff]
            pos = np.append(pos, L)
            rates = rates[:cutoff]
        else:
            # L == pos[-1]
            rates = rates[:-1]
    else:
        rates = rates[:-1]
    assert len(rates) == len(pos) - 1
    rmap = build_recombination_map(pos, rates * scaling)
    return rmap


def load_bedgraph_recombination_map(
    fname, 
    sep=",", 
    L=None, 
    scaling=1,
    start_col="start",
    end_col="end",
    rate_col="rate"
):
    """
    Load a recombination map stored in bedgraph format. If L is given, the map
    is truncated to end at position L- if L exceeds the length of the map, an 
    error is raised.
    """
    map_df = pandas.read_csv(fname, sep=sep)
    starts = np.array(map_df[start_col])
    ends = np.array(map_df[end_col])
    rates = np.array(map_df[rate_col])
    edges = np.concatenate(([starts[0]], ends))
    if L is not None:
        if L > ends[-1]:
            raise ValueError("`L` cannot exceed the highest physical position")
        else:
            cutoff = np.where(L <= edges)[0][0]
            edges = np.append(edges[:cutoff], L) 
            rates = rates[:cutoff]
    assert len(rates) == len(edges) - 1
    ratemap = build_recombination_map(edges, rates * scaling)
    return ratemap


def compute_average_recombination_rate(windows, rmap):
    """
    Calculates the average per-bp recombination rate of `rmap` in an array of
    windows.

    :param windows: Array of genomic intervals
    :param rmap: Recombination map interpolation function
    """
    r_dists = rmap(windows[:, 1]) - rmap(windows[:, 0])
    bp_dists = windows[:, 1] - windows[:, 0]
    return r_dists / bp_dists


def haldane_map_function(ds):
    """
    Returns recombination fraction following Haldane's map function.
    """
    return 0.5 * (1 - np.exp(-2 * ds))


def inverse_haldane_map_function(rs):
    """
    Convert recombination fraction `rs` to Morgans, using the inverse of 
    Haldane's map functon.
    """
    return np.abs(-0.5 * np.log(1 - 2 * rs))


# Mutation maps, rates


def compute_window_averages(windows, site_map):
    """
    Computes windowed averages of some site-resolution quantity `site_map`. It
    is expected that `site_map` is a masked array (np.ma.array).

    Also returns the number of non-masked entries in each interval.

    :param windows: Array of BED-style genomic intervals.
    :param site_map: Site-resolution map to average within `windows`.

    :returns: Array of window averages.
    """
    num_sites = np.zeros(len(windows), dtype=np.int64)
    avgs = np.zeros(len(windows), dtype=np.float64)
    for ii, (start, end) in enumerate(windows):
        if np.all(site_map[start:end].mask):
            continue
        else:
            num_sites[ii] = np.sum(np.logical_not(site_map[start:end].mask))
            avgs[ii] = np.mean(site_map[start:end])
    return avgs, num_sites


def build_site_map(intervals, values, L=None):
    """
    Constructs a site-resolution map from a set of window starts/ends
    `intervals` and an array of window `values`.

    :param intervals: Array of BED-style genomic intervals.
    :param values: Array of values applying to each interval
    :param L: Optional chromosome length. If None (default), the highest
        position in `intervals` is used.
    :returns: Array holding a site-resolution map of length `L`.
    """
    assert len(intervals) == len(values)
    if L is None:
        L = int(intervals[-1, 1])
    site_map = np.zeros(L, dtype=np.float64)
    for ii, (start, end) in enumerate(intervals):
        if start >= L:
            break
        if end > L:
            end = L
        site_map[start:end] = values[ii]
    return site_map


def compute_window_mutation_rates(windows, elements, u, fill_val="mean"):
    """
    """
    if np.isscalar(u):
        L = windows[-1, 1]
        u_arr = u * np.ma.array(np.ones(L), mask=False)
    else:
        # Windows shouldn't extend beyond the end of the chromosome
        L = len(u)
        assert windows[-1, 1] <= L
        u_arr = 1 * u
    assert not np.any(np.isnan(u_arr))

    if fill_val == "mean":
        fill_val = np.mean(u_arr)
    else:
        assert isinstance(fill_val, float)

    sites = np.zeros(len(windows), np.int64)
    U_arr = np.zeros(len(windows), np.float64)
    index, window_elems = decompose_elements(windows, elements)
    for idx, (start, end) in zip(index, window_elems):
        elem_sites = end - start
        sites[idx] += elem_sites
        u_elem = u_arr[start:end]
        if np.all(u_elem.mask):
            U_arr[idx] += fill_val * elem_sites
        else:
            n_masked = np.sum(u_elem.mask)
            U_arr[idx] += np.sum(u_elem) + fill_val * n_masked
    return U_arr, sites


def decompose_elements(windows, elements):
    """
    """
    elem_mask = elements_to_mask(elements)
    index = []
    elems_out = []
    for i, (start, end) in enumerate(windows):
        segment = elem_mask[start:end]
        elems = mask_to_elements(segment) + start
        for e in elems:
            index.append(i)
            elems_out.append(e)
    elems_out = np.array(elems_out)
    return index, elems_out


def filter_empty_windows(windows, U_arrs):
    """
    Filters out windows with deleterious mutation rate = 0
    """
    tot_U = np.sum(U_arrs, axis=0)
    keep = tot_U > 0
    filtered_windows = windows[keep]
    filtered_U_arrs = [Us[keep] for Us in U_arrs]
    return filtered_windows, filtered_U_arrs


def compute_element_mutation_rates(elements, u_arr, fill_val="mean"):
    """
    Return an array of average mutation rates for each interval in `elements`.

    :param elements:
    :param u_arr:
    :param fill_val:
    """

    # Elements shouldn't extend beyond the end of the chromosome
    assert elements[-1, 1] <= len(u_arr)
    assert not np.any(np.isnan(u_arr))

    if fill_val == "mean":
        fill_val = np.mean(u_arr)
    else:
        assert isinstance(fill_val, float)

    element_u = np.zeros(len(elements), np.float64)

    for i, (start, end) in enumerate(elements):
        u_elem = u_arr[start:end]
        if np.all(u_elem.mask):
            element_u[i] = fill_val
        else:
            element_u[i] = np.mean(u_elem)
    return element_u


def split_mutation_windows(windows, U_arrs):
    """
    Makes inputs to `Bvals_fast` comport to `Bvals_dfes`.
    """
    all_elements = []
    avg_u_arrs = []
    for U_arr in U_arrs:
        keep = np.where(U_arr > 0)[0]
        elements = windows[keep]
        all_elements.append(elements)
        n_sites = elements[:, 1] - elements[:, 0]
        avg_u = U_arr[keep] / n_sites
        avg_u_arrs.append(avg_u)
    return all_elements, avg_u_arrs


def load_mutation_arrays(
    mut_fname,
    annot_fnames,
    keep_zeros=False,
    masked=False
):
    """
    Load arrays of deleterious mutation rates.

    :param mut_fname: File with average mutation rates
    :param annot_fnames: Files with deleterious site counts and `scale`, the
        ratio of average deleterious u to average u, for one or more classes
        of constrained sites.
    :param keep_zeros: If False (default), remove windows without deleterious
        sites. Windows are shared across constraint classes.
    :param masked: If True (default False), load data from the `masked` columns
        of `annot_fnames`.
    """
    mut_tbl = pandas.read_csv(mut_fname)
    windows = np.array([mut_tbl["chromStart"], mut_tbl["chromEnd"]]).T
    if masked:
        avg_mut = np.array(mut_tbl["avg_mut_masked"])
    else:
        avg_mut = np.array(mut_tbl["avg_mut"])
    avg_mut[np.isnan(avg_mut)] = 0
    del_sites_arrs = []
    u_arrs = []
    U_arrs = []
    for fname in annot_fnames:
        annot_tbl = pandas.read_csv(fname)
        annot_windows = np.array(
            [annot_tbl["chromStart"], annot_tbl["chromEnd"]]).T
        if not np.all(annot_windows == windows):
            raise ValueError(
                "Annotation/mutation tables have mismatching windows")
        if masked:
            del_sites = np.array(annot_tbl["del_sites_masked"])
            scale = np.array(annot_tbl["scale_masked"])
        else:
            del_sites = np.array(annot_tbl["del_sites"])
            scale = np.array(annot_tbl["scale"])
        scale[np.isnan(scale)] = 0
        u_arr = scale * avg_mut
        U_arr = del_sites * u_arr
        del_sites_arrs.append(del_sites)
        u_arrs.append(u_arr)
        U_arrs.append(U_arr)
    # Remove windows with 0 total deleterious sites
    if not keep_zeros:
        keep = np.where(np.sum(del_sites_arrs, axis=0) > 0)[0]
        windows = windows[keep]
        del_sites_arrs = [arr[keep] for arr in del_sites_arrs]
        u_arrs = [arr[keep] for arr in u_arrs]
        U_arrs = [arr[keep] for arr in U_arrs]
    return windows, del_sites_arrs, u_arrs, U_arrs


# Elements/constraint windows


def read_bedfile(fname, filter_col=None, sep=None, L=None, get_chrom=False):
    """
    Load a bed file, returning an array of intervals and the chromosome number.

    :param dict filter_col: Optional 1-dictionary for filtering intervals. If
        given, return only intervals where the column named by the key of
        `filter_col` has an entry matching `filter_col[key]`.
    :param str sep: Optional separator string, defaults to "\t" if None.
    """

    # Check whether there is a header
    if fname.endswith(".gz"):
        with gzip.open(fname, "rb") as fin:
            first_line = fin.readline().decode()
    else:
        with open(fname, "r") as fin:
            first_line = fin.readline()

    if "," in first_line:
        split_line = first_line.split(",")
        if sep is None:
            sep = ","
    else:
        split_line = first_line.split()
        if sep is None:
            sep = r"\s+"

    if split_line[1].isnumeric():
        data = pandas.read_csv(fname, sep=sep, header=None)
    else:
        data = pandas.read_csv(fname, sep=sep)

    if filter_col is not None:
        assert len(filter_col) == 1
        col_name = next(iter(filter_col))
        data = data[data[col_name] == filter_col[col_name]]

    cols = data.columns
    starts = np.array(data[cols[1]]).astype(np.int64)
    ends = np.array(data[cols[2]]).astype(np.int64)

    if L:
        # Delete elements with start >= L
        keep = np.where(starts < L)[0]
        starts = ends[keep]
        ends = ends[keep]
        # Truncate elements with end > L
        to_trunc = np.where(ends > L)[0]
        ends[to_trunc] = L

    intervals = np.stack((starts, ends), axis=1)

    if get_chrom:
        uniq_chroms = list(set(data[cols[0]]))
        if len(uniq_chroms) > 1:
            warnings.warn(f"BED file has more than one unique chrom")
        chrom = str(uniq_chroms[0])
        ret = (intervals, chrom)
    else:
        ret = intervals
    return ret


def load_elements(bed_file, L=None):
    """
    From a bed file, load elements. If L is not None, we exclude regions
    greater than L, and any region that overlaps with L is truncated at L.
    """
    elem_left = []
    elem_right = []
    data = pandas.read_csv(bed_file, sep="\t", header=None)
    elem_left = np.array(data[1])
    elem_right = np.array(data[2])
    if L is not None:
        to_del = np.where(elem_left >= L)[0]
        elem_left = np.delete(elem_left, to_del)
        elem_right = np.delete(elem_right, to_del)
        to_trunc = np.where(elem_right > L)[0]
        elem_right[to_trunc] = L
    elements = np.zeros((len(elem_left), 2), dtype=int)
    elements[:, 0] = elem_left
    elements[:, 1] = elem_right
    return elements


def get_elements(df, L=None):
    """
    From a bed file, load elements. If L is not None, we exlude regions
    greater than L, and any region that overlaps with L is truncated at L.
    """
    elem_left = []
    elem_right = []
    df_sub = df[df["selected"] == 1] # select only exons

    elem_left = np.array(df_sub["start"])
    elem_right = np.array(df_sub["end"])
    if L is not None:
        to_del = np.where(elem_left >= L)[0]
        elem_left = np.delete(elem_left, to_del)
        elem_right = np.delete(elem_right, to_del)
        to_trunc = np.where(elem_right > L)[0]
        elem_right[to_trunc] = L
    elements = np.zeros((len(elem_left), 2), dtype=int)
    elements[:, 0] = elem_left
    elements[:, 1] = elem_right
    return elements


def collapse_elements(elements):
    """
    Collapses any overlapping `elements`.
    """
    return mask_to_elements(elements_to_mask(elements))


def break_up_elements(elements, max_size=500):
    """
    """
    elements_br = []
    for l, r in elements:
        if r - l > max_size:
            num_breaks = (r - l) // max_size
            z = np.floor(np.linspace(l, r, 2 + num_breaks)).astype(int)
            for x, y in zip(z[:-1], z[1:]):
                elements_br.append([x, y])
        else:
            elements_br.append([l, r])
    return np.array(elements_br)


def resolve_elements(elements, L=None, verbose=False):
    """
    Removes redundant sites that appear in more than one array of `elements`.

    The order of `elements` determines priority; the zeroth array is unchanged,
    while the last element loses any sites which appeared in prior elements.
    """
    if not L:
        L = max([elems[-1, 1] for elems in elements])
    covered_sites = np.zeros(L)
    resolved_elements = []
    for i, elems in enumerate(elements):
        mask = elements_to_mask(elems, L=L)
        redundant_sites = np.where((mask == 0) & (covered_sites == 1))[0]
        # Remove redundant sites
        mask[redundant_sites] = True
        resolved_elems = mask_to_elements(mask)
        for l, r in resolved_elems:
            covered_sites[l:r] = True
        resolved_elements.append(resolved_elems)
        if verbose and i > 0:
            print(_get_time(), f"removed {len(redundant_sites)} "
                  f"redundant sites from element class {i}")
    return resolved_elements


def elements_to_mask(elements, L=None):
    """
    Returns an array which equals False within `elements` and True elsewhere.
    """
    if L is None:
        L = elements[-1, 1]
    mask = np.ones(L, dtype=bool)
    for (start, end) in elements:
        if start > L:
            break
        if end > L:
            end = L
        mask[start:end] = 0
    return mask


def mask_to_elements(mask):
    """
    Returns an array of starts and ends corresponding to regions where `mask`
    is False.
    """
    jumps = np.diff(np.concatenate(([1], mask, [1])))
    starts = np.where(jumps == -1)[0]
    ends = np.where(jumps == 1)[0]
    elements = np.stack([starts, ends], axis=1)
    return elements


def intersect_elements(elements, L=None):
    """
    Takes several arrays of `elements` and returns an array of regions where
    all of them have coverage.

    :param elements: List of elements arrays
    :param L: Optional maximum position for output. If None (default), the
        highest end position in `elements` is used.
    """
    if L is None:
        L = max([elems[-1, 1] for elems in elements])
    coverage = np.zeros(L, dtype=np.uint8)
    for elems in elements:
        for (start, end) in elems:
            if start >= L:
                continue
            if end > L:
                end = L
            coverage[start:end] += 1
    mask = coverage < len(elements)
    elements_out = mask_to_elements(mask)
    return elements_out


def combine_elements(elements, L=None):
    """
    Returns an array of intervals representing all sites present in any
    interval array in `elements`.

    :param elements: List of elements arrays
    :param L: Optional maximum position for output. If None (default), the
        highest end position in `elements` is used.
    """
    if L is None:
        L = max([elems[-1, 1] for elems in elements])
    mask = np.ones(L, dtype=bool)
    for elems in elements:
        for (start, end) in elems:
            if start >= L:
                continue
            if end > L:
                end = L
            mask[start:end] = False
    elements_out = mask_to_elements(mask)
    return elements_out


def subtract_elements(elements0, elements1, L=None):
    """
    Returns an array of intervals representing sites that belong to `elements0`
    but not `elements1`.

    :param elements: List of elements arrays
    :param L: Optional maximum position for output. If None (default), the
        highest end position in `elements` is used.
    """
    if L is None:
        L = max((elements0[-1, 1], elements1[-1, 1]))
    mask = np.ones(L, dtype=bool)
    for (start, end) in elements0:
        if start >= L:
            continue
        if end > L:
            end = L
        mask[start:end] = False
    for (start, end) in elements1:
        if start >= L:
            continue
        if end > L:
            end = L
        mask[start:end] = True
    elements_out = mask_to_elements(mask)
    return elements_out


# DFE discretization and weighting


def integrate_with_weights(vals, weights, u_fac=1):
    """
    Takes weighted product of `vals`.
    """
    if len(vals) != len(weights):
        raise ValueError("values and weights are not same length")
    out = np.prod([v ** (w * u_fac) for v, w in zip(vals, weights)], axis=0)
    return out


def integrate_with_dfe(vals, ss, dfe, u_fac=1):
    """
    Computes DFE weights and uses them to integrate across `vals`.

    :param vals: Values to integrate
    :param ss: Array of selection coefficients
    :param dfe: Dictionary specifying DFE parameters. See `get_dfe_weights`.
    """
    weights = get_dfe_weights(ss, dfe)
    out = integrate_with_weights(vals, weights, u_fac=u_fac)
    return out


def get_dfe_weights(ss, dfe):
    """
    Get weights for a DFE of supported type.

    :param ss: Vector of selection coefficients
    :param dfe: Dictionary specifying DFE parameters

    :returns: Vector of DFE weights
    """
    if dfe["type"] == "gamma":
        weights = weights_gamma_dfe(ss, dfe["shape"], dfe["scale"])
    elif dfe["type"] == "gamma_neutral":
        weights = weights_gamma_neutral_dfe(
            ss, dfe["shape"], dfe["scale"], dfe["p_neu"])
    else:
        raise ValueError(f"DFE type {df['type']} is unknown")
    return weights


def weights_gamma_dfe(ss, shape, scale):
    """
    Computes discretized weights for selection coefficient grid `ss` using a
    gamma distribution.

    Zero weight is placed on the bin s = 0.

    :param ss: Selection coefficients (all ss <= 0)
    :param shape: Shape parameter
    :param scale: Scale parameter
    :param p_neu: Neutral fraction parameter

    :returns: Array of DFE weights with length `len(ss)`
    """
    # Make sure ss are sorted negative values
    assert np.all(ss <= 0)
    ss_sorted = np.sort(ss)
    if np.any(ss != ss_sorted):
        raise ValueError("selection values are not sorted")
    assert ss[-1] == [0]
    midpoints = (ss[1:] + ss[:-1]) / 2
    weights = np.zeros(len(ss))
    cdf = lambda x: stats.gamma.cdf(-x, shape, scale=scale)
    for i in range(len(ss)):
        # The bin about s = -1 extends to -infinity
        if i == 0 :
            weights[i] = 1 - cdf(midpoints[0])
        # The s = 0 bin receives 0 mass
        elif i == len(ss) - 1:
            weights[i] = 0
        # The smallest |s| bin above s = 0 receives all density down to s = 0
        elif i == len(ss) - 2:
            weights[i] = cdf(midpoints[-1])
        else:
            weights[i] = cdf(midpoints[i]) - cdf(midpoints[i + 1])
    return weights


def weights_gamma_neutral_dfe(ss, shape, scale, p_neu):
    """
    Computes discretized weights for selection coefficient grid `ss` using a
    gamma neutral distribution. Exactly `p_neu` mass is placed on the bin
    where s = 0.

    :param ss: Selection coefficients (all ss <= 0)
    :param shape: Shape parameter
    :param scale: Scale parameter
    :param p_neu: Neutral fraction parameter

    :returns: Array of DFE weights with length `len(ss)`
    """
    weights = weights_gamma_dfe(ss, shape, scale)
    weights *= (1 - p_neu)
    weights[-1] = p_neu
    return weights


# Scaling and manipulating tables


def scale_genome_table(df, scale):
    """
    Scales a table with genomic information, `df`, to a coarser resolution
    (larger window sizes).

    Tables must have these columns:
        chrom
        chromStart or start
        chromEnd or end

    If `num_sites` is present, then most quantities (with exceptions given
    below) are weighted by the number of sites in each original window. If
    it is not present, then simple averages are taken.

    For the fields `del_mut` and `exp_del_pi`, there must be a `del_sites`
    column in `df` for a weighted average to be computed. Otherwise, an error
    will be raised. The same holds for `neu_mut` and `exp_neu_mut` in relation
    to `neu_sites`.

    If `num_sites` is present, then for coarse windows with no data (`num_sites`
    equals zero), all average quantities equal np.nan and counts are zero.

    :param df: Table with window starts/ends and window information.
        Intervals need not all be of the same size, but they must all be
        divisible by the target `scale`.
    :param scale: Target resolution.
    """
    scale = int(scale)

    def get_avg(vals):
        if len(vals) == 0:
            ret = np.nan
        else:
            ret = np.mean(vals)
        return ret

    def weighted_avg(vals, weights):
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            ret = np.nan
        else:
            ret = (weights * vals).sum() / sum_weights
        return ret

    if "chrom" in df.columns:
        chrom = next(iter(df["chrom"]))
    elif "#chrom" in df.columns:
        chrom = next(iter(df["#chrom"]))
    else:
        chrom = "none"

    if "chromStart" in df:
        windows0 = np.stack([df["chromStart"], df["chromEnd"]], axis=1)
    elif "start" in df:
        windows0 = np.stack([df["start"], df["end"]], axis=1)
    else:
        raise Valuerror("could not find window start/ends")

    intervals0 = windows0[:, 1] - windows0[:, 0]
    assert np.all(scale % intervals0 == 0)

    L = windows0[-1, 1]
    windows1 = np.stack((np.arange(0, L, scale, dtype=np.int64),
        np.arange(scale, L + scale, scale, dtype=np.int64)), axis=1)

    # Find the index in `windows1` that corresponds to each `windows0`
    mapping = np.searchsorted(windows1[:, 1], windows0[:, 1])
    idxs = list(range(len(windows1)))

    # Start building output data
    data = {
        "chrom": [chrom] * len(windows1),
        "chromStart": windows1[:, 0],
        "chromEnd": windows1[:, 1]}

    # Iterate over fields and take weighted averages
    for col in df.columns[3:]:
        arr = np.array(df[col])

        # Sum over site counts
        if col in ["num_sites", "del_sites", "neu_sites"]:
            data[col] = [np.sum(arr[mapping == i]) for i in idxs]

        elif col in ["del_mut", "exp_del_pi"]:
            assert "del_sites" in df
            weights = np.array(df["del_sites"])
            data[col] = [weighted_avg(arr[mapping == i], weights[mapping == i])
                for i in idxs]

        elif col in ["neu_mut", "exp_neu_pi"]:
            assert "neu_sites" in df
            weights = np.array(df["neu_sites"])
            data[col] = [weighted_avg(arr[mapping == i], weights[mapping == i])
                for i in idxs]

        else:
            if "num_sites" in df:
                weights = np.array(df["num_sites"])
                data[col] = [weighted_avg(arr[mapping == i],
                                          weights[mapping == i]) for i in idxs]
            else:
                warnings.warn("num_sites is not present: taking simple avg")
                data[col] = [get_avg(arr[mapping == i]) for i in idxs]

    df_out = pandas.DataFrame(data)
    return df_out


# Assorted utilities


def get_max_distances(df, tolerance=1e-10):
    """
    """
    ss = np.sort(np.unique(df["s"]))
    thresholds = np.zeros(len(ss), dtype=np.float64)
    for i, s in enumerate(ss):
        Bs = np.array(df[df["s"] == s]["B"])
        Ms = np.array(df[df["s"] == s]["M"])
        assert np.all(np.sort(Ms) == Ms)
        deviations = 1 - Bs
        beyond_tolerance = np.where(deviations < tolerance)[0]
        if np.all(deviations > 0):
            thresholds[i] = np.inf
        else:
            idx = beyond_tolerance[0]
            thresholds[i] = Ms[idx]
    return thresholds


def convert_bedgraph_mutation_map(
    fname,
    out_fname,
    chrom_col="chr",
    start_col="start", 
    end_col="end", 
    rate_col="u"
):
    """
    Construct a bedgraph file formatted for use in the B prediction pipeline
    from a bedgraph file with only a `rate_col`.
    """
    df = pandas.read_csv(fname)
    starts = np.array(df[start_col])
    ends = np.array(df[end_col])
    avg_mut = np.array(df[rate_col])
    num_sites = ends - starts
    data = {
        "chrom": df[chrom_col],
        "chromStart": starts,
        "chromEnd": ends,
        "num_sites": num_sites, 
        "avg_mut": avg_mut,
        "num_sites_masked": num_sites, 
        "avg_mut_masked": avg_mut
    }
    pandas.DataFrame(data).to_csv(out_fname, index=False)
    return


def _get_time():
    """
    Return a string representing the time and date with yy-mm-dd format.
    """
    return '[' + datetime.strftime(datetime.now(), '%y-%m-%d %H:%M:%S') + ']'


def write_bedfile(fname, intervals, chrom):
    """
    Write an array of elements/regions to file in .bed format.

    :param str chrom: String representing the chromosome number.
    """
    if not isinstance(chrom, str): 
        chrom = str(chrom)
    data = {
        "chrom": [chrom] * len(intervals), 
        "chromStart": intervals[:, 0],
        "chromEnd": intervals[:, 1]
    }
    pandas.DataFrame(data).to_csv(fname, index=False, sep="\t")
    return


def write_uniform_mutation_interval(fname, L, u, chrom):
    """
    Write a one-line .csv file specifying an interval and uniform mutation rate. 
    """
    data = {
        "chrom": [chrom],
        "chromStart": [0],
        "chromEnd": [L],
        "rate": [u]
    }
    pandas.DataFrame(data).to_csv(fname, index=False)
    return
