
from concurrent import futures
import numpy as np
import warnings
from scipy import interpolate
import copy
import pandas
from multiprocessing import Pool

from . import Util, ClassicBGS, Predict


def get_Bmap(xs, Bs):
    """
    Get a function that interpolates B-values at non-focal sites, using cubic
    splines.
    """
    return interpolate.CubicSpline(xs, Bs, bc_type="natural")


def Bvals(
    xs,
    s,
    splines,
    u=1e-8,
    L=None,
    rmap=None,
    r=None,
    Bmap=None,
    elements=[],
    max_r=0.1,
    r_dists=None
):
    """
    Get predicted reduction for given s-value, assuming a constant per-base
    deleterious mutation rate of u within elements.

    If s is a single value, we compute just a single B value reduction array
    over the xs values. If s is a list of values, we repeat for each s in that
    list, and return a list of B value recution arrays.

    The mutation rate(s) u can be a single scalar or a list the length of
    elements, in which case each element has its own average mutation rate.
    """

    # Get grid of s and u values
    u_vals = np.sort(list(set([k[0] for k in splines.keys()])))
    s_vals = np.sort(list(set([k[1] for k in splines.keys()])))
    if len(u_vals) != 1:
        raise ValueError("more than one mutation rate present")
    uL = u_vals[0]

    # Build recombination map if needed
    if rmap is None:
        if L is None:
            L = max([xs[-1], elements[-1][1]])
        if r is None:
            warnings.warn("No recombination rates provided, assuming r=1e-8")
            r = 1e-8
        rmap = Util.build_uniform_rmap(r, L)

    # Apply interference correction if Bmap is provided
    if Bmap is not None:
        B_elem = _get_B_per_element(Bmap, elements)
        rmap = Util.adjust_recombination_map(rmap, Bmap)
    else:
        B_elem = np.ones(len(elements))

    ## TODO: pull the rec map adjustment back inside this function, since
    ## recursive calls should not recompute distances from the rec map

    # Allow for multiple s values to be computed, to reduce number
    # of calls to the recombination map when integrating over a DFE
    if np.isscalar(s):
        all_s = [s]
    else:
        all_s = [_ for _ in s]
    # Each element could have its own average mutation rate
    if np.isscalar(u):
        all_u = [u for _ in elements]
    else:
        all_u = [_ for _ in u]
    if not len(all_u) == len(elements):
        raise ValueError("list of mutation rates and elements must be same length")

    Bs = [np.ones(len(xs)) for s_val in all_s]
    if len(elements) > 0:
        if r_dists is None:
            # recombination distances between each position in xs and element midpoint
            # r_dists = _get_r_dists(xs, elements, rmap)
            r_dists = _get_distances(xs, elements, rmap)

        for j, s_val in enumerate(all_s):
            for i, (e, u_val) in enumerate(zip(elements, all_u)):
                # if we correct for local B value, update the s value in this element
                u_elem = B_elem[i] * u_val
                s_elem = B_elem[i] * s_val
                # scalar factor to account for differences btw input u and lookup table
                u_fac = u_elem / uL
                if s_elem in s_vals:
                    # get length of the constrained element
                    L_elem = e[1] - e[0]
                    # restrict to distances within the maximum recombination distance
                    indexes = np.where(r_dists[i] <= max_r)[0]
                    r_dist = r_dists[i][indexes]
                    fac = splines[(uL, s_elem)](r_dist) ** (u_fac * L_elem)
                    Bs[j][indexes] *= fac
                else:
                    # call this function recursively to interpolate between the s-grid
                    s0, s1, p0, p1 = _get_interpolated_svals(s_elem, s_vals)
                    # adjust mutation rates by interpolation and any local B correction
                    u0 = u_val * B_elem[i] * p0
                    u1 = u_val * B_elem[i] * p1
                    fac0 = Bvals(
                        xs,
                        s0,
                        splines,
                        u=u0,
                        L=L,
                        rmap=rmap,
                        elements=[e],
                        max_r=max_r,
                        r_dists=[r_dists[i]],
                    )
                    fac1 = Bvals(
                        xs,
                        s1,
                        splines,
                        u=u1,
                        L=L,
                        rmap=rmap,
                        elements=[e],
                        max_r=max_r,
                        r_dists=[r_dists[i]],
                    )
                    Bs[j] *= fac0 * fac1

    if np.isscalar(s):
        assert len(Bs) == 1
        return Bs[0]
    else:
        return Bs


def Bvals_dfe(
    xs,
    splines,
    ss=None,
    u=1e-8,
    L=None,
    rmap=None,
    r=None,
    elements=None,
    max_dist=0.1,
    Bmap=None,
    dfe=None
):
    """
    Integrate across a DFE.
    """
    if ss is None:
        ss = np.sort(list(set([k[1] for k in splines.keys()])))

    Bs = Bvals(
        xs,
        ss,
        splines,
        u=u,
        L=L,
        rmap=rmap,
        r=r,
        elements=elements,
        max_r=max_dist,
        Bmap=Bmap
    )
    B = Util.integrate_with_dfe(Bs, ss, dfe)
    return B


def Bvals_dfes(
    xs,
    splines,
    ss=None,
    u=None,
    u_arrs=[],
    L=None,
    rmap=None,
    r=None,
    all_elements=[],
    max_dist=0.1,
    Bmap=None,
    dfes=None,
):
    """
    Takes input data structures similar to `Bvals_fast` and operates on one or
    more DFEs.
    """

    if shapes is None or scales is None:
        raise ValueError("must provide gamma dfe shape and scale")

    if ss is None:
        ss = np.sort(list(set([k[1] for k in splines.keys()])))

    if p_neus is None:
        p_neus = [0] * len(all_elements)

    Bs = []
    for i in range(len(all_elements)):
        Bs_i = Bvals_dfe(
            xs,
            splines,
            ss=ss,
            u=u_arrs[i],
            L=L,
            rmap=rmap,
            r=r,
            elements=all_elements[i],
            max_dist=max_dist,
            Bmap=Bmap,
            dfe=dfes[i]
        )
        Bs.append(Bs_i)
    Bs_out = np.prod(Bs, axis=0)
    return Bs_out


def split_U_windows(windows, U_arrs):
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


def Bvals_fast(
    xs,
    splines,
    windows=None,
    U_arrs=None,
    rmap=None,
    r=None,
    L=None,
    ss=None,
    max_dists=None,
    dfes=None,
    Bmap=None,
    chunk_size=1000,
    n_cores=1
):
    """
    Get predicted diversity reductions given one or more classes of constrained
    elements, each with a distinct DFE. The deleterious mutation rates of each
    class (`U_arrs`) must be specified on a set of genomic `windows` that are
    shared across classes. Designed to handle multiple DFEs efficiently.

    Features:
    - If DFE parameters are given, integrates across B-values. Otherwise returns
      arrays with B-values for each selection coefficient.
    - Interference correction: local scaling is handled internally

    """
    if windows is None or U_arrs is None:
        raise Valuerror("Must provide `windows` and `U_arrs`")

    # Build a uniform recombination map if needed
    if rmap is None:
        if L is None:
            L = max([xs[-1], elements[-1][1]])
        if r is None:
            warnings.warn("No recombination rate provided, assuming r=1e-8")
            r = 1e-8
        rmap = Util.build_uniform_rmap(r, L)

    # The function calls itself recursively to handle many focal sites
    if len(xs) > chunk_size:
        # Set up "chunks"; continuous blocks of focal sites (xs)
        n_chunks = int(np.ceil(len(xs) / chunk_size))
        chunks = [xs[i * chunk_size:(i + 1) * chunk_size]
                  for i in range(n_chunks)]

        # Static arguments
        args = (
            splines,
            windows,
            U_arrs,
            rmap,
            r,
            L,
            ss,
            max_dists,
            dfes,
            Bmap,
            chunk_size,
            n_cores
        )

        # If several cores were alloted, parallelize computation across them
        if n_cores > 1:
            map_args = [chunks] + [[arg] * n_chunks for arg in args]
            with futures.ProcessPoolExecutor(max_workers=n_cores) as ex:
                chunk_Bs = list(ex.map(Bvals_fast, *map_args))
        # Otherwise, perform vectorized computations in a loop
        else:
            chunk_Bs = []
            for chunk in chunks:
                chunk_Bs.append(Bvals_fast(chunk, *args))

        # If DFEs were given, output is a vector and can be concatenated
        if dfes is not None:
            Bs_out = np.concatenate(chunk_Bs)
        # Otherwise, output is a list of 2d arrays (see docstring)
        else:
            Bs_out = [np.hstack([chunk_Bs[i][j] for i in range(len(chunk_Bs))])
                      for j in range(len(chunk_Bs[0]))]

    else:
        # Apply interference correction if Bmap was given
        if Bmap is not None:
            rmap = Util.adjust_recombination_map(rmap, Bmap)
            B_windows = _get_B_per_element(Bmap, windows)
            assert np.all(B_windows < 1)
            U_arrs = adjust_mutation_arrays(U_arrs, windows, Bmap)

        # Get grid of s and u values
        uLs = np.sort(list(set([k[0] for k in splines.keys()])))
        if ss is None:
            ss = np.sort(list(set([k[1] for k in splines.keys()])))
            # assert np.min(ss) == -1 and np.max(ss) == 0
        assert len(uLs) == 1
        uL = uLs[0]

        # Scale deleterious mutation rates by `uL` from the table
        U_arrs = [Us / uL for Us in U_arrs]

        # Instantiate B arrays
        Bs = [np.ones((len(ss), len(xs))) for _ in U_arrs]

        # Loop over s coefficients and constraint categories
        distances = _get_distances(xs, windows, rmap)
        for i, s in enumerate(ss):
            if s == 0:
                continue
            if max_dists is not None:
                max_dist = max_dists[i]
                # Indices ...
                lower_lim = np.searchsorted(
                    _get_signed_distances(xs[0], windows, rmap), -max_dist)
                upper_lim = np.searchsorted(
                    _get_signed_distances(xs[-1], windows, rmap), max_dist)
            else:
                lower_lim, upper_lim = 0, -1

            if Bmap is None:
                unit_Bs = splines[(uL, s)](distances[lower_lim:upper_lim])
            else:
                s_windows = s * B_windows[lower_lim:upper_lim]
                unit_Bs = _interpolate_Bs(
                    xs, s_windows, ss, distances[lower_lim:upper_lim], splines)

            for j, Us in enumerate(U_arrs):
                Bs[j][i] = np.prod(
                    unit_Bs ** Us[lower_lim:upper_lim, None], axis=0)

        # Integrate across Bs
        if dfes is not None:
            assert len(dfes) == len(U_arrs)
            Bs_out = np.prod([Util.integrate_with_dfe(Bs_dfe, ss, dfe)
                            for Bs_dfe, dfe in zip(Bs, dfes)], axis=0)

        # If no DFEs were given, return full Bs array
        else:
            Bs_out = Bs

    return Bs_out


# B-prediction utilities


def _get_B_per_element(bmap, elements):
    if bmap is None:
        # without bmap, correction factor is 1
        B_elem = np.ones(len(elements))
    else:
        # average B in each element
        B_elem = np.array(
            [bmap.integrate(e[0], e[1]) / (e[1] - e[0]) for e in elements]
        )
    return B_elem


def _get_element_midpoints(elements):
    x = np.zeros(len(elements))
    for i, e in enumerate(elements):
        x[i] = np.mean(e)
    return x


def _get_r_dists(xs, elements, rmap):
    """
    Compute a matrix of recombination distances between focal neutral sites `xs`
    and constrained elements/windows `elements`.
    """
    x_midpoints = np.mean(elements, axis=1)
    r_midpoints = rmap(x_midpoints)
    r_xs = rmap(xs)
    r_dists = np.zeros((len(elements), len(xs)))
    r_dists[:, :] = Util.haldane_map_function(
        np.abs(r_xs[None, :] - r_midpoints[:, None]))
    return r_dists


def _interpolate_Bs(xs, s_elems, s_vals, distances, splines):
    """

    """

    # Retrieve uL
    uLs = np.sort(list(set([k[0] for k in splines.keys()])))
    assert len(uLs) == 1
    uL = uLs[0]

    unit_Bs = np.zeros((len(s_elems), len(xs)))
    for i, s_elem in enumerate(s_elems):
        s0, s1, p0, p1 = _get_interpolated_svals(s_elem, s_vals)
        fac0 = p0 * splines[(uL, s0)](distances[i])
        fac1 = p1 * splines[(uL, s1)](distances[i])
        unit_Bs[i] = fac0 + fac1
    return unit_Bs


def _get_interpolated_svals(s_elem, s_vals):
    """
    """
    s0 = s_vals[np.where(s_elem > s_vals)[0][-1]]
    s1 = s_vals[np.where(s_elem <= s_vals)[0][0]]
    p1 = (s_elem - s0) / (s1 - s0)
    p0 = 1 - p1
    assert 0 < p0 < 1
    return s0, s1, p0, p1


def _get_distances(xs, windows, rmap):
    """
    """
    return np.abs(_get_signed_distances(xs, windows, rmap))


def _get_signed_distances(xs, windows, rmap):
    """
    """
    window_mids = np.mean(windows, axis=1)
    window_map = rmap(window_mids)
    xs_map = rmap(xs)
    if np.isscalar(xs):
        distances = window_map - xs_map
    else:
        distances = window_map[:, np.newaxis] - xs_map[np.newaxis, :]
    return distances


def adjust_mutation_arrays(U_arrs, windows, Bmap):
    """
    """
    window_Bs = _get_B_per_element(Bmap, windows)
    adjusted_U_arrs = [Us * window_Bs for Us in U_arrs]
    return adjusted_U_arrs


# Computing pi0


def get_expected_pi0(
    df,
    avg_u,
    num_sites,
    u_arrs=None,
    num_sites_dfes=None,
    dfes=None
):
    """
    Compute expected pi0 using windowed mutation rate data for several DFEs.

    :param df: Lookup table
    :param avg_u: Array of average mutation rates per window
    :param num_sites: Array of accessible site counts per window
    :param u_arrs: List of arrays with average mutation rates for each DFE class
    :param num_sites_dfes: List of accessible site counts for each DFE class
    """
    assert len(dfes) == len(num_sites_dfes) == len(u_arrs)

    # Check lookup table properties
    assert len(set(df["uL"])) == 1
    assert len(set(df["uR"])) == 1
    uL = next(iter(set(df["uL"])))

    # Neutral diversity
    del_sites = np.sum(num_sites_dfes, axis=0)
    neu_sites = num_sites - del_sites
    assert np.all(neu_sites >= 0)
    if len(u_arrs) > 0:
        del_U = np.sum([u * L for L, u in zip(num_sites_dfes, u_arrs)], axis=0)
        neu_U = (avg_u * num_sites) - del_U
        neu_u = np.zeros(len(neu_U))
        neu_u[neu_sites > 0] = neu_U[neu_sites > 0] / neu_sites[neu_sites > 0]
        assert np.all(neu_u >= 0)
    else:
        neu_u = avg_u
    pi0_df = np.unique(df["pi0"])[0]
    neu_pi0 = 2 * pi0_df * neu_u / uL

    # Deleterious diversity
    if len(u_arrs) > 0:
        sum_del_pi0 = 0.0
        for u_arr, dfe_sites, dfe in zip(u_arrs, num_sites_dfes, dfes):
            pi_dfe = _get_del_pi0_dfe(df, dfe)
            sum_del_pi0 += pi_dfe * u_arr / uL * dfe_sites
        # Average across deleterious diversity
        del_pi0 = np.zeros(len(sum_del_pi0))
        del_pi0[del_sites > 0] = (
            sum_del_pi0[del_sites > 0] / del_sites[del_sites > 0])
    else:
        del_pi0, del_sites = 0, 0

    # Weighted avg of neutral and deleterious diversity
    pi0 = np.zeros(len(num_sites))
    pi0[num_sites > 0] = (
        (neu_pi0 * neu_sites + del_pi0 * del_sites)[num_sites > 0]
        / num_sites[num_sites > 0])
    return pi0, neu_pi0, del_pi0


def _get_del_pi0_dfe(df, dfe):
    """
    Integrate `Hl` (deleterious pi0) from a lookup table `df` across a gamma
    or gamma-neutral DFE.
    """
    df_sub = df[df["r"] == 0]
    ss = np.sort(df_sub["s"])
    assert ss[0] == -1 and ss[-1] == 0
    weights = Util.get_dfe_weights(ss, dfe)
    Hls = 2 * np.array([df_sub[df_sub["s"] == s]["Hl"].iloc[0] for s in ss])
    del_pi0 = np.sum(weights * Hls)
    return del_pi0
