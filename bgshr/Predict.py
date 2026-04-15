
import numpy as np
import warnings
from scipy import interpolate
import copy
import pandas
from multiprocessing import Pool

from . import Util, ClassicBGS


def Bvals(
    xs,
    s,
    splines,
    u=1e-8,
    L=None,
    rmap=None,
    r=None,
    bmap=None,
    elements=[],
    max_r=0.1,
    r_dists=None,
    B_elem=None,
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
    # Apply interference correction if bmap is provided
    if B_elem is None:
        B_elem = _get_B_per_element(bmap, elements)

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
        # TODO allow for constant recombination rate, handle with bmap if provided?
        if r is None:
            warnings.warn("No recombination rates provided, assuming r=1e-8")
            r = 1e-8
        rmap = Util.build_uniform_rmap(r, L)

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
            r_dists = _get_r_dists(xs, elements, rmap)

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
    Compute a matrix of recombination distance between focal neutral sites `xs`
    and constrained elements/windows `elements`. 
    """
    # element midpoints
    x_midpoints = np.mean(elements, axis=1)
    # get recombination map values
    r_midpoints = rmap(x_midpoints)
    r_xs = rmap(xs)
    r_dists = np.zeros((len(elements), len(xs)))
    # get recombination fractions
    r_dists[:, :] = Util.haldane_map_function(
        np.abs(r_xs[None, :] - r_midpoints[:, None])
    )
    return r_dists


def _get_interpolated_svals(s_elem, s_vals):
    s0 = s_vals[np.where(s_elem > s_vals)[0][-1]]
    s1 = s_vals[np.where(s_elem < s_vals)[0][0]]
    p1 = (s_elem - s0) / (s1 - s0)
    p0 = 1 - p1
    assert 0 < p0 < 1
    return s0, s1, p0, p1


def Bmap(
    xs,
    s,
    splines,
    u=1e-8,
    L=None,
    rmap=None,
    bmap=None,
    elements=[],
    max_r=0.1,
):
    bs = Bvals(
        xs,
        s,
        splines,
        u=u,
        L=L,
        rmap=rmap,
        bmap=bmap,
        elements=elements,
        max_r=max_r,
    )
    return interpolate.CubicSpline(xs, bs, bc_type="natural")


#
# def Bvals_dfe(
#    xs,
#    ss,
#    splines,
#    u=1e-8,
#    shape=None,
#    scale=None,
#    L=None,
#    rmap=None,
#    bmap=None,
#    elements=[],
#    max_r=0.1,
# ):
#    if shape is None or scale is None:
#        raise ValueError("must provide gamma dfe shape and scale")
#
#    Bs = Bvals(
#        xs, ss, splines, u=u, L=L, rmap=rmap, bmap=bmap, elements=elements, max_r=max_r
#    )
#    B = Util.integrate_gamma_dfe(Bs, ss, shape, scale)
#    return B


def _get_distances(xs, windows, rmap):
    """
    """
    window_mids = np.mean(elements, axis=1)
    window_map = rmap(window_mids)
    xs_map = rmap(xs)
    distances = np.abs(r_xs[np.newaxis, :] - r_midpoints[:, np.newaxis])
    return distances


def Bvals_fast():


    return














################################


def _table_expected_neu_pi0(df, neu_mut):
    """
    Compute average neutral pi0 in windows by scaling the `Hr` entry in the 
    lookup table to the local neutral mutation rate.
    """
    Hr = np.unique(df[df["s"] == 0]["Hr"])[0]
    uR = np.unique(df["uR"])[0]
    neu_pi0 = 2 * Hr * (neu_mut / uR)
    return neu_pi0


def _expected_del_pi0(dfes, s_vals, del_sites_arrs, uL_arrs, Ne=None, uL0=1e-8):
    """
    Compute pi0 for constrained sites, taking direct selection into account.
    For use with tables with equilibrium demographic histories. Works over
    one or more classes of constrained element, each with its own DFE.

    :param dfes: List of dictionaries defining DFE parameters for each class of
        constrained element.
    :param s_vals: Array of selection coefficients. We compute constrained pi0
        by calculating Hl (diversity at the selected locus) for each s value,
        then integrating over these using weights given by the DFEs.
    :param del_sites_arrs: List of arrays. Each array corresponds to a class
        of elements, and each element in one array corresponds to a genomic
        window.
    :param uL_arrs: List of arrays recording average mutation rate for each 
        class in genomic windows.
    :param Ne: Effective population size (default None).
    """
    if Ne is None: 
        raise ValueError("You must provide Ne")
    Hls = 2 * np.array([ClassicBGS._get_Hl(s, Ne, uL0) for s in s_vals])
    sum_del_pi0 = np.zeros(len(uL_arrs[0]), dtype=np.float64)
    for (dfe, uL_arr, del_sites) in zip(dfes, uL_arrs, del_sites_arrs):
        weights = Util._get_dfe_weights(dfe, s_vals)
        unit_del_pi0 = np.sum(weights * Hls)
        sum_del_pi0 += unit_del_pi0 * del_sites * (uL_arr / uL0)
    del_pi0 = np.zeros(len(del_sites), dtype=np.float64)
    tot_del_sites = np.sum(del_sites_arrs, axis=0)
    del_pi0[tot_del_sites > 0] = (
        sum_del_pi0[tot_del_sites > 0] / tot_del_sites[tot_del_sites > 0])
    return del_pi0


def _table_expected_del_pi0(df, dfes, del_sites_arrs, uL_arrs):
    """
    Compute window-average pi0 for constrained sites under direct selection from
    values stored in a lookup table. For use with tables with non-equilibrium
    demographic histories, where scaling table entries by a single value of Ne
    is erroneous.
    """
    df_sub = df[df["r"] == 0]
    uL0 = np.unique(df_sub["uL"])[0]
    s_vals = np.sort(np.unique(df_sub["s"]))
    Hls = 2 * np.array(
        [df_sub[df_sub["s"] == s]["Hl"] for s in s_vals]).flatten()
    sum_del_pi0 = np.zeros(len(uL_arrs[0]), dtype=np.float64)
    for (dfe, uL_arr, del_sites) in zip(dfes, uL_arrs, del_sites_arrs):
        weights = Util._get_dfe_weights(dfe, s_vals)
        unit_del_pi0 = np.sum(Hls * weights)
        sum_del_pi0 += unit_del_pi0 * del_sites * (uL_arr / uL0)
    del_pi0 = np.zeros(len(del_sites), dtype=np.float64)
    tot_del_sites = np.sum(del_sites_arrs, axis=0)
    del_pi0[tot_del_sites > 0] = (
        sum_del_pi0[tot_del_sites > 0] / tot_del_sites[tot_del_sites > 0])
    return del_pi0


def parallel_Bvals(
    xs,
    s_vals,
    splines,
    dfes,
    uL_windows,
    uL_arrs,
    rmap=None,
    r=None,
    B_map=None,
    B_elem=None,
    tolerance=None,
    df=None,
    block_size=5000,
    cores=None,
    verbose=True
):
    """
    Predict a B-landscape at focal neutral sites along a chromosome for one or 
    more classes of functionally constrained elements. 

    Requires that constrained sites are aggregated together into an array of 
    shared genomic windows `uL_windows`. The intensity of deleterious mutation
    (e.g. the number of constrained sites times the average deleterious mutation 
    rate) in each window is given by `uL_arrs`. This is a list of arrays, where
    each array corresponds to a class and each array element to a window in 
    `uL_windows`. Deleterious rates are scaled by 1/`u0`, the deleterious 
    mutation rate embodied in the precomputed lookup table. When we have an 
    equilibrium demographic model, uL arrays may also be scaled by the ratio 
    Ne/Ne0, where Ne is the effective size we wish to model and Ne0 is the 
    effective size embodied in the lookup table. 

    We collect constrained elements into shared windows in this manner to 
    optimize predictions. The tradeoff, a loss of precision in the distances
    between constrained and focal sites, is tolerable, especially when we wish
    to consider landscapes at larger scales down the line.

    :param xs: Array holding the positions of focal neutral sites.
    :param s_vals: Array of selection coefficients. These should be a subset of
        the s-values in `splines`. The last element of `s_vals` must be 0.
    :param splines: A dictionary that maps (u, s) tuples to cubic splines which
        interpolate "unit" B values as a function of recombination distance. 
    :param dfes: A list of dictionaries defining parameters of gamma-distributed 
        DFEs. These must have keys "shape", "scale" and "type". "type" should 
        map to "gamma" or "gamma_neutral"; if "gamma_neutral", an additional
        parameter "p_neu" defining the fraction of neutral mutations is needed.
    :param uL_arrs: List of arrays holding deleterious mutation factors. Each 
        element in an array corresponds to a window in `uL_windows`. Scaling is 
        twofold: first, uL is a product of the average deleterious mutation rate 
        and the number of constrained sites that belong to the class of interest 
        in each window. Second, uL is normalized by u0, the mutation rate that 
        was used to build the input lookup table- and under equilibrium 
        demographies where the Ne for which we wish to predict differs from the 
        Ne used to build the lookup table, uL is also scaled by Ne / Ne0 where 
        Ne0 is the table Ne.
    :param uL_windows: Array recording windows that hold constrained sites. The
        density of constrained sites in each window is reflected in `uL_arrs`.
    :param rmap: Recombination map function (default None builds a uniform map).
    :param r: Optional uniform recombination map rate (defaults to 1e-8).
    :param B_map: Optional function, interpolating physical coordinates to B
        values from a prior prediction- used to rescale local parameters in the
        interference correction. Mutually exclusive with `B_elem` (default None)
    :param B_elem: Optional array specifying average B values in each 
        constrained window, obtained from a prior round of prediction. For use
        in the interference correction and mutually exclusive with `B_map`
        (default None)
    :param tolerance:
    :param df: Lookup table, loaded as a pandas dataframe. Required when 
        specifying a `tolerance` (default None).
    :param block_size: Number of focal neutral sites/constrained elements to 
        handle at once on each core (default 5000). Expect that memory 
        requirements scale as the square of this quantity at least.
    :param cores: Number of cores across which to parallelize predictions. If
        1 or None, simply loops over pairs of neutral/constrained blocks.
    :param verbose: If True (default), print reports as block predictions run.
    
    :returns: An array of predicted B-values matching `xs` in shape.
    """
    # If a B-map was given, prepare to perform interference correction
    if B_map is not None:
        if B_elem is None:
            B_elem = _get_B_per_element(B_map, uL_windows)
        else:
            raise ValueError("You cannot give both `B_elem` and `B_map`")

    # Build a uniform recombination map if one was not provided
    if rmap is None:
        L = max([xs[-1], uL_windows[-1][1]])
        if r is None:
            print("No recombination rates provided, assuming r=1e-8")
            r = 1e-8
        rmap = Util.build_uniform_rmap(r, L)

    if np.isscalar(s_vals):
        s_vals = np.array([s_vals])

    # If an error tolerance is provided, find a maximum r value for each s
    if tolerance is not None:
        if df is None:
            raise ValueError(
                "You must provide a lookup table to use `tolerance`")
        thresholds = _get_r_thresholds(df, tolerance=tolerance)
    else:
        thresholds = None

    # Group focal sites and constrained elements into windows
    build_blocks = lambda xs, size: [
        np.arange(start_idx, end_idx) for start_idx, end_idx in 
        zip(np.arange(0, len(xs), size), 
            np.append(np.arange(size, len(xs), size), len(xs)))
    ]
    xblocks = build_blocks(xs, block_size)
    ublocks = build_blocks(uL_windows, block_size)

    Bs = np.ones(len(xs), dtype=np.float64)

    if cores is None or cores == 1:
        for ii, xblock in enumerate(xblocks):
            for jj, ublock in enumerate(ublocks):
                block_uL_arrs = [uLs[ublock] for uLs in uL_arrs]
                if B_elem is not None:
                    block_B_elem = B_elem[ublock]
                else:
                    block_B_elem = None
                Bs[xblock] *= _predict_block_B(
                    xs[xblock],
                    s_vals,
                    splines,
                    uL_windows[ublock],
                    block_uL_arrs,
                    rmap,
                    dfes,
                    thresholds=thresholds,
                    B_elem=block_B_elem
                )       
                if verbose: 
                    num_blocks = (len(xblocks), len(ublocks))
                    print(Util._get_time(), 
                        f"Predicted B in block {(ii+1, jj+1)} of {num_blocks}")
    else:
        pairings = [(xb, ub) for xb in xblocks for ub in ublocks]
        groups = [pairings[i:i + cores] for i in range(0, len(pairings), cores)]
        for ii, group in enumerate(groups):
            args = []
            for xblock, ublock in group:
                block_uL_arrs = [uLs[ublock] for uLs in uL_arrs]
                if B_elem is not None:
                    block_B_elem = B_elem[ublock]
                else:
                    block_B_elem = None
                args.append((
                    xs[xblock],
                    s_vals,
                    splines,
                    uL_windows[ublock],
                    block_uL_arrs,
                    rmap,
                    dfes,
                    thresholds,
                    block_B_elem
                ))
            with Pool(len(group)) as p:
                group_Bs = p.starmap(_predict_block_B, args)
            for (xblock, _), window_B in zip(group, group_Bs):
                Bs[xblock] *= window_B
            if verbose: 
                if ii == 0:
                    num_pairs = len(pairings)
                    progress = len(group)
                else:
                    progress += len(group)
                print(Util._get_time(), 
                    f"Predicted B in {progress} of {num_pairs} blocks")
    return Bs


def _predict_block_B(    
    xs,
    s_vals,
    splines,
    uL_windows,
    uL_arrs,
    rmap,
    dfes,
    thresholds=None,
    B_elem=None
):
    """
    Take the element class and s-value-specific B predictions made by 
    `_predict_block_B_classwise` and aggregate them together into a vector of
    site B predictions. Integrates each class B array with weights from the 
    class DFE, then takes the element-wise product of these.

    :param xs: Vector of focal neutral sites.
    :param s_vals: Vector of selection coefficients. This should be a subset
        of the selection coefficients in `splines`.
    :param splines: Dictionary of cubic splines mapping recombination
        distances to B values. Dictionary keys have the form (u0, s).
    :param uL_windows: An array of windows that hold constrained sites in one 
        or more element classes.
    :param uL_arrs: List of vectors, each holding the intensity of deleterious
        mutation across windows for a class. These factors are windowed sums of
        deleterious mutation rates scaled by 1/u0.
    :param rmap: Recombination map function, mapping physical coordinates to
        map coordinates in M.
    :param dfes: List of dictionaries, defining parameters of the gamma DFE for
        each class.
    :param thresholds: Array specifying recombination distance thresholds for
        each element of `s_vals`. If *all* focal site-constrained window 
        distances exceed the threshold for an s value, diversity reduction 
        exerted by mutations at that s value is treated as negligible and
        prediction is skipped.
    :param B_elem: Array specifying diversity reductions on each window, for 
        use in the interference correction.

    :returns: A vector of sitewise predicted B values.
    """
    B_arrays = _predict_block_B_classwise(
        xs,
        s_vals,
        splines,
        uL_windows,
        uL_arrs,
        rmap,
        thresholds=thresholds,
        B_elem=B_elem
    )
    B_vec = np.ones(len(xs))
    for B_vals, dfe in zip(B_arrays, dfes):
        weights = Util._get_dfe_weights(dfe, s_vals)
        B_vec *= Util.integrate_with_weights(B_vals, weights[:-1])
    return B_vec


def _predict_block_B_classwise(
    xs,
    s_vals,
    splines,
    uL_windows,
    uL_arrs,
    rmap,
    thresholds=None,
    B_elem=None
):
    """
    Make B predictions for one or more classes of constrained elements using an 
    efficient vectorized algorithm.

    Returns a list of 2d B prediction arrays. Arrays correspond to constrained
    element classes and resemble the output of `Bvals` in shape, e.g.
    (number of s values, number of xs). 

    See `_predict_block_B` for parameter definitions.
    :returns: A list of arrays holding predicted B values for each site and
        selection coefficient.
    """
    uL0 = np.unique([k[0] for k in splines.keys()])[0]
    r_dists = _get_r_dists(xs, uL_windows, rmap)
    B_arrays = [np.ones((len(s_vals[:-1]), len(xs))) for uLs in uL_arrs]
    # Initial predictions
    if B_elem is None:
        for ii, s_val in enumerate(s_vals[:-1]):
            if thresholds is not None:
                if np.all(r_dists > thresholds[ii]):
                    continue
            unit_Bs = splines[(uL0, s_val)](r_dists)
            for jj, uL_arr in enumerate(uL_arrs):
                B_arrays[jj][ii] = np.prod(unit_Bs ** uL_arr[:, None], axis=0)
    # Predictions under the interference correction
    else:
        for ii, s_val in enumerate(s_vals[:-1]):
            if thresholds is not None:
                if np.all(r_dists > thresholds[ii]):
                    continue
            s_elems = s_val * B_elem
            unit_Bs = _get_interpolated_B_vals(
                xs, s_elems, s_vals, r_dists, splines, uL0=uL0)
            for jj, uLs in enumerate(uL_arrs):
                B_arrays[jj][ii] = np.prod(unit_Bs ** uLs[:, None], axis=0)
    return B_arrays


def _adjust_uL_arrays(B_map, uL_arrs, uL_windows):
    """
    Scale deleterious mutation rate factors uL by local B-value. Used in the
    interference correction. 

    :param B_map: A function mapping coordinates to diversity reductions.
    :param uL_arrs: List of deleterious mutation intensity arrays for one or 
        more classes of constrained sites.
    :param uL_windows: Array of windows containing constrained sites.
    """
    window_Bs = _get_B_per_element(B_map, uL_windows)
    adjusted_uL_arrays = [uLs * window_Bs for uLs in uL_arrs]
    return adjusted_uL_arrays


def _get_interpolated_B_vals(xs, s_elems, s_vals, r_dists, splines, uL0=1e-8):
    """
    For use in the interference correction. After scaling an s value by local
    diversity reduction (`s_elems`), interpolate between splines to find the 
    appropriate B value for each element of `s_elems`.

    :param xs: Array of focal neutral sites.
    :param s_elems: Array of local scalings of a selection coefficient.
    :param s_vals: Array of selection coefficients in `splines`.
    :param r_dists: Array of recombination distances between focal sites and
        constrained elements.
    :param splines: Dictionary of cubic splines.
    :param u0: Mutation rate embodied in cubic splines.

    :returns: Array of scaled unit B values.
    """
    unit_Bs = np.zeros((len(s_elems), len(xs)))
    for i, s_elem in enumerate(s_elems):
        s0, s1, p0, p1 = _get_interpolated_svals(s_elem, s_vals)
        fac0 = p0 * splines[(uL0, s0)](r_dists[i])
        fac1 = p1 * splines[(uL0, s1)](r_dists[i])
        unit_Bs[i] = fac0 + fac1
    return unit_Bs


def _get_r_thresholds(df, tolerance=1e-10):
    """
    Compute distance thresholds in `r`, beyond which B values in the lookup
    table `df` differ from 1 by less than `tolerance`. 

    Used when predicting B in blocks. For a selection coefficient s, if all
    focal neutral site/constrained site distances exceed the threshold assigned
    to s, then s is skipped.
    
    :param df: Lookup table loaded as a pandas dataframe.
    :param tolerance: Maximum allowable deviation of B value from 1 
        (default 1e-10).
    :returns: Array of recombination distances satisfying `tolerance`.
    """
    s_vals = np.sort(np.unique(df["s"]))
    thresholds = np.zeros(len(s_vals), dtype=np.float64)
    for i, s in enumerate(s_vals):
        B_vals = np.array(df[df["s"] == s]["B"])
        r_vals = np.array(df[df["s"] == s]["r"])
        assert np.all(np.sort(r_vals) == r_vals)
        deviations = 1 - B_vals
        beyond_tolerance = np.where(deviations < tolerance)[0]
        # If all deviations are beyond tolerance, make the threshold r=0.5
        if len(beyond_tolerance) == 0:
            thresholds[i] = 0.5
        else:
            threshold_idx = beyond_tolerance[0]
            thresholds[i] = r_vals[threshold_idx]
    return thresholds
