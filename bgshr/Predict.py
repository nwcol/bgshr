
from concurrent import futures
import numpy as np
import warnings
from scipy import interpolate
import copy
import pandas

from . import Util, ClassicBGS, Predict


def get_Bmap(xs, Bs):
    """
    Get a function that interpolates B-values at non-focal sites, using cubic
    splines.
    """
    return interpolate.CubicSpline(xs, Bs, bc_type="natural")


def Bvals(
    xs,
    ss,
    splines,
    u=1e-8,
    L=None,
    rmap=None,
    r=None,
    bmap=None,
    elements=[],
    max_r=0.1,
    r_dists=None,
    B_elem=None
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

    # Apply interference correction if bmap is provided
    if bmap is not None:
        if B_elem is None:
            B_elem = _get_B_per_element(bmap, elements)
        rmap = Util.adjust_recombination_map(rmap, bmap)
    else:
        if B_elem is None:
            B_elem = np.ones(len(elements))

    ## TODO: pull the rec map adjustment back inside this function, since
    ## recursive calls should not recompute distances from the rec map

    # Allow for multiple s values to be computed, to reduce number
    # of calls to the recombination map when integrating over a DFE
    if np.isscalar(ss):
        all_s = [ss]
    else:
        all_s = [_ for _ in ss]
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
            if s_val == 0:
                continue
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

    if np.isscalar(ss):
        assert len(Bs) == 1
        return Bs[0]
    else:
        return Bs


def _get_B_per_element(bmap, elements):
    """
    Computes the average B-value across each interval in `elements`.

    :param bmap: Scipy cubic splines function
    :param elements: Array of element start/end positions

    :returns: Array of element-specific average B-values
    """
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
    return np.mean(elements, axis=1)


def _get_r_dists(xs, elements, rmap):
    """
    Computes a matrix of recombination distances between focal sites `xs`
    and constrained elements `elements`.

    :param xs: Array of focal sites
    :param elements: Array of element start/end positions
    :param rmap: Recombination map function

    :returns: Array of distances with shape (len(elements), len(xs))
    """
    x_midpoints = np.mean(elements, axis=1)
    r_midpoints = rmap(x_midpoints)
    r_xs = rmap(xs)
    r_dist = Util.haldane_map_function(
        np.abs(r_xs[None, :] - r_midpoints[:, None]))
    return r_dist


def _get_interpolated_svals(s_elem, s_vals):
    """
    Computes the nearest elements in the s-grid `s_vals` for an arbitrary
    selection coefficient `s_elem`, and weights determined by its proximity to
    each grid point.

    For use in the interference correction.

    :param s_elem: Scalar target s-coefficient
    :param s_vals: Array of grid s-coefficients

    :returns: Nearest values in the s-grid and corresponding weights
    """
    s0 = s_vals[np.where(s_elem > s_vals)[0][-1]]
    s1 = s_vals[np.where(s_elem <= s_vals)[0][0]]
    p1 = (s_elem - s0) / (s1 - s0)
    p0 = 1 - p1
    assert 0 < p0 < 1
    return s0, s1, p0, p1


def Bvals_dfe(
    xs,
    splines,
    ss=None,
    u=None,
    L=None,
    rmap=None,
    r=None,
    elements=[],
    max_r=0.1,
    r_dists=None,
    dfe=None,
    Bmap=None
):
    """
    Computes B-values using `Bvals` and integrates them across a DFE.
    TODO document
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
        max_r=max_r,
        bmap=Bmap
    )

    if dfe is not None:
        Bs_out = Util.integrate_with_dfe(Bs, ss, dfe)
    else:
        Bs_out = Bs
    return Bs_out


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
    max_r=0.1,
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
            max_r=max_r,
            Bmap=Bmap,
            dfe=dfes[i]
        )
        Bs.append(Bs_i)
    Bs_out = np.prod(Bs, axis=0)
    return Bs_out


def interference_Bvals(
    xs,
    splines,
    windows=None,
    U_arrs=None,
    rmap=None,
    max_dist=None,
    dfes=None,
    n_corrs=0,
    chunk_size=1000,
    n_cores=1,
    B_unlinked=None,
    verbose=False
):
    """
    Predicts B with several rounds of interference correction. Returns B-arrays
    from each round of correction and uses `Bvals_fast`.
    """
    if B_unlinked is None:
        B_unlinked = 1
        Bmap = None
    else:
        assert B_unlinked < 1
        L = max([windows[-1, 1], xs[-1]])
        Bmap = get_Bmap([0, L], [B_unlinked, B_unlinked])

    Bs = Bvals_fast(
        xs,
        splines,
        windows=windows,
        U_arrs=U_arrs,
        rmap=rmap,
        max_dist=max_dist,
        dfes=dfes,
        Bmap=Bmap,
        chunk_size=chunk_size,
        n_cores=n_cores
    )
    Bs *= B_unlinked
    interf_Bs = [Bs]
    if verbose:
        print(Util._get_time(), "completed initial B-prediction")

    for i in range(n_corrs):
        Bs0 = Bs
        Bmap = get_Bmap(xs, Bs0)
        Bs = Bvals_fast(
            xs,
            splines,
            windows=windows,
            U_arrs=U_arrs,
            rmap=rmap,
            max_dist=max_dist,
            dfes=dfes,
            Bmap=Bmap,
            chunk_size=chunk_size,
            n_cores=n_cores
        )
        Bs *= B_unlinked
        interf_Bs.append(Bs)
        if verbose:
            print(Util._get_time(), f"completed interference correction {i+1}")
    return interf_Bs


def Bvals_fast(
    xs,
    splines,
    windows=None,
    U_arrs=None,
    rmap=None,
    r=None,
    L=None,
    ss=None,
    max_dist=None,
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
            max_dist,
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

        # Compute a maximum distance for each s-coefficient
        test_dists = np.logspace(-8, 1, 200)
        max_dev = 1e-10
        max_dists = []
        for s in ss:
            idx = np.where(splines[(uL, s)](test_dists) > 1 - max_dev)[0][0]
            max_dist_s = test_dists[idx]
            if max_dist is not None:
                max_dist_s = min(max_dist, max_dist_s)
            max_dists.append(max_dist_s)

        # Loop over s coefficients and constraint categories
        distances = _get_distances(xs, windows, rmap)
        for i, s in enumerate(ss):
            if s == 0:
                continue

            # Find the first/last windows to consider, given s and `max_dists`
            max_dist = max_dists[i]
            dists_below = _get_signed_distances(xs[0], windows, rmap)
            bound0 = np.searchsorted(dists_below, -max_dist)
            dists_above = _get_signed_distances(xs[-1], windows, rmap)
            bound1 = np.searchsorted(dists_above, max_dist)

            if Bmap is None:
                bvals = splines[(uL, s)](distances[bound0:bound1])
            else:
                s_windows = s * B_windows[bound0:bound1]
                bvals = _get_interference_bvals(
                    s_windows, distances[bound0:bound1], splines)

            for j, Us in enumerate(U_arrs):
                Bs[j][i] = np.prod(bvals ** Us[bound0:bound1, None], axis=0)

        # Integrate across Bs
        if dfes is not None:
            assert len(dfes) == len(U_arrs)
            Bs_out = np.prod([Util.integrate_with_dfe(Bs_dfe, ss, dfe)
                            for Bs_dfe, dfe in zip(Bs, dfes)], axis=0)

        # If no DFEs were given, return full Bs array
        else:
            Bs_out = Bs

    return Bs_out


def _get_s_indices(s_vals, s_grid):
    """

    """
    idx_1 = np.searchsorted(s_grid, s_vals)
    idx_0 = idx_1 - 1
    return idx_0, idx_1


def _get_s_facs(s_vals, s_grid, idx_0, idx_1):

    grid_vals_1 = s_grid[idx_1]
    grid_vals_0 = s_grid[idx_0]
    facs_0 = (grid_vals_1 - s_vals) / (grid_vals_1 - grid_vals_0)
    facs_1 = 1 - facs_0
    return facs_0, facs_1


def _get_interference_bvals(s_windows, distances, splines):
    """
    Computes unit B-values under the interference correction, for a particular
    s-coefficient.

    :param s_windows: Window selection coefficients; these are B*s, where B is
        the local B-value from the last round of interference correction, and
        s is a scalar selection coefficient.
    :param distances: A precomputed matrix of distances between focal sites
        and windows, with shape (len(windows), len(xs)), where xs is an array
        of focal sites.
    :param splines: Splines for interpolating B-values.
    """
    uL = next(iter(splines.keys()))[0]
    s_grid = np.sort([key[1] for key in splines])
    idx_0, idx_1 = _get_s_indices(s_windows, s_grid)
    facs_0, facs_1 = _get_s_facs(s_windows, s_grid, idx_0, idx_1)
    req_grid_idxs = np.unique([idx_0, idx_1])
    ret = np.zeros(distances.shape)
    for idx in req_grid_idxs:
        # get the spline for this s
        spline =  splines[(uL, s_grid[idx])]
        mask = idx_0 == idx
        ret[mask] += facs_0[mask, None] * spline(distances[mask])
        mask = idx_1 == idx
        ret[mask] += facs_1[mask, None] * spline(distances[mask])
    return ret


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


def _get_distances(xs, windows, rmap):
    """
    Computes map distances between sites `xs` and the centers of `windows`.
    """
    return np.abs(_get_signed_distances(xs, windows, rmap))


def _get_signed_distances(xs, windows, rmap):
    """
    Computes signed map distances between `xs` and `windows`.
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
    Adjusts local deleterious mutation rate arrays recorded in a list `U_arrs`
    by local B-values, computed using `windows` and `Bmap`.
    """
    window_Bs = _get_B_per_element(Bmap, windows)
    adjusted_U_arrs = [Us * window_Bs for Us in U_arrs]
    return adjusted_U_arrs
