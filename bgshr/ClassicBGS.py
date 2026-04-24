
import numpy as np
import pandas
from scipy import linalg
import warnings

from . import Util


def extend_lookup_table(df_sub, ss, generation=0):
    """
    Wraps CBGS extension functions for equilibrium/n-epoch lookup tables.

    :param df_sub: Lookup table to extend
    :param ss: Selection coefficients with which to extend the table
    """
    all_Ts = set(df_sub["Ts"])
    all_Ns = set(df_sub["Ns"])
    assert len(all_Ts) == 1
    assert len(all_Ns) == 1
    Ts = next(iter(all_Ts))
    Ns = next(iter(all_Ns))

    rs = np.unique(df_sub["r"])
    uL = np.unique(df_sub["uL"])[0]
    uR = np.unique(df_sub["uR"])[0]

    # 1-epoch or equilibrium tables
    if len(str(Ts).split(";")) == 1:
        df_new = build_lookup_table(ss, rs, Ne=Ns, uL=uL, uR=uR)

    # n-epoch tables
    else:
        Ts = np.array([int(float(x)) for x in Ts.split(";")])
        Ns = np.array([int(float(x)) for x in Ns.split(";")])
        df_new = build_lookup_table_n_epoch(
            ss, rs, Ns, Ts, generations=[generation], uL=uL, uR=uR)

    df_comb = pandas.concat([df_sub, df_new], ignore_index=True)
    return df_comb


def reduction_CBGS(s, u, r, L=1):
    """
    Compute a diversity reduction with classic BGS theory. Valid for `s`
    where 2N_e*s >> 1.

    Derived from Charlesworth (2012) [Appendix], without making the assumption
    that s and r are small.

    :param s: Selection coefficient. Heterozygotes with the deleterious allele
        have fitness (1+s). Should be < 0.
    :param u: Deleterious haploid mutation rate.
    :param r: Recombination rate between focal site and constrained locus.
    :param L: Optional scaling factor for the mutation rate. Assumes a non-
        recombining locus of length `L` and average per-base mutation rate `u`.
    """
    return np.exp(s * u * (1 + (2 * r * (1+s) - s) ** 2) / (r * (1+s) - s) ** 2)


def unlinked_reduction_CBGS(s, u, L=1):
    """
    Compute a diversity reduction due to an unlinked locus, using classic BGS
    theory.

    Given by Charlesworth (2012) [Appendix].

    :param s: Selection coefficient. Heterozygotes with the deleterious allele
        have fitness (1+s). Should be < 0.
    :param u: Deleterious haploid mutation rate.
    :param L: Optional scaling factor for the mutation rate. Assumes a non-
        recombining locus of length `L` and average per-base mutation rate `u`.
    """
    return np.exp(8 * s * u * L / (1 - s) ** 2)


def approx_reduction_CBGS(s, u, r, L=1):
    """
    Compute diversity reduction with classic BGS theory; valid for small `r`.
    This function should not be used in prediction.

    Given by Nordborg and Charlesworth (1996) using diffusion and Nordborg
    (1997) using a Markov chain approach.

    :param s: Selection coefficient for the heterozygote.
    :param u: The deleterious haploid mutation rate.
    :param r: The recombination rate between the focal site and the selected
        locus.
    :param L: Optionally, a scaling factor for the mutation rate, assuming a
        non-recombining selected locus of length L, and per-base mutation rate
        of `u`.
    """
    return np.exp(-u * L / (-s * (1 + r * (1 + s) / -s) ** 2))


def classic_BGS(xs, s, u, L=None, rmap=None, elements=[]):
    """
    Compute classic BGS reduction due to selection in constrained elements.

    B values are computed from the midpoints of elements. If elements are
    large, consider splitting into smaller regions using `Util.xzy()`.

    THis assumes that we are at steady state.

    :param xs: An arary of positions to compute B-values at.
    :param s: The selection coefficient for each deleterious mutation.
    :param u: Optional, the length of the sequence. If it is not given, we set
        it to the largest value found in `xs` or the end of the final element,
        whichever is larger.
    :rmap: An interpolated recombination map funcion. If a recombination map
        is not provided, we assume the region is non-recombining.
    :param elements: A (sorted) list of non-overlapping [l, r] regions defining
        selected elements.
    """
    B = np.ones(len(xs))
    if len(elements) == 0:
        return B

    if L is None:
        L = max([xs[-1], elements[-1][1]])

    if rmap is None:
        r = 0
        rmap = Util.build_uniform_rmap(r, L)

    r_xs = rmap(xs)
    for e in elements:
        mid = np.mean(e)
        L_elem = e[1] - e[0]
        r_mid = rmap(mid)
        r_dists = np.abs(r_xs - r_mid)
        B *= reduction_CBGS(s, u, r_dists, L=L_elem)
    return B


def extend_lookup_table_1_epoch(df_sub, ss, generation=0):
    """
    Extend a lookup table at present recombination values for given s values.
    """
    r_vals = np.array(sorted(list(set(df_sub["r"]))))
    cols = df_sub.columns
    data = {
        "Ns": np.unique(list(set(df_sub["Ns"])))[0],
        "Ts": np.unique(df_sub["Ts"])[0],
        "uL": np.unique(df_sub["uL"])[0],
        "uR": np.unique(df_sub["uR"])[0],
        "Order": 0,
        "Generation": generation,
        "pi0": np.unique(df_sub["pi0"])[0]
    }

    new_data = []
    for s in ss:
        Bs = reduction_CBGS(s, data["uL"], r_vals)
        data["s"] = s
    
        Nanc = []
        if type(data["Ns"]) is str:
            Nvec = np.unique(np.array(data["Ns"]))[0].split(";")
            Nanc = Nvec[len(Nvec)-1]
        else: # eq. demography and single Ns has been converted
            Nanc = data["Ns"]
            
        data["Hl"] = _get_Hl(s, Nanc, np.unique(data["uL"])[0])
        Hrs = Bs * data["pi0"]
        data["piN_pi0"] = data["Hl"] / data["pi0"]
        for r, B, Hr in zip(r_vals, Bs, Hrs):
            data["r"] = r
            data["B"] = B
            data["Hr"] = Hr
            data["piN_piS"] = data["Hl"] / data["Hr"]
            new_row = [data[k] for k in df_sub.columns]
            new_data.append(new_row)
    df_new = pandas.DataFrame(new_data, columns=df_sub.columns)
    df_comb = pandas.concat((df_sub, df_new), ignore_index=True)
    return df_comb


def build_lookup_table(ss, rs, Ne=1e4, uL=1e-8, uR=1e-8):
    """
    Given a list of selection coefficients and recombination rates, build a
    a diversity-reduction lookup table using just classic background
    selected theory.
    """
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
        "Ns": Ne,
        "Ts": 0,
        "uL": uL,
        "uR": uR,
        "Order": 0,
        "Generation": 0,
    }

    data["pi0"] = 2 * Ne * uR

    new_data = []
    for s in ss:
        if s == 0:
            Bs = np.ones(len(rs))
        else:
            Bs = reduction_CBGS(s, uL, rs)
        data["s"] = s
        data["Hl"] = _get_Hl(s, Ne, uL)
        data["piN_pi0"] = data["Hl"] / data["pi0"]
        Hrs = Bs * data["pi0"]
        for r, B, Hr in zip(rs, Bs, Hrs):
            data["r"] = r
            data["B"] = B
            data["Hr"] = Hr
            data["piN_piS"] = data["Hl"] / data["Hr"]
            new_row = [data[k] for k in cols]
            new_data.append(new_row)
    df_new = pandas.DataFrame(new_data, columns=cols)
    return df_new


def _get_Hl(s, Ne, u):
    """
    From integrating Eq. 31 in Evans et al (2007) against `4*Ne*u*x*(1-x)`.

    Note that here, Hl is defined as 2Nu under neutrality, rather than
    the typical 4Nu.
    """
    if s == 0:
        return 2 * Ne * u
    else:
        return 4 * Ne * u * np.exp(4 * Ne * s) / (np.exp(4 * Ne * s) - 1) - u / s


def unlinked_CBGS(U, dfe, ss=None, grid_size=500):
    """
    Computes an unlinked B-value using Classic BGS theory, for a single DFE.

    :param U: Total deleterious mutation rate of unlinked, constrained
        elements. Assuming an average rate `u` and constrained sequence length
        `L`, this is `u * L`.
    :param dfe: Dictionary defining DFE type and parameters. See
        `Util.get_dfe_weights` for specification.
    :param ss: Optional grid of selection coefficients to integrate across.
        If None (default), a log-spaced grid from -1 to -10^-6 with `grid_size`
        steps is used.
    :param grid_size: Optional number of steps in the `s` grid, to be used if
        `ss` is None (default 500).

    :returns: Scalar unlinked B-value
    """
    if ss is None:
        ss = np.concatenate([-np.logspace(0, -6, grid_size - 1), [0]])
    else:
        # ss must begin at -1 and increase monotonically
        assert np.all(np.diff(ss) > 0)
        assert ss[0] == -1 and ss[-1] == 0

    weights = Util.get_dfe_weights(ss, dfe)
    unlinked_Bs = unlinked_reduction_CBGS(ss[:-1], U)
    unlinked_B = Util.integrate_with_weights(unlinked_Bs, weights[:-1])
    return unlinked_B


###################################################################
# Functions to compute classic BGS predictions with size changes. #
###################################################################


def _sub_intensity_matrix(N_epoch, s, u, r, Ne):
    """
    Given in Nordborg (1997), this is the sub-intensity matrix for the
    structured coalescence model with background selection. It has been
    adjusted to allow for different population size in a given epoch
    from the Ne used to scale time.
    """
    q = u / -s
    p = 1 - q
    b12 = q * r
    b21 = p * (-s + r)
    nu = N_epoch / Ne  # size relative to Ne
    S = np.array(
        [
            [-4 * Ne * b12 - 1 / (nu * p), 4 * Ne * b12, 0],
            [2 * Ne * b21, -2 * Ne * b12 - 2 * Ne * b21, 2 * Ne * b12],
            [0, 4 * Ne * b21, -4 * Ne * b21 - 1 / (nu * q)],
        ]
    )
    return S


def _probability_absorption(N_epoch, s, u, r, Ne, gens):
    """
    Time is measured in units of 2Ne generations. This returns the probability
    of coalescence in the structured coalescent model for BGS in an epoch of a
    given size.
    """
    t = gens / 2 / Ne
    S = _sub_intensity_matrix(N_epoch, s, u, r, Ne)
    q = u / -s
    p = 1 - q
    alpha = np.array([p ** 2, 2 * p * q, q ** 2])
    return 1 - alpha.dot(linalg.expm(S * t)).dot([1, 1, 1])


def _bgs_coalescent_rate(s, u, r, nu):
    """
    The parameter lambda in Nordborg (1997). The parameter `nu` is the relative
    size of the epoch compared to a reference Ne.
    """
    q = u / -s
    p = 1 - q
    b12 = q * r
    b21 = p * (-s + r)
    return b21 ** 2 / (b12 + b21) ** 2 / p / nu + b12 ** 2 / (b12 + b21) ** 2 / q / nu


def expected_tmrca_n_epoch_neutral(Ns, Ts):
    """
    Ns and Ts are vectors of the same length, specifying (piecewise constant)
    population sizes and epoch break points. `Ns` and `Ts` must have the same
    length, and time is measured in generations into the past. The first entry
    in `Ts` should be zero, and values should be monotonically increasing and
    not include infinity.  We assume the last entry in `Ns` specifies the
    steady-state population size prior to any size changes in the past.
    """
    p_coal = 0
    ET = 0
    for N, T0, T1 in zip(Ns, Ts[:-1], Ts[1:]):
        gens = T1 - T0
        if gens < 0:
            raise ValueError("Ts are not monotonically increasing")
        if gens == 0:
            continue
        # probability of coalescence within this epoch
        p_coal_epoch = 1 - np.exp(-gens / 2 / N)
        # weighted by the probability that we haven't coalescece before this epoch
        epoch_weight = p_coal_epoch * (1 - p_coal)
        p_coal += epoch_weight
        # expected TMRCA conditional on coalescing within this epoch
        ET_cond = 2 * N - gens * (1 - p_coal_epoch) / p_coal_epoch
        # add to ET
        ET += epoch_weight * (T0 + ET_cond)
    # final epoch extending to infinity
    assert p_coal < 1
    ET += (1 - p_coal) * (Ts[-1] + 2 * Ns[-1])
    return ET


def expected_tmrca_n_epoch_bgs(Ns, Ts, s, u, r):
    """
    Given a piecewise-constant population size history, get the expected TMRCA,
    in units of Ne generations. Ne is taken to be the size from the most
    ancient epoch. The expected TMRCA under classic BGS is an extension of the
    result from Nordborg (1997), conditioning on coalescence occurring within a
    given epoch, and then weighting total contribution to TMRCA across
    probability of coalescence occurring within each epoch.

    Ns and Ts must be the same length, and Ts must start at zero and be
    monotonically increasing. Then the size between Ts[0] and Ts[1] is
    Ns[0], the size between Ts[1] and Ts[2] is Ns[1], and so on. The final
    epoch estends from Ts[-1] to infinity, and has size Ns[-1] (== Ne).
    """
    if len(Ns) != len(Ts):
        raise ValueError("Ns and Ts must be the same length")
    p_coal = 0
    ET = 0
    Ne = Ns[-1]
    for N, T0, T1 in zip(Ns, Ts[:-1], Ts[1:]):
        gens = T1 - T0
        if gens < 0:
            raise ValueError("Ts are not monotonically increasing")
        if gens == 0:
            continue
        # probability of coalescence within this epoch
        p_coal_epoch = _probability_absorption(N, s, u, r, Ne, gens)
        # weighted by the probability that we haven't coalescece before this epoch
        epoch_weight = p_coal_epoch * (1 - p_coal)
        p_coal += epoch_weight
        # expected TMRCA conditional on coalescing within this epoch
        ET_cond = (
            2 * Ne / _bgs_coalescent_rate(s, u, r, N / Ne)
            - gens * (1 - p_coal_epoch) / p_coal_epoch
        )
        # add to ET
        ET += epoch_weight * (T0 + ET_cond)
    # final epoch extending to infinity
    assert p_coal < 1
    ET += (1 - p_coal) * (Ts[-1] + 2 * Ne / _bgs_coalescent_rate(s, u, r, 1))
    return ET


def reduction_CBGS_n_epoch(Ns, Ts, s, u, r, L=1, scale_mutation=True):
    """
    Expected B-value (diversity reduction) for a size change history under
    classical BGS theory.

    Ns and Ts must be the same length, and Ts must start at zero and be
    monotonically increasing. Then the size between Ts[0] and Ts[1] is
    Ns[0], the size between Ts[1] and Ts[2] is Ns[1], and so on. The final
    epoch estends from Ts[-1] to infinity, and has size Ns[-1] (== Ne).

    This method can fail when u is small compared to 1/N, especially for large
    recombination rates. Instead, with scale_mutation=True, we rescale u to be
    larger (Ne*u ~ O(1)), while ensuring that still u<<s. Then we scale the
    B value back to the original u value.

    """
    if -s <= 1 / np.min(Ns):
        warnings.warn(
            "N*s is 1 or smaller for some epochs - expect results to be wrong"
        )
    if scale_mutation:
        # scale mutation to be equal to the ancestral Ne, which should be
        # reasonable if population sizes do not fluctuate dramatically
        u_scale = 1 / np.mean(Ns[-1])
        scale_fac = u_scale / u
        TBGS = expected_tmrca_n_epoch_bgs(Ns, Ts, s, u_scale, r)
    else:
        TBGS = expected_tmrca_n_epoch_bgs(Ns, Ts, s, u, r)
    Tneu = expected_tmrca_n_epoch_neutral(Ns, Ts)
    if TBGS < 0:
        warnings.warn(
            f"BGS calculation is nonsense - s: {s}, u: {u}, r: {r}, Ns: {Ns}, Ts: {Ts}"
        )
    B = TBGS / Tneu
    if B > 1:
        # reduction cannot be larger than 1, which can
        # occur with weaker selection, large r
        B = 1
    if scale_mutation:
        B = B ** (1 / scale_fac)
    return B ** L


def _shift_Ns_Ts(Ns, Ts, gen):
    Ns_gen = []
    Ts_gen = []
    for i, (N, T0, T1) in enumerate(zip(Ns[:-1], Ts[:-1], Ts[1:])):
        if T0 >= gen:
            Ns_gen.append(N)
            Ts_gen.append(T0 - gen)
        elif T1 > gen:
            Ns_gen.append(N)
            Ts_gen.append(0)
    Ns_gen.append(Ns[-1])
    Ts_gen.append(max(Ts[-1] - gen, 0))
    return Ns_gen, Ts_gen


def build_lookup_table_n_epoch(
    ss,
    rs,
    Ns,
    Ts,
    generations=None,
    uL=1e-8,
    uR=1e-8
):
    # here Ns and Ts are numeric vector, not semi-colon separated vals on a string
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
        "Ns",  # list of population sizs from present to past
        "Ts",  # list of epoch time points from 0 to past
    ]
    if len(Ns) != len(Ts):
        raise ValueError("Ns and Ts must be the same length")
    if Ts[0] != 0:
        raise ValueError("Ts must start at time zero (present time)")

    Ne = Ns[-1]
    if generations is None:
        generations = [0] # TODO check

    Nstring = ";".join([str(N) for N in Ns])
    Tstring = ";".join([str(T) for T in Ts])
    data = {
        "Ns": Nstring,
        "Ts": Tstring,
        "uL": uL,
        "uR": uR,
        "Order": 0,
    }
    new_data = []
    for gen in generations:
        Ns_gen, Ts_gen = _shift_Ns_Ts(Ns, Ts, gen) # TODO check
        data["Generation"] = gen
        # fill in data
        data["pi0"] = expected_tmrca_n_epoch_neutral(Ns_gen, Ts_gen) * uL
        for s in ss:
            data["s"] = s
            # Assume negligible change under size history, if strong enough selection
            data["Hl"] = _get_Hl(s, Ns[0], uL)
            data["piN_pi0"] = data["Hl"] / data["pi0"]
            for r in rs:
                if s == 0:
                    data["B"] = 1
                else:
                    data["B"] = reduction_CBGS_n_epoch(Ns_gen, Ts_gen, s, uL, r)
                data["Hr"] = data["B"] * data["pi0"]
                data["r"] = r
                data["piN_piS"] = data["Hl"] / data["Hr"]
                new_row = [data[k] for k in cols]
                new_data.append(new_row)
    df_new = pandas.DataFrame(new_data, columns=cols)
    return df_new
