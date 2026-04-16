
import numpy as np
from scipy import stats
import warnings
import gzip
import re

from . import Util


def num_diff_same(counts):
    nT = sum(counts) * (sum(counts) - 1) // 2
    if len(counts) == 1:
        return [0, nT]
    else:
        nS = sum([c * (c - 1) // 2 for c in counts])
        nD = nT - nS
        return [nD, nS]


def parse_vcf(vcf_fname, L=None, samples=None):
    """
    Get nD and nS for all sites in chromosome.

    We assume that any site missing from the VCF is invariant.

    NOTE: at some point we'll need to deal with zero- vs one-based indexing.
    """
    if samples is None:
        warnings.warn("computing nD and nS for all samples in VCF")
    else:
        raise ValueError("need to implement with samples")

    if L is None:
        raise ValueError("Need to provide the length of the chromosome")

    if not vcf_fname.endswith("gz"):
        raise ValueError("Expected gzipped VCF file")

    with gzip.open(vcf_fname, "rb") as fin:
        for line in fin:
            l = line.decode()
            if l.startswith("#"):
                if l.startswith("#CHROM"):
                    n_samples = len(line.decode().split()[9:])
                    nD = np.zeros(L, dtype=int)
                    nS = np.ones(L, dtype=int) * num_diff_same([2 * n_samples])[1]
            else:
                # maybe faster
                gts = [re.split("/|\\||:", g) for g in l.split()[9:]]
                counts = np.unique(
                    np.reshape(gts, (2 * n_samples,)), return_counts=True
                )[1]
                pos = int(l.split()[1])
                nD[pos], nS[pos] = num_diff_same(counts)
    return nD, nS


def ll_per_site(nD, nS, Epi):

    """
    Following BK24, nD is the number of differences (heterozygotes) between all
    pairwise comparisons at a site, nS is the number of homozygotes, and Epi is
    the expected diversity at each site.

    These are numpy arrays that must all be the same length. The log-likelihood
    is then `log(pi) * nD + log(1 - pi) * nS`.
    """
    return np.log(Epi) * nD + np.log(1 - Epi) * nS


def ll(nD, nS, Epi):
    """
    Given arrays of number of differences, homozygous comparisons, and expected
    pi per-site, return the composite log-likelihood over the entire sequence.
    """
    ll_arr = ll_per_site(nD, nS, Epi)
    return ll_arr.sum()


def expected_pi(pi0, B, mask=None):
    """
    Given a pi0 value or array of values, multiply by the per-site diversity
    reduction to get expected pi after linked selection.

    If mask is given, it is a boolean array with same length as B, with 1/True
    at sites that should be masked and excluded from likelihood calculation,
    0/False for sites that should be included.
    """
    if not np.isscalar(pi0):
        if len(pi0) != len(B):
            raise ValueError("pi0 and B must be the same length")
    if mask is not None:
        if len(mask) != len(B):
            raise ValueError("mask and B must be the same length")
    else:
        mask = False
    return np.ma.masked_array(pi0 * B, mask=mask)


def expected_pi0(u, df, L=None, elements=[], dfes=[]):
    """
    Get expected pi0, given mutation rate and any elements under selection. The
    mutation rate can be a single scalar value valid across the entire region,
    or an array of per-base pair mutation rates. The elements are lists of
    intervals (a list of lists, with half open intervals [left, right) defined
    within). The dfes correspond to those elements.

    The DFEs are defined as a dictionary, specifying the DFE type and any
    parameters associated with that DFE. For example, a gamma DFE is defined as
    `{"type": "gamma", "shape": shape, "scale": scale}`. A gamma DFE with a
    proportion of sites being neutral (e.g., gamma for nonsynonymous and
    neutral for synonymous mutations) would be `{"type": "gamma_neu", "shape":
    shape, "scale": scale, "p_neu": 1 / (2.31 + 1)}`, or whatever value `p_neu`
    should be.

    Elements should not overlap, since overlapping elements will have values
    set by the last-seen element in this function.
    """
    if np.isscalar(u):
        if L is None:
            raise ValueError("L must be provided if u is a scalar value")
        u_arr = u * np.ones(L)
    else:
        if L is not None and len(u) != L:
            raise ValueError("L does not equal length of u")
        u_arr = 1 * u
    if len(elements) != len(dfes):
        raise ValueError("length of dfes does not match length of element sets")
    if len(set(df["uL"])) != 1:
        raise ValueError("only a single uL value in the lookup table is allowed")

    # neutral diversity
    uL = (df[(df["s"] == 0) & (df["r"] == 0)]["uL"]).iloc[0]
    pi0 = 2 * df[(df["s"] == 0) & (df["r"] == 0)]["Hl"].iloc[0] * u_arr / uL

    for elems, dfe in zip(elements, dfes):
        # get diversity for uL
        pi_dfe = _get_pi_dfe(df, dfe)
        # scale by u_arr
        pi_arr = pi_dfe * u_arr / uL
        # fill in pi0 for each element
        for e in elems:
            pi0[e[0] : e[1]] = pi_arr[e[0] : e[1]]

    return pi0


def _get_pi_dfe(df, dfe):
    df_sub = df[df["r"] == 0]
    ss = np.sort(df_sub["s"])
    assert ss[-1] == 0
    weights = Util.get_dfe_weights(ss, dfe)
    Hls = 2 * np.array([df_sub[df_sub["s"] == s]["Hl"].iloc[0] for s in ss])
    return np.sum(Hls * weights)


def _get_gamma_weights(ss, shape, scale):
    weights = np.concatenate(
        (
            Util.weights_gamma_dfe(ss[:-1], shape, scale),
            [stats.gamma.cdf(-ss[-2], shape, scale=scale)],
        )
    )
    return weights


def _get_gamma_neutral_weights(ss, shape, scale, p_neu):
    weights = np.concatenate(
        (
            (1 - p_neu) * Util.weights_gamma_dfe(ss[:-1], shape, scale),
            [p_neu + (1 - p_neu) * stats.gamma.cdf(-ss[-2], shape, scale=scale)],
        )
    )
    return weights


def load_mask(mask_fname, L=None):
    """
    A mask can be provided as a fasta file or a bed file. File type is checked
    by the file-type extension. From a fasta file, we keep sites marked P.

    These files must be gzipped, and should have the extension `fasta.ga` or
    `bed.gz`. If a bed file is provided, we need to also pass the sequence
    length `L`, which should not be specified if a fasta file is provided.

    Return a boolean array, with 1s for sites that should be masked and
    excluded from any likelihood calculation, and 0s for sitest that should be
    included, e.g., sites that pass some callability threshold.

    TODO: check zero vs one indexed
    """
    if mask_fname.endswith("fasta.gz"):
        if L is not None:
            raise ValueError("sequence length cannot be provided with a fasta file")
        raise ValueError("haven't implemented mask parsing for fasta files yet")
    elif mask_fname.endswith("bed.gz"):
        if L is None:
            raise ValueError("sequence length must be provided with a bed file")
        mask = np.ones(L, dtype=bool)
        with gzip.open(mask_fname, "rb") as fin:
            for line in fin:
                ld = line.decode()
                _, l, r = ld.split()
                mask[int(l) : int(r)] = 0
    else:
        raise ValueError("file type not recognized")
    return mask
