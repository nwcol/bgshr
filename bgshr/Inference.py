
import numpy as np
from scipy import stats
import warnings
import gzip
import re

from . import Util, Predict


_data_cache = {}
_ll_cache = {}


def objective_func(
    params,
    xs,
    df,
    ndns,
    mask,
    u_map,
    elements,
    U_windows,
    U_arrs,
    r_map,
    dfes,
    n_corrs,
    chunk_size,
    n_cores,
    B_unlinked
):
    """
    Objective function for Ne optimization.
    """
    Ne = params[0]

    if Ne in _ll_cache:
        return -_ll_cache[Ne]

    # Check geometry
    L = len(u_map)
    nd, ns = ndns
    assert len(nd) == L

    # Send lookup table to target Ne
    df = rebuild_lookup_table(df, Ne)
    _, __, splines = Util.generate_linear_splines(df)
    max_dists = Util.get_max_distances(df)

    # Predict B-values
    interf_Bs = Predict.interference_Bvals(
        xs,
        splines,
        windows=windows,
        U_arrs=U_arrs,
        rmap=rmap,
        max_dists=max_dists,
        dfes=dfes,
        chunk_size=chunk_size,
        n_cores=n_cores,
        B_unlinked=B_unlinked,
        verbose=verbose
    )

    # Get expected pi
    Bmap = Predict.get_Bmap(xs, Bs)
    site_B = Bmap(np.arange(L))
    exp_pi0 = expected_pi0(u_map, df, elements=elements, dfes=dfes)
    exp_pi = expected_pi(exp_pi0, site_B, mask=mask)

    # Compute likelihood and update caches
    ll = bgshr.Inference.ll(nd, ns, exp_pi)
    _cache[Ne] = ll
    _data_cache[Ne] = (interf_Bs, exp_pi)
    if verbose:
        pass
    return -ll


def rebuild_lookup_table(df, Ne):
    """
    Copies a lookup table and scales it to the appropriate Ne.
    """
    df = Util.cap_max_lookup_table_B(df)
    df = Util.scale_lookup_table(df, Ne)
    df = Util.fill_in_lookup_table(df)
    ss_extend = -np.logspace(0, np.log10(-np.min(df["s"])), 17)[:-1]
    df = ClassicBGS.extend_lookup_table(df, ss_extend)
    df = Util.convert_lookup_table_to_morgans(df)
    return df


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
