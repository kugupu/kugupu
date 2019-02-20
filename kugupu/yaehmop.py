"""Controls running and parsing yaehmop tight binding calculations

"""
from collections import Counter, namedtuple
from scipy import sparse
import numpy as np
import subprocess
import os
from tqdm import tqdm
from MDAnalysis.lib import distances
import warnings

import yaehmop

from . import logger

# contains information on the number of orbitals and number of valence
# electrons in all fragments
FragSizes = namedtuple('FragSizes', ['n_orbitals', 'n_electrons'])


def shift_dimer_images(frag_i, frag_j):
    """Determine positions that place frag_j next to frag_i

    Returns
    -------
    pos : numpy ndarray
      concatenated positions of frag_i and frag_j
    """
    logger.debug("Checking if fragments are in correct image")
    c_i = frag_i.center_of_mass()
    c_j = frag_j.center_of_mass()

    tol = 0.001
    d1 = distances.calc_bonds(c_i, c_j)
    d2 = distances.calc_bonds(c_i, c_j, frag_i.dimensions)
    if not abs(d1 - d2) < tol:
        logger.debug("Shifting fragment")
        shift = (c_i - c_j) / frag_i.dimensions[:3]

        pos_j = frag_j.positions + (np.rint(shift) * frag_i.dimensions[:3])
    else:
        pos_j = frag_j.positions

    return np.concatenate([frag_i.positions, pos_j])


ORBITALS = {
    'H': 1,
    'C': 4,
    'S': 4,
    'O': 4,
    'N': 4,
}
ELECTRONS = {
    'C': 4,
    'H': 1,
    'S': 6,
    'O': 6,
    'N': 5,
}
def count_orbitals(ag):
    """Count the number of orbitals and valence electrons in an AG

    Parameters
    ----------
    ag : AtomGroup
      AtomGroup to count orbitals and electrons for
    orbitals, electrons : dict
      mapping of element to number of orbitals/electrons

    Returns
    -------
    norbitals, nelectrons : int
      total number of orbitals and valence electrons in AtomGroup
    """
    # number of each element
    count = Counter(ag.names)
    # number of orbitals in fragment
    norbitals = sum(n_e * ORBITALS[e] for e, n_e in count.items())
    # number of valence electrons in fragment
    nelectrons = sum(n_e * ELECTRONS[e] for e, n_e in count.items())

    return norbitals, nelectrons


def run_single_fragment(ag, index):
    """Create input run YAEHMOP and parse output for single fragmnet

    Parameters
    ----------
    ag : mda.AtomGroup
      single fragment to run
    index : int
      identifier for this fragment

    Returns
    -------
    H_frag, S_frag : dict of sparse.csr_matrix
      dict of contributions for different (i,j) pairs
    orb, ele : dicts
      fragment index to orbital size and number of valence electrons
    """
    logger.debug('Calculating lone fragment {}'.format(index))
    base = '{}.bind'.format(index)

    H_mat, S_mat = yaehmop.run_bind(
        ag.positions.astype(np.float64), ag.names, 0.0)
    norbitals, nelectrons = count_orbitals(ag)

    return H_mat, S_mat, norbitals, nelectrons


def run_single_dimer(ags, indices, keep_i, keep_j):
    """Push a dimer into yaehmop tight bind

    Parameters
    ----------
    ags : tuple of mda.AtomGroup
      The dimer to run
    indices : tuple of int
      the indices of this dimer pairing, eg (4, 5)
    keep_i, keep_j : bool
      whether to keep the self contribution for fragments i & j

    Returns
    -------
    H_frag, S_frag : dict of sparse.csr_matrix/numpy.array
      dict of contributions for different (i,j) pairs
    orb, ele : dicts
      fragment index to orbital size and number of valence electrons
    """
    logger.debug('Calculating dimer {}'.format(indices))
    i, j = indices
    ag_i, ag_j = ags
    pos = shift_dimer_images(ag_i, ag_j)

    H_mat, S_mat = yaehmop.run_bind(
        pos.astype(np.float64), (ag_i + ag_j).names, 0.0)

    orb_i, ele_i = count_orbitals(ag_i)
    orb_j, ele_j = count_orbitals(ag_j)

    H_mats = {}
    S_mats = {}
    H_mats[i, j] = sparse.csr_matrix(H_mat[:orb_i, orb_i:])
    if keep_i:
        H_mats[i, i] = H_mat[:orb_i, :orb_i]
        S_mats[i, i] = S_mat[:orb_i, :orb_i]
    if keep_j:
        H_mats[j, j] = H_mat[orb_i:, orb_i:]
        S_mats[j, j] = S_mat[orb_i:, orb_i:]

    return H_mats, S_mats, (orb_i, orb_j), (ele_i, ele_j)


def run_all_dimers(fragments, dimers):
    """Run all yaehmop calculations for dimer pairs

    Parameters
    ----------
    fragments : list of MDA.AtomGroup
      list of fragments
    dimers : dict
      mapping of dimer tuple to dimer AtomGroups (two fragments)

    Returns
    -------
    H_orb - dict of csr sparse (off diagonal) or numpy arrays (self) representing
            Hamiltonian matrices
    S_orb - dict of numpy arrays of overlap matrix
    fragsize - namedtuple with attributes 'starts', 'stops', 'sizes' and
          'n_electrons' which can be indexed using fragment ids (0..nfrags-1)
    """
    logger.info('Starting yaehmop calculation for {} dimers'.format(len(dimers)))
    # keep track of which fragment ids we've seen once
    done = set()
    # coordinate representation of sparse matrices
    # ie H_coords[1, 2] gives dimer(1-2) sparse matrix
    H_coords = dict()
    S_coords = dict()
    # values per fragment id
    fragsize = FragSizes(
        np.zeros(len(fragments), dtype=int),
        np.zeros(len(fragments), dtype=int),
    )

    for (i, j), ags in tqdm(sorted(dimers.items())):
        # only keep each self contribution once
        keep_i = not i in done
        keep_j = not j in done
        done.add(i)
        done.add(j)

        H_frag, S_frag, orb, ele = run_single_dimer(
            ags, (i, j), keep_i, keep_j)
        H_coords.update(H_frag)
        S_coords.update(S_frag)
        for o, e, x in zip(orb, ele, (i, j)):
            fragsize.n_orbitals[x] = o
            fragsize.n_electrons[x] = e

    # for each fragment that wasn't in a dimer pairing
    for i in set(range(len(fragments))) - done:
        H_frag, S_frag, orb, ele = run_single_fragment(fragments[i], i)
        H_coords[i, i] = H_frag
        S_coords[i, i] = S_frag
        fragsize.n_orbitals[i] = orb
        fragsize.n_electrons[i] = ele

    logger.info('Done with yaehmop')
    return H_coords, S_coords, fragsize
