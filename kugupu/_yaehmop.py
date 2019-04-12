"""Controls running yaehmop tight binding calculations

"""
from collections import Counter
import numpy as np
from MDAnalysis.lib import distances
import yaehmop  # pyyaehmop bindings

from . import logger


def shift_dimer_images(frag_i, frag_j):
    """Determine positions that place frag_j next to frag_i

    Important as yaehmop doesn't do periodic boundaries,
    but our distance search does!

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
    # does the distance between the two molecules changes if we use
    # periodic boundaries?
    if not abs(d1 - d2) < tol:
        logger.debug("Shifting fragment")
        # calculate number of box images to shift
        shift = (c_i - c_j) / frag_i.dimensions[:3]

        pos_j = frag_j.positions + (np.rint(shift) * frag_i.dimensions[:3])
    else:
        # else just take positions as-is
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


def run_fragment(ag):
    """Run tight binding on single fragment

    Parameters
    ----------
    ag : mda.AtomGroup
      single fragment to run

    Returns
    -------
    H_mat, S_mat : numpy array
      Hamiltonian and Overlap matrices
    nelectrons : int
      number of valence electrons
    """
    H_mat, S_mat = yaehmop.run_bind(ag.positions, ag.names, 0.0)
    _, nelectrons = count_orbitals(ag)

    return H_mat, S_mat, nelectrons


def run_dimer(ags):
    """Tight binding calculation on pair of fragments

    The positions of the pair will be shifted according
    to periodic boundaries to choose the closest image.

    Parameters
    ----------
    ags : tuple of mda.AtomGroup
      The dimer to run

    Returns
    -------
    Hij : numpy array of off diagonal (intermolecular) Hamiltonian
    (Hii, Sii, ele_i) : intramolecular Hamiltonian and overlap matrix
                        and number of electrons for first fragment
    (Hjj, Sjj, ele_j) : same for second fragment
    """
    ag_i, ag_j = ags
    pos = shift_dimer_images(ag_i, ag_j)

    logger.debug('Running bind')
    H_mat, S_mat = yaehmop.run_bind(pos, (ag_i + ag_j).names, 0.0)

    orb_i, ele_i = count_orbitals(ag_i)
    orb_j, ele_j = count_orbitals(ag_j)

    Hij = H_mat[:orb_i, orb_i:]
    Hii = H_mat[:orb_i, :orb_i]
    Sii = S_mat[:orb_i, :orb_i]
    Hjj = H_mat[orb_i:, orb_i:]
    Sjj = S_mat[orb_i:, orb_i:]

    return Hij, (Hii, Sii, ele_i), (Hjj, Sjj, ele_j)
