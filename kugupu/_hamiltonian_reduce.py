"""Converts Hamiltonian from orbital to fragment basis"""
import numpy as np
from scipy import linalg
from tqdm import tqdm

from . import logger

# maximum degenerate states to find
MAX_DEGEN = 20
# maximum energy different between degenerate states
# in eV
DEGEN_TOL = 0.02


def find_psi(H, S, n_electrons, state, degeneracy):
    """Find wavefunction for single fragment

    Parameters
    ----------
    H, S : numpy array
      Hamiltonian and Overlap matrix for fragment
    n_electrons : int
      number of valence electrons
    state : str
      'homo' or 'lumo'
    degeneracy : int
      number of degenerate states to consider

    Returns
    -------
    energy : float
    wavefunction : numpy array
    """
    homo = int(n_electrons // 2) - 1
    if state.lower() == 'homo':
        lo = homo - (degeneracy - 1)
        hi = homo
    elif state.lower() == 'lumo':
        lo = homo + 1
        hi = homo + 1 + (degeneracy - 1)

    # e - eigenvalues
    # v - eigenvectors
    # grab only (lo->hi) eigenvalues
    e, v = linalg.eigh(H, S, lower=False,
                       eigvals=(lo, hi))

    return e, v


def find_fragment_eigenvalues_auto_degen(H_orb, S_orb,
                                         n_electrons, state,
                                         max_degeneracy=None,
                                         degeneracy_tolerance=None):
    """Same as find_fragment_eigenvalues but detects degeneracy

    Parameters
    ----------
    H_orb, S_orb : scipy.sparse matrix
      Hamiltonian and Overlap matrix for all orbitals
    n_electrons : array
      number of electrons in each fragment, shape (nfrags,)
    state : str
      either 'homo' or 'lumo'
    max_degeneracy : int
      number of states to check (Default 20)
    degeneracy_tolerance : float
      difference in energy values from first and nth state
      to consider them degenerate (Default 0.02)

    Returns
    -------
    e_frag : np.array
      eigenvalues of wavefunction
    v_frag : np.array
      eigenvectors of wavefunction
    degeneracy : np.array
      degeneracy per fragment
    """
    nfrags = len(n_electrons)
    if max_degeneracy is None:
        max_degeneracy = MAX_DEGEN
    if degeneracy_tolerance is None:
        degeneracy_tolerance = DEGEN_TOL

    # we don't know how big these will be since we allow degeneracy to
    # fluctuate, so for now e_frag is created as a list
    e_frag = []
    v_frag = dict()
    degeneracy = np.zeros(nfrags, dtype=int)

    for frag in tqdm(range(nfrags)):
        # grab section relevant to fragment *frag*
        frag_H = H_orb[frag, frag]
        frag_S = S_orb[frag, frag]

        # figure out which eigenvalues we want
        homo = int(n_electrons[frag] / 2) - 1

        # we start by grabbing a bunch of
        # orbitals - this is defined by MAX_DEGEN
        if state.lower() == 'homo':
            lo = homo - (MAX_DEGEN - 1)
            hi = homo
        elif state.lower() == 'lumo':
            lo = homo + 1
            hi = homo + 1 + (MAX_DEGEN - 1)

        # e - eigenvalues
        # v - eigenvectors
        # grab only (lo->hi) eigenvalues
        e, v = linalg.eigh(frag_H, frag_S, lower=False,
                           eigvals=(lo, hi))

        # eigenvalues (orbital energies) are sorted
        if state.lower() == 'homo':
            # if we're doing HOMOs, last value is the HOMO
            # so this is the reference value
            e0 = e[-1]
        else:
            # if LUMO then the first value is the reference value
            e0 = e[0]
        # iterate over eigenvalues/energies
        # taking whilst less than TOL away
        for e_val in e:
            if abs(e_val - e0) < degeneracy_tolerance:
                e_frag.append(e_val)
                degeneracy[frag] += 1
        v_frag[frag] = v[:, :degeneracy[frag]]

    e_frag = np.array(e_frag)

    return e_frag, v_frag_arr, degeneracy


def find_fragment_eigenvalues(H_orb, S_orb, n_electrons, state,
                              degeneracy):
    """Find eigenvalues and vectors for each fragments orbitals

    Parameters
    ----------
    H_orb, S_orb : dict of numpy arrays
      Hamiltonian and Overlap matrix for all orbitals
    n_electrons : array
      number of electrons in each fragment, shape (nfrags,)
    state : str
      either 'homo' or 'lumo'
    degeneracy : numpy array
      how many relevant MOs to consider (besides HOMO or LUMO)

    Returns
    -------
    e_frag : np.array
      eigenvalues of wavefunction
    v_frag : dict
      mapping of fragment index to eigenvectors of wavefunction
    """
    nfrags = len(n_electrons)

    degen_counter = 0
    e_frag = np.zeros(degeneracy.sum())
    v_frag = dict()

    for frag in tqdm(range(nfrags)):
        # grab section relevant to fragment *frag*
        frag_H = H_orb[frag, frag]
        frag_S = S_orb[frag, frag]

        # figure out which eigenvalues we want
        homo = int(n_electrons[frag] / 2) - 1

        deg = degeneracy[frag]
        if state.lower() == 'homo':
            lo = homo - (deg - 1)
            hi = homo
        elif state.lower() == 'lumo':
            lo = homo + 1
            hi = homo + 1 + (deg - 1)

        # e - eigenvalues
        # v - eigenvectors
        # grab only (lo->hi) eigenvalues
        e, v = linalg.eigh(frag_H, frag_S, lower=False,
                       eigvals=(lo, hi))

        e_frag[degen_counter:degen_counter + degeneracy[frag]] = e
        v_frag[frag] = v.real
        degen_counter += degeneracy[frag]

    return e_frag, v_frag


def convert_to_fragment_basis(dimers, degeneracy, H_orb, e_frag, v_frag):
    """Find Hamiltonian on fragment basis

    Parameters
    ----------
    dimers : dict
      which dimers were ran
    degneracy : numpy array
      for each fragment, its
    H_orb : dict of sparse matrices
      Hamiltonian for all orbital basis
    e_frag : array
      eigenvalues of wavefunction
    v_frag : dict
      eigenvectors of wavefunction

    Returns
    -------
    H_frag : numpy array
      coupling matrix on the basis of fragments
      Will be square with size (nfrags * degeneracy, nfrags * degeneracy)
    """
    # arrays for indexing H_frag
    stops = np.cumsum(degeneracy)
    starts = np.r_[0, stops[:-1]]

    n = e_frag.shape[0]  # number of frags * degeneracy
    H_frag = np.zeros((n, n))
    H_frag[np.diag_indices_from(H_frag)] = e_frag

    # TODO: This probably becomes a single function call somehow
    for x, y in dimers:
        # < psi_x | H_xy | psi_y >
        # place H_xy first as sparse matrix
        H_pipo = np.abs(H_orb[x, y].T.dot(v_frag[x]).T.dot(v_frag[y]))
        H_frag[starts[x]:stops[x], starts[y]:stops[y]] = H_pipo
        H_frag[starts[y]:stops[y], starts[x]:stops[x]] = H_pipo.T

    return H_frag


def squish_Hij(H_frag, d, n_frag):
    """
    Calculate an effective coupling J_eff
    as in J. Mater. Chem. C, 2016, 4, 3747

    J_eff =  sqrt( sum J**2)/degeneracy

    this reduces the size of Hij from (nfrags * degeneracy, nfrags * degeneracy)
    to (nfrags * nfrags)
    """
    Hij_eff = np.zeros((n_frag,n_frag))
    for i in range(n_frag):
        for j in range(n_frag):
            Hij_eff[i,j] = np.sqrt(np.sum(H_frag[d*i:d*(i+1),d*j:d*(j+1)]**2)/d)
    return Hij_eff


def calculate_H_frag(dimers, fragsize, H_orb, S_orb, state, degeneracy=None):
    """Take orbital basis Hamiltonian and convert to fragment basis

    Parameters
    ----------
    dimers : dict
      mapping of fragment indices to fragment AtomGroups
    fragsize : namedtuple
      info on fragment orbital size from tight binding calculation
    H_orb, S_orb : dicts of sparse matrices
      Hamiltonian and overlap matrices on orbital basis
    degeneracy : np.ndarray or None
      how many orbitals deep to go for each molecule.
      If None, this will be automatically detected and returned
    state : str
      'HOMO' or 'LUMO'

    Returns
    -------
    H_frag : numpy array
      intermolecular Hamiltonian
      Will have shape (nfrags * nfrags)
    degeneracy : numpy array
      if degeneracy was None, the automatically calculated degeneracy per fragment
    """
    auto_degen = degeneracy is None
    if auto_degen:
        logger.info("Determining degeneracy")
        e_frag, v_frag, degeneracy = find_fragment_eigenvalues_auto_degen(
            H_orb, S_orb,
            fragsize.starts, fragsize.stops,
            fragsize.n_electrons,
            state,
        )
    logger.info("Finding fragment eigenvalues")

    e_frag, v_frag = find_fragment_eigenvalues(
        H_orb, S_orb,
        fragsize.n_electrons,
        state, degeneracy
    )

    logger.info("Calculating fragment Hamiltonian matrix")
    H_frag = convert_to_fragment_basis(dimers, degeneracy, H_orb, e_frag, v_frag)
    #TOFINISH
    #pass nfrags from somewhere
    #allow for fragments to have different degeneracy
    #Hij_eff =  squish_Hij(H_frag, nfrags, degeneracy)
    #it will return Heff once i'm done

    if auto_degen:
        return H_frag, degeneracy
    else:
        return H_frag
