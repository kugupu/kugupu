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


def find_fragment_eigenvalues_auto_degen(H_orb, S_orb, starts, stops,
                                         n_electrons, state,
                                         max_degeneracy=None,
                                         degeneracy_tolerance=None):
    """Same as find_fragment_eigenvalues but detects degeneracy

    Parameters
    ----------
    H_orb, S_orb : scipy.sparse matrix
      Hamiltonian and Overlap matrix for all orbitals
    starts, stops : array
      indices to find fragment values in matrices, shape (nfrags,)
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
    nfrags = len(starts)
    if max_degeneracy is None:
        max_degeneracy = MAX_DEGEN
    if degeneracy_tolerance is None:
        degeneracy_tolerance = DEGEN_TOL

    # we don't know how big these will be since we allow degeneracy to
    # fluctuate, so for now e_frag, v_frag are created as lists
    e_frag = []
    v_frag = []
    degeneracy = np.zeros(nfrags, dtype=int)

    for frag, (i, j) in tqdm(enumerate(zip(starts, stops)), total=nfrags):
        if not (j - i):
            # if fragment had no dimer pairing, we never ran it
            # therefore there is no data for this fragment
            continue

        # grab section relevant to fragment *frag*
        frag_H = H_orb[i:j, i:j].real.todense()
        frag_S = S_orb[i:j, i:j].real.todense()

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
        for e_val, v_val in zip(e, v.T):
            if abs(e_val - e0) < degeneracy_tolerance:
                e_frag.append(e_val)
                v_frag.append(v_val)
                degeneracy[frag] += 1

    # reconstruct v_frag
    v_frag_arr = np.zeros((H_orb.shape[0], degeneracy.sum()),
                          dtype='complex128')
    degen_counter = 0
    for frag, (i, j) in enumerate(zip(starts, stops)):
        if not (j - i):
            continue
        for d in range(degeneracy[frag]):
            v_frag_arr[i:j, degen_counter] = v_frag[degen_counter]
            degen_counter += 1

    e_frag = np.array(e_frag)

    return e_frag, v_frag_arr, degeneracy


def find_fragment_eigenvalues(H_orb, S_orb, starts, stops, n_electrons, state,
                              degeneracy):
    """Find eigenvalues and vectors for each fragments orbitals

    Parameters
    ----------
    H_orb, S_orb : scipy.sparse matrix
      Hamiltonian and Overlap matrix for all orbitals
    starts, stops : array
      indices to find fragment values in matrices, shape (nfrags,)
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
    v_frag : np.array
      eigenvectors of wavefunction
    """
    nfrags = len(starts)

    degen_counter = 0
    e_frag = np.zeros(degeneracy.sum())
    v_frag = np.zeros((H_orb.shape[0], degeneracy.sum()),
                      dtype='complex128')

    for frag, (i, j) in tqdm(enumerate(zip(starts, stops)), total=nfrags):
        if not (j - i):
            # if fragment had no dimer pairing, we never ran it
            # therefore there is no data for this fragment
            continue

        # grab section relevant to fragment *frag*
        frag_H = H_orb[i:j, i:j].real.todense()
        frag_S = S_orb[i:j, i:j].real.todense()

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
        v_frag[i:j, degen_counter:degen_counter + degeneracy[frag]] = v
        degen_counter += degeneracy[frag]

    return e_frag, v_frag


def convert_to_fragment_basis(H_orb, e_frag, v_frag):
    """Find Hamiltonian on fragment basis

    Parameters
    ----------
    H_orb, S_orb : sparse matrices
      Hamiltonian, Overlap for all orbital basis
    e_frag, v_frag : array
      eigenvalues and eigenvectors of wavefunction

    Returns
    -------
    H_frag, S_frag : np array
      Hamiltonian and Overlap on the basis of fragments
      Will be square with size (nfrags * degeneracy, nfrags * degeneracy)
    """
    n = e_frag.shape[0]  # number of frags * degeneracy
    H_frag = np.zeros((n, n))
    #S_frag = np.zeros((n, n))

    H_frag[np.diag_indices_from(H_frag)] = e_frag
    #S_frag[np.diag_indices_from(S_frag)] = 1

    # TODO: This probably becomes a single function call somehow
    for i in tqdm(range(n)):
        # technically we want:
        # v_frag[:, i].conj().T <dot> H <dot> v_frag[:, j]
        # H is a sparse matrix, therefore we want to use its methods!
        # because H is Hermitian: H.T is the conjugate
        # therefore np.vdot(v_frag[:, i], H) == H.T.dot(v_frag)
        # calculate and reuse this for all j iterations!
        H_pipo = H_orb.T.dot(v_frag[:, i])
        #S_pipo = S_orb.T.dot(v_frag[:, i])
        for j in range(i+1, n):
            H_frag[i, j] = H_frag[j, i] = H_pipo.dot(v_frag[:, j])
            #S_frag[i, j] = S_frag[j, i] = S_pipo.dot(v_frag[:, j])

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


def calculate_H_frag(fragsize, H_orb, S_orb, state, degeneracy=None):
    """Take orbital basis Hamiltonian and convert to fragment basis

    Parameters
    ----------
    fragsize : namedtuple
      info on fragment orbital size from tight binding calculation
    H_orb, S_orb : sparse matrix
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
        fragsize.starts, fragsize.stops,
        fragsize.n_electrons,
        state, degeneracy
    )

    logger.info("Calculating fragment Hamiltonian matrix")
    H_frag = convert_to_fragment_basis(H_orb, e_frag, v_frag)
    #TOFINISH
    #pass nfrags from somewhere
    #allow for fragments to have different degeneracy
    #Hij_eff =  squish_Hij(H_frag, nfrags, degeneracy)
    #it will return Heff once i'm done

    if auto_degen:
        return H_frag, degeneracy
    else:
        return H_frag
