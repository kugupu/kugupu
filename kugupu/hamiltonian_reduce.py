"""Converts Hamiltonian from orbital to fragment basis"""
import numpy as np
from scipy import linalg
from tqdm import tqdm

from . import logger


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
    degeneracy : int
      how many relevant MOs to consider (besides HOMO or LUMO)

    Returns
    -------
    e_frag : np.array
      eigenvalues of wavefunction
    v_frag : np.array
      eigenvectors of wavefunction
    """
    nfrags = len(starts)
    e_frag = np.zeros((nfrags * degeneracy))
    v_frag = np.zeros((H_orb.shape[0], nfrags * degeneracy),
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
        if state.lower() == 'homo':
            lo = homo - (degeneracy - 1)
            hi = homo
        elif state.lower() == 'lumo':
            lo = homo + 1
            hi = homo + 1 + (degeneracy - 1)

        # e - eigenvalues
        # v - eigenvectors
        # grab only (lo->hi) eigenvalues
        e, v = linalg.eigh(frag_H, frag_S, lower=False,
                           eigvals=(lo, hi))
        e_frag[frag*degeneracy : (frag+1)*degeneracy] = e
        v_frag[i:j, frag*degeneracy : (frag+1)*degeneracy] = v

    return e_frag, v_frag


def convert_to_fragment_basis(H_orb, S_orb, e_frag, v_frag):
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
    S_frag = np.zeros((n, n))

    H_frag[np.diag_indices_from(H_frag)] = e_frag
    S_frag[np.diag_indices_from(S_frag)] = 1

    # TODO: This probably becomes a single function call somehow
    for i in tqdm(range(n)):
        # technically we want:
        # v_frag[:, i].conj().T <dot> H <dot> v_frag[:, j]
        # H is a sparse matrix, therefore we want to use its methods!
        # because H is Hermitian: H.T is the conjugate
        # therefore np.vdot(v_frag[:, i], H) == H.T.dot(v_frag)
        # calculate and reuse this for all j iterations!
        H_pipo = H_orb.T.dot(v_frag[:, i])
        S_pipo = S_orb.T.dot(v_frag[:, i])
        for j in range(i+1, n):
            H_frag[i, j] = H_frag[j, i] = H_pipo.dot(v_frag[:, j])
            S_frag[i, j] = S_frag[j, i] = S_pipo.dot(v_frag[:, j])

    return H_frag, S_frag


def calculate_H_frag(fragsize, H_orb, S_orb, degeneracy, state):
    """Take orbital basis Hamiltonian and convert to fragment basis

    Parameters
    ----------
    fragsize : namedtuple
      info on fragment orbital size from tight binding calculation
    H_orb, S_orb : sparse matrix
      Hamiltonian and overlap matrices on orbital basis
    degeneracy : int
      how many orbitals deep to go
    state : str
      'HOMO' or 'LUMO'

    Returns
    -------
    H_frag, S_frag : numpy array
      intermolecular Hamiltonian and overlap matrices.
      Will have shape (nfrags * degeneracy, nfrag * degeneracy)
    """
    logger.info("Finding fragment eigenvalues")
    e_frag, v_frag = find_fragment_eigenvalues(H_orb, S_orb,
                                               fragsize.starts, fragsize.stops,
                                               fragsize.n_electrons,
                                               state, degeneracy)

    logger.info("Calculating fragment Hamiltonian matrix")
    H_frag, S_frag = convert_to_fragment_basis(H_orb, S_orb, e_frag, v_frag)

    return H_frag, S_frag
