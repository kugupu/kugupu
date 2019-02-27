"""runs the whole thing from head to toe"""

import numpy as np
from tqdm import tqdm

from . import logger
from . import KugupuResults
from .dimers import find_dimers
from ._yaehmop import run_dimer
from ._hamiltonian_reduce import find_psi


def _single_frame(fragments, nn_cutoff, degeneracy, state):
    """Results for a single frame

    Parameters
    ----------
    fragments : list of AtomGroup
      all fragments in system
    nn_cutoff : float
      distance for dimer pairs
    degeneracy : numpy array
      degenerate states per fragment
    state : str
      'homo' or 'lumo'

    Returns
    -------
    H_frag : numpy array
      coupling matrix
    """
    dimers = find_dimers(fragments, nn_cutoff)

    size = degeneracy.sum()
    H_frag = np.zeros((size, size))
    # start and stop indices for each fragment
    stops = np.cumsum(degeneracy)
    starts = np.r_[0, stops[:-1]]
    diag = np.arange(size)  # diagonal indices
    wave = dict()  # wavefunctions for each fragment

    for (i, j), ags in tqdm(sorted(dimers.items())):
        # indices for indexing H_frag for each fragment
        ix, iy = starts[i], stops[i]
        jx, jy = starts[j], stops[j]

        keep_i = not i in wave
        keep_j = not j in wave
        logger.debug('Calculating dimer {}-{}'.format(i, j))
        Hij, frag_i, frag_j = run_dimer(ags)
        if keep_i:
            # If we didn't have fragment i already done,
            # calculate the wavefunction and state energy
            e_i, psi_i = find_psi(frag_i[0], frag_i[1], frag_i[2],
                                  state, degeneracy[i])
            # fill diagonal with energy of states
            H_frag[diag[ix:iy], diag[ix:iy]] = e_i
            # store the wavefunction for future use
            wave[i] = psi_i
        else:
            # If we already did fragment i, just retrieve psi
            psi_i = wave[i]

        if keep_j:
            e_j, psi_j = find_psi(frag_j[0], frag_j[1], frag_j[2],
                                  state, degeneracy[j])
            H_frag[diag[jx:jy], diag[jx:jy]] = e_j
            wave[j] = psi_j
        else:
            psi_j = wave[j]

        # H = <psi_i|Hij|psi_j>
        H_frag[ix:iy, jx:jy] = abs(psi_i.T.dot(Hij).dot(psi_j))
        H_frag[jx:jy, ix:iy] = H_frag[ix:iy, jx:jy]
    # do single fragment calculations for all missing
    for i in (set(range(len(degeneracy))) - set(wave.keys())):
        ix, iy = starts[i], stops[i]
        logger.debug('Calculating lone fragment {}'.format(index))
        H, S, ele = run_fragment(fragments[i])

        e_i, psi_i = find_psi(H, S, ele, state, degeneracy[i])

        H_frag[diag[ix:iy], diag[ix:iy]] = e_i
        # don't need to save the psi for this fragment
        #wave[i] = psi_i

    return H_frag


def coupling_matrix(u, nn_cutoff, state, degeneracy=None,
                    start=None, stop=None, step=None):
    """Generate Hamiltonian matrix H_frag for each frame in trajectory

    Parameters
    ----------
    u : mda.Universe
      Universe to analyse
    nn_cutoff : float
      maximum distance between dimers to consider neighbours
    degeneracy : int or dict or None
      number of orbitals deep to go; can be an integer for homo
    state : str
      'HOMO' or 'LUMO'
    start, stop, step : int, optional
      slice through Universe trajectory

    Returns
    -------
    hams : KugupuResults namedtuple
      coupling matrix for each frame
      with shape (nframes, nfrags * degeneracy, nfrags * degeneracy)
    """
    Hs, frames = [], []

    nframes = len(u.trajectory[start:stop:step])
    logger.info("Processing {} frames".format(nframes))

    if degeneracy is not None:
        # we need to pass a vector n_frags long
        if isinstance(degeneracy, int):
            # if only one value is given the elements are all the same
            degeneracy = np.full(len(u.atoms.fragments), degeneracy)
        elif isinstance(degeneracy, dict):
            # if our system is multi-component,
            # different residues have different degeneracy
            deg_arr = np.zeros(len(u.atoms.fragments))
            for i, frag in enumerate(u.atoms.fragments):
                # for a molecule with more than 1 residue,
                #  only the 1st one is checked
                deg_arr[i] = degeneracy[frag.residues[0].resname]
            degeneracy = deg_arr

    for i, ts in enumerate(u.trajectory[start:stop:step]):
        logger.info("Processing frame {} of {}"
                    "".format(i + 1, nframes))
        H_frag = _single_frame(u.atoms.fragments, nn_cutoff, degeneracy, state)

        frames.append(ts.frame)
        Hs.append(H_frag)

    logger.info('Done!')
    return KugupuResults(
        frames=np.array(frames),
        H_frag=np.stack(Hs),
        degeneracy=degeneracy,
    )
