"""runs the whole thing from head to toe"""

import yaml
import numpy as np
import MDAnalysis as mda
from tqdm import tqdm

from . import logger
from . import KugupuResults
from .dimers import find_dimers
from . import yaehmop
from . import hamiltonian_reduce
from .networks import find_networks


def make_universe(topologyfile, dcdfile):
    universe = mda.Universe(topologyfile, dcdfile)

    if not hasattr(universe.atoms, 'bonds'):
        universe.atoms.guess_bonds()

    if not hasattr(universe.atoms, 'names'):
        universe.add_TopologyAttr('names')
        namedict = { 1.008: 'H', 12.011: 'C', 14.007: 'N', 15.999: 'O', 32.06 : 'S', 18.99800: 'F' }
         #TODO: add fluorine at least
        for m, n in namedict.items():
            universe.atoms[universe.atoms.masses == m].names = n

    return universe


def read_param_file(param_file):
    """
    Reads the yaml parameter file

    Parameters
    ----------
    param_file : yaml file
    this should contain:
     state : string
      homo or lumo
     V_cutoff : list of reals
      energy thresholds in eV
     nn_cutoff : real
      cutoff for nearest neighbor search in Å
     degeneracy : string
      how many frontier orbitals to consider (including homo/lumo)
    """
    params = yaml.load(open(param_file))

    return params

find_psi = hamiltonian_reduce.find_fragment_eigenvalue


def _single_frame(dimers, degeneracy, state):
    """Results for a single frame

    """
    size = degeneracy.sum()
    H_frag = np.zeros((size, size))
    wave = dict()  # wavefunctions for each fragment

    for (i, j), ags in tqdm(sorted(dimers.items())):
        keep_i = not i in wave
        keep_j = not j in wave
        logger.debug('Calculating dimer {}-{}'.format(i, j))
        Hij, H, S, eles = yaehmop.run_dimer(ags)
        if keep_i:
            e_i, psi_i = find_psi(H[0], S[0], eles[0],
                                  state, degeneracy[i])
            H_frag[i, i] = e_i
            wave[i] = psi_i
        else:
            psi_i = wave[i]

        if keep_j:
            e_j, psi_j = find_psi(H[1], S[1], eles[1],
                                  state, degeneracy[j])
            H_frag[j, j] = e_j
            wave[j] = psi_j
        else:
            psi_j = wave[j]

        # H = <psi_i|Hij|psi_j>
        H_frag[i, j] = H_frag[j, i] = abs(psi_i.T.dot(Hij).dot(psi_j))

    return H_frag

def generate_H_frag_trajectory(u, nn_cutoff, state, degeneracy=None,
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
      full Hamiltonian matrix and overlap matrix for each frame.
      The Hamiltonian and overlap matrices will have shape
      (nframes, nfrags, nfrags)
    """
    Hs, Ss, frames = [], [], []

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

        dimers = find_dimers(u.atoms.fragments, nn_cutoff)

        H_frag = _single_frame(dimers, degeneracy, state)

        frames.append(ts.frame)
        Hs.append(H_frag)

    logger.info('Done!')
    return KugupuResults(
        frames=np.array(frames),
        H_frag=np.stack(Hs),
        degeneracy=degeneracy,
    )
