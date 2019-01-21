"""runs the whole thing from head to toe"""

import yaml
import shutil
import numpy as np
import MDAnalysis as mda

from . import logger
from . import KugupuResults
from .dimers import find_dimers
from .yaehmop import run_all_dimers
from .hamiltonian_reduce import calculate_H_frag
from .networks import find_networks


def check_exists():
    """
    Checks if yaehmop executable is in the path
    """
    if shutil.which('yaehmop') is None:
        raise NameError("yaehmop executable not found!\n"
                        "add yaehmop path to your .bash_profile")


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
                    "".format(i, nframes))

        dimers = find_dimers(u.atoms.fragments, nn_cutoff)

        H_orb, S_orb, fragsize = run_all_dimers(u.atoms.fragments, dimers)

        if degeneracy is None:
            # first time through with auto degen
            H_frag, degeneracy = calculate_H_frag(fragsize, H_orb, S_orb,
                                                  state, degeneracy=None)
        else:
            H_frag = calculate_H_frag(fragsize, H_orb, S_orb,
                                      state, degeneracy)

        frames.append(ts.frame)
        Hs.append(H_frag)

    return KugupuResults(
        degeneracy=degeneracy,
        frames=np.array(frames),
        hamiltonian=np.stack(Hs),
    )


def cli_kugupu(dcdfile, topologyfile, param_file):
    """Command line entry to Kugupu

    Parameters
    ----------
    dcdfile, topologyfile : str
      inputs to MDAnalysis
    param_file : str
      filename which holds run settings
    """
    #creates universe object from trajectory
    u = make_universe(topologyfile, dcdfile)
    # returns parameter dictionary from parameter yaml file
    params = read_param_file(param_file)

    hams = generate_traj_H_frag(u, **params)

    # collects output from entire trajectory into a pandas dataframe
    dataframe = run_analysis(H_frag, networks)

    write_shizznizz(dataframe)
