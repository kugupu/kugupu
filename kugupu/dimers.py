import numpy as np
from MDAnalysis.lib import distances

from . import logger


def _find_contacts(fragments, cutoff):
    """Raw version to return indices of touching fragments

    Parameters
    ----------
    fragments : list of AtomGroup
      molecules to consider
    cutoff : float
      threshold for touching or not

    Returns
    -------
    frag_idx : numpy array, shape (n, 2)
      indices of fragments that are touching, e.g. [[0, 1], [2, 3], ...]
    """
    # indices of atoms within cutoff of each other
    idx = distances.self_capped_distance(sum(fragments).positions,
                                         max_cutoff=cutoff,
                                         box=fragments[0].dimensions,
                                         return_distances=False)
    nfrags = len(fragments)
    fragsizes = [len(f) for f in fragments]
    # translation array from atom index to fragment index
    translation = np.repeat(np.arange(nfrags), fragsizes)
    # this array now holds pairs of fragment indices
    fragidx = translation[idx]
    # remove self contributions (i==j) and don't double count (i<j)
    fragidx = fragidx[fragidx[:, 0] < fragidx[:, 1]]

    return fragidx


def find_dimers(fragments, cutoff):
    """Calculate dimers to run

    Parameters
    ----------
    fragments : list of AtomGroups
      list of all fragments in system.  Must all be centered in box and
      unwrapped
    cutoff : float
      maximum distance allowed between fragments to be considered
      a dimer

    Returns
    -------
    dimers : dictionary
      mapping of {(x, y): (ag_x, ag_y)} for all dimer pairs
    """
    logger.info("Finding dimers within {}, passed {} fragments"
                "".format(cutoff, len(fragments)))
    fragidx = _find_contacts(fragments, cutoff)

    dimers = {(i, j): (fragments[i], fragments[j])
              for i, j in fragidx}

    logger.info("Found {} dimers".format(len(dimers)))

    return dimers


def contact_matrix(u, frags, nn_cutoff, start=None, stop=None, step=None):
    """Calculate a contact adjacency matrix

    Parameters
    ----------
    u : mda.Universe
      the system
    frags : list
      list of fragments to consider
    nn_cutoff : float
      distance at which to consider two fragments to be in contact
    start, stop, step : int, optional
      control which frames are analysed

    Returns
    -------
    contacts : numpy array, shape (nframes, nfrags, nfrags)
      binary array with contacts marked as 1.  Self contributions
    """
    output = []

    ag = sum(frags)
    nfrags = len(frags)

    for ts in u.trajectory[start:stop:step]:
        adj = np.zeros((nfrags, nfrags), dtype=np.int)

        contacts = _find_contacts(frags, nn_cutoff)

        adj[contacts[:, 0], contacts[:, 1]] = 1
        adj[contacts[:, 1], contacts[:, 0]] = 1

        output.append(adj)

    return np.stack(output)
