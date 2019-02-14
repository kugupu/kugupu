import numpy as np
from MDAnalysis.lib import distances

from . import logger


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
    dimers = {}

    nfrags = len(fragments)
    # indices of atoms within cutoff of each other
    idx = distances.self_capped_distance(sum(fragments).positions,
                                         max_cutoff=cutoff,
                                         box=fragments[0].dimensions,
                                         return_distances=False)
    fragsizes = [len(f) for f in fragments]
    # translation array from atom index to fragment index
    translation = np.repeat(np.arange(nfrags), fragsizes)
    # this array now holds pairs of fragment indices
    fragidx = translation[idx]
    # remove self contributions (i==j) and don't double count (i<j)
    fragidx = fragidx[fragidx[:, 0] < fragidx[:, 1]]

    dimers = {(i, j): (fragments[i], fragments[j])
              for i, j in fragidx}

    logger.info("Found {} dimers".format(len(dimers)))

    return dimers
