import itertools
from MDAnalysis.lib import distances
from tqdm import tqdm

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
    npairs = nfrags * (nfrags - 1) // 2
    for i, j in tqdm(itertools.combinations(range(nfrags), 2),
                     total=npairs):
        frag_i, frag_j = fragments[i], fragments[j]

        da = distances.distance_array(frag_i.positions,
                                      frag_j.positions,
                                      frag_i.dimensions)
        if (da < cutoff).any():
            dimers[(i, j)] = (frag_i, frag_j)

        #da = distances.capped_distance(frag_i.positions,
        #                               frag_j.positions,
        #                               max_cutoff=cutoff,
        #                               return_distances=False)
        #if len(da):
        #    dimers[(i, j)] = (frag_i, frag_j)

    logger.info("Found {} dimers".format(len(dimers)))

    return dimers
