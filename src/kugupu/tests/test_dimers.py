import itertools
import MDAnalysis as mda
from kugupu import dimers
import pytest

def test_find_dimers(u):
    dims = dimers.find_dimers(u.atoms.fragments, 5.0)

    assert len(dims) == 2070


@pytest.mark.slow
def test_find_dimers_thorough(u):
    frags = u.atoms.fragments

    dims = dimers.find_dimers(frags, 5.0)
    ref = []
    for i, j in itertools.combinations(range(len(frags)), 2):
        frag_i, frag_j = frags[i], frags[j]

        idx = mda.lib.distances.capped_distance(
            frag_i.positions,
            frag_j.positions,
            box=u.dimensions,
            max_cutoff=5.0,
            return_distances=False)
        if len(idx):
            ref.append((i, j))

    assert len(ref) == len(dims)
    for val in ref:
        assert val in dims
