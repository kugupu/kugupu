"""Tests for yaehmop functionality


"""
from kugupu import _yaehmop as yaehmop
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import os


def test_molecule_shifting(u):
    """Check that molecules are correctly shifted into adjacent images"""
    frags = u.atoms.fragments
    # these fragments are in contact, but only via PBC
    frag_i, frag_j = frags[0], frags[9]

    ref_i = frag_i.positions.copy()
    ref_j = frag_j.positions.copy()

    coords = yaehmop.shift_dimer_images(frag_i, frag_j)

    # check we didn't alter positions
    assert_equal(ref_i, frag_i.positions)
    assert_equal(ref_j, frag_j.positions)

    # the desired shift
    # this has been visually checked to be correct
    des_shift = np.array([0., u.dimensions[1], 0.])
    # frag_i positions should have been written unchanged
    assert_almost_equal(coords[:len(frag_i)], frag_i.positions)
    assert_almost_equal(coords[len(frag_i):], frag_j.positions - des_shift)

