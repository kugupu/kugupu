"""Tests for yaehmop functionality


"""
from kugupu import yaehmop
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


def test_create_input(u, ref_bind_inp):
    # regression test for bind input for frags(0, 7)

    f1, f2 = u.atoms.fragments[0], u.atoms.fragments[7]
    pos = yaehmop.shift_dimer_images(f1, f2)
    inp = yaehmop.create_bind_input(ag=sum(f1, f2),
                                    pos=pos,
                                    name='{}-{}'.format(0, 7))

    assert inp == ref_bind_inp


def test_run_yaehmop(in_tmpdir, ref_bind_inp):
    yaehmop.run_bind('inp.bind', ref_bind_inp)

    assert os.path.exists('inp.bind.out')
    assert os.path.exists('inp.bind.OV')
    assert os.path.exists('inp.bind.HAM')


def test_parse_yaehmop(u, ref_bind_out, ref_yaehmop_results):
    ag1, ag2 = u.atoms.fragments[0], u.atoms.fragments[7]

    H, S, orb, ele = yaehmop.parse_yaehmop_out(ref_bind_out,
                                               (0, 7), ag1, ag2,
                                               keep_i=True, keep_j=True)
    for k in [(0, 0), (0, 7), (7, 7)]:
        assert k in H
        assert k in S
    assert orb[0] == 498
    assert orb[7] == 498
    assert ele[0] == 514
    assert ele[7] == 514
