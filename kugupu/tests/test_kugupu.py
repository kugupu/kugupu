"""
Unit and regression test for the kugupu package.
"""

# Import package, test suite, and other packages as needed
import kugupu
import pytest
import sys

from numpy.testing import assert_almost_equal


def test_kugupu_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "kugupu" in sys.modules


def test_u(u):
    assert len(u.atoms) > 1


def test_regression(u, ref_results):
    # overall regression test, very slow
    nn_cutoff = 5.0

    dimers = kugupu.dimers.find_dimers(u.atoms.fragments, nn_cutoff)

    H_orb, S_orb, fragsize = kugupu.yaehmop.run_all_dimers(u.atoms.fragments, dimers)

    H_frag, S_frag = kugupu.hamiltonian_reduce.calculate_H_frag(
        fragsize, H_orb, S_orb,
        degeneracy=1,
        state='lumo',
    )

    assert_almost_equal(H_frag, ref_results.H_frag[0], decimal=4)
    assert_almost_equal(S_frag, ref_results.S_frag[0], decimal=4)
