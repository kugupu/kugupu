"""
Unit and regression test for the kugupu package.
"""

# Import package, test suite, and other packages as needed
import kugupu as kgp
import numpy as np
import pytest
import sys

from numpy.testing import assert_almost_equal


def test_kugupu_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "kugupu" in sys.modules

def test_struct_sizes():
    assert kgp._yaehmop._pyeht.check_struct_sizes()


def test_mini_regression(mini_u, mini_ix, ref_results):
    # slice to grab mini results from 200 frag matrix
    ix = np.ix_(mini_ix, mini_ix)
    results = kgp.coupling_matrix(mini_u,
                                  nn_cutoff=5.0, degeneracy=1,
                                  state='lumo')

    for H_ref, H_new in zip(ref_results.H_frag,
                            results.H_frag):
        assert_almost_equal(abs(H_ref[ix]), abs(H_new), decimal=3)

@pytest.mark.parametrize('start,stop,step', [
    (None, 5, None),
    (15, None, None),
    (None, None, 5),
    (10, 16, 2),
])
def test_slicing_options(mini_u, mini_ix, ref_results, start, stop, step):
    ix = np.ix_(mini_ix, mini_ix)

    results = kgp.coupling_matrix(mini_u,
                                  nn_cutoff=5.0, degeneracy=1,
                                  state='lumo',
                                  start=start, stop=stop, step=step)

    assert results.H_frag.shape[0] == ref_results.H_frag[slice(start, stop, step)].shape[0]
    for H_ref, H_new in zip(ref_results.H_frag[slice(start, stop, step)],
                            results.H_frag):
        assert_almost_equal(abs(H_ref[ix]), abs(H_new))
