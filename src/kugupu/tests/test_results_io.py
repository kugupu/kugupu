"""Tests for loading and saving of Kugupu results files"""
import h5py
import kugupu as kgp
import numpy as np
import os

from numpy.testing import assert_almost_equal, assert_equal


def test_load_results(results_file):
    r = kgp.load_results(results_file)

    assert hasattr(r, 'H_frag')
    assert r.H_frag.shape == (20, 200, 200)
    assert r.H_frag.dtype == np.float64
    assert hasattr(r, 'degeneracy')
    assert r.degeneracy.shape == (200,)
    assert r.degeneracy.dtype == np.int64
    assert hasattr(r, 'frames')
    assert r.frames.shape == (20,)
    assert r.frames.dtype == np.int64

def test_metadata(results_file):
    # check invisible metadata used internally
    with h5py.File(results_file) as f:
        assert 'kugupu_version' in f.attrs
        assert 'creation_date' in f.attrs


def test_save_results(in_tmpdir):
    H_frag = np.random.random((5, 50, 50))
    deg = np.ones(50, dtype=int)
    frames = np.arange(5)

    res = kgp.KugupuResults(frames=frames,
                            H_frag=H_frag,
                            degeneracy=deg)

    kgp.save_results('test', res)

    assert os.path.exists('test.hdf5')

    res2 = kgp.load_results('test')

    assert_almost_equal(res.H_frag, res2.H_frag)
    assert_equal(res.degeneracy, res2.degeneracy)
    assert_equal(res.frames, res2.frames)
