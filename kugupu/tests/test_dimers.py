from kugupu import dimers


def test_find_dimers(u):
    dims = dimers.find_dimers(u.atoms.fragments, 5.0)

    assert len(dims) == 2070
