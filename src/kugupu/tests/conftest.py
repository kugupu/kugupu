import os
import pytest
import MDAnalysis as mda
from importlib.resources import as_file, files

import kugupu as kgp


@pytest.fixture
def in_tmpdir(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(cwd)


@pytest.fixture
def system():
    top_src = files('kugupu.data').joinpath('lammps.data.bz2')
    traj_src = files('kugupu.data').joinpath('run9last20.dcd')

    with as_file(top_src) as top, as_file(traj_src) as traj:
        return str(top), str(traj)


@pytest.fixture
def u(system):
    sys = mda.Universe(*system)

    sys.add_TopologyAttr('names')

    namedict = {
        1.008: 'H',
        12.011: 'C',
        14.007: 'N',
        15.999: 'O',
        32.06: 'S',
    }

    for m, n in namedict.items():
        sys.atoms[sys.atoms.masses == m].names = n

    return sys


@pytest.fixture
def mini_system():
    top_src = files('kugupu.data').joinpath('mini.pdb')
    traj_src = files('kugupu.data').joinpath('mini.dcd')

    with as_file(top_src) as top, as_file(traj_src) as traj:
        return str(top), str(traj)


@pytest.fixture
def mini_u(mini_system):
    return mda.Universe(*mini_system)


@pytest.fixture
def mini_ix():
    # these are the fragment indices that make the mini system
    return [19, 70, 72, 150]


@pytest.fixture
def results_file():
    with as_file(files('kugupu.data').joinpath('full_traj.hdf5')) as f:
        return str(f)


@pytest.fixture
def ref_results(results_file):
    return kgp.load_results(results_file)


