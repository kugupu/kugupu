import numpy as np
import os
import shutil
import pytest
import MDAnalysis as mda
from pkg_resources import resource_filename

import kugupu as kgp


DATA_DIR = resource_filename('kugupu', 'data')

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
    top = os.path.join(DATA_DIR, 'lammps.data.bz2')
    traj = os.path.join(DATA_DIR, 'run9last20.dcd')

    return top, traj


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
    top = os.path.join(DATA_DIR, 'mini.pdb')
    traj = os.path.join(DATA_DIR, 'mini.dcd')
    return top, traj

@pytest.fixture
def mini_u(mini_system):
    return mda.Universe(*mini_system)


@pytest.fixture
def mini_ix():
    # these are the fragment indices that make the mini system
    return [19, 70, 72, 150]


@pytest.fixture
def ref_results():
    return kgp.load_results(os.path.join(DATA_DIR, 'full_traj'))


@pytest.fixture
def ref_bind_inp():
    with open(os.path.join(DATA_DIR, 'ref_yaehmop_out', 'ref_0-7_bind.inp'),
              'r') as f:
        return f.read()

@pytest.fixture
def ref_bind_out(in_tmpdir):
    base = os.path.join(DATA_DIR, 'ref_yaehmop_out')
    for f in ['ref_0-7_bind.inp.out',
              'ref_0-7_bind.inp.OV',
              'ref_0-7_bind.inp.HAM']:
        shutil.copy2(os.path.join(base, f), f)
    return 'ref_0-7_bind.inp'


@pytest.fixture
def ref_yaehmop_results():
    return None
