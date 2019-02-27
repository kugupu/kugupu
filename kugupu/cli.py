import MDAnalysis as mda
import yaml

from . import coupling_matrix
from . import save_results

def make_Universe(topfile, trjfile):
    """Make a Universe and check it has required attributes

    Universes require:
      - bonds
      - masses
    """
    u = mda.Universe(topfile, trjfile)
    # todo, check attributes
    return u


def read_settings(fn):
    """Read yaml settings and check required fields exist"""
    with open(fn, 'r') as f:
        settings = yaml.load(f)
    # todo check keys exist and are valid
    return settings


def cli_generate_hamiltonians(topfile, trjfile, param_file, output):
    """Command line entry to Kugupu

    Parameters
    ----------
    topfile, trjfile : str
      inputs to MDAnalysis Universe
    param_file : str
      filename which holds run settings
    output : str
      where to output results to
    """
    # TODO:
    # check results file will be available before calculation

    u = make_Universe(topfile, trjfile)
    params = read_settings(param_file)

    hams = generate_H_frag_trajectory(u, **params)

    save_results(output, hams)
